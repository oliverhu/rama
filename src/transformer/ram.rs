use std::{io::BufReader, fs::File};
use crate::device::cpu::CPU;
use super::{hbm::TransformerWeightsGPU, state::{RunState, RunStateView}, Config, Storage, Transformer, View};
use cudarc::driver::CudaSlice;
use rand_chacha::ChaCha20Rng;
use rand::{Rng, SeedableRng};
use crate::utils::read::*;
pub struct TransformerCPU {
    pub config: Config,
    pub weights: TransformerWeights,
    pub state: RunState<Vec<f32>>,
    pub device: CPU
}

pub fn forward<'a, T>(cfg: &Config, wv: &TransformerWeightsView<'a, Vec<f32>>,
                   rsv: &mut RunStateView<'a, Vec<f32>>, token: usize, pos: usize,
                   device: &CPU
                ) {
    let dim = cfg.dim;

    let hidden_dim = cfg.hidden_dim;
    let head_size = dim / cfg.n_heads;

    // Load the embedding into x.
    device.copy_from_slice(&mut rsv.x, &wv.token_embedding_table.slice((token * cfg.dim)..((token + 1) * cfg.dim)));

    // Positional encoding
    let pos_real = wv.freq_cis_real.slice(pos*(head_size/2)..(pos+1)*(head_size/2));
    let pos_img = wv.freq_cis_imag.slice(pos*(head_size/2)..(pos+1)*(head_size/2));

    // Forward through the layers.
    for layer in 0..cfg.n_layers {

        // Apply layer normalization. a.k.a we normalize by data items (rows).
        // In comparison, batch normalization normalizes by features (columns);
        // Since we don't need to recenter but only rescaling, we use RMSNorm than
        // actual layer norm, and it worked well in practice.
        device.rmsnorm(&mut rsv.xb, &rsv.x.as_view(), &wv.rms_att_weight.slice(layer * dim..(layer + 1) * dim), dim);
        // self.device.rmsnorm(rsv.xb, rsv.x.as_view(), wv.rms_att_weight.slice(layer * dim..(layer + 1) * dim), dim);

        // Calculate Q, K, V
        device.matmul_1d(&mut rsv.q, &wv.wq.slice(layer * dim * dim..(layer + 1) * dim * dim),&rsv.xb.as_view(), dim);
        device.matmul_1d(&mut rsv.k, &wv.wk.slice(layer * dim * dim..(layer + 1) * dim * dim),&rsv.xb.as_view(), dim);
        device.matmul_1d(&mut rsv.v, &wv.wv.slice(layer * dim * dim..(layer + 1) * dim * dim),&rsv.xb.as_view(), dim);

        // RoPE relative positional encoding. https://arxiv.org/pdf/2104.09864.pdf
        // b/c in attention we only care about the distance between two words, we use relative attention.
        // Rotary position embeddings are only applied to the Q & Ks, not the values,
        // also, they are applied after the multi with weights. (The vanilla transformer is different,
        // positional embeddings were applied before multiplication with weights.)
        for h in 0..cfg.n_heads {

            // let q = &gpu_state.q.slice(h * head_size..);
            // let k = &gpu_state.k.slice(h * head_size..);
            // gpu.apply_position(q, k, &pos_real, &pos_img, cfg.n_heads as i32, head_size as i32);
            let q = &mut rsv.q.mut_slice(h * head_size..(h + 1) * head_size);
            let k = &mut rsv.k.mut_slice(h * head_size..(h + 1) * head_size);
            device.apply_position(q, k, &pos_real, &pos_img, head_size);
        }

        let lo = layer * cfg.seq_len * dim;
        device.copy_from_slice(&mut rsv.key_cache.mut_slice((lo + pos * dim)..(lo + (pos + 1) * dim)), &rsv.k.as_view());
        device.copy_from_slice(&mut rsv.value_cache.mut_slice((lo + pos * dim)..(lo + (pos + 1) * dim)), &rsv.v.as_view());

        // Multihead attention.
        device.multi_head_attention(rsv, cfg, layer, pos);

        device.matmul_1d(&mut rsv.xb2, &wv.wo.slice(layer * dim * dim..(layer + 1) * dim * dim), &rsv.xb.as_view(), dim);


        device.array_add(&mut rsv.x, &rsv.xb2.as_view());

        // pre ffn rmsnorm
        device.rmsnorm(&mut rsv.xb, &rsv.x.as_view(), &wv.rms_ffn_weight.slice(layer * dim .. (layer + 1) * dim), dim);

        // ffn
        device.matmul_1d(&mut rsv.hb,  &wv.w1.slice(layer * hidden_dim * dim..(layer + 1) * hidden_dim * dim), &rsv.xb.as_view(), dim);
        device.matmul_1d(&mut rsv.hb2, &wv.w3.slice(layer * hidden_dim * dim..(layer + 1) * hidden_dim * dim), &rsv.xb.as_view(), dim);

        // silu
        device.sinu(&mut rsv.hb);

        device.array_mult(&mut rsv.hb, &rsv.hb2.as_view());

        device.matmul_1d(&mut rsv.xb,  &wv.w2.slice(layer *dim*hidden_dim..(layer + 1)*dim*hidden_dim), &rsv.hb.as_view(), hidden_dim);

        device.array_add(&mut rsv.x, &rsv.xb.as_view());
    }

    // final rmsnorm
    // rsv.xb.as_mut().copy_from_slice(&rsv.x.as_ref());
    device.copy_from_slice(&mut rsv.xb, &rsv.x.as_view());
    device.rmsnorm(&mut rsv.x, &rsv.xb.as_view(), &wv.rms_final_weight, 0);

    // compute logits
    let wcls = match &wv.wcls {
        Some(wcls) => wcls,
        None => &wv.token_embedding_table,
    };
    device.matmul_1d(&mut rsv.logits,  wcls, &rsv.x.as_view(), dim);
}

pub fn sample<'a>(cfg: &Config, rsv: &mut RunStateView<'a, Vec<f32>>, device: &CPU, temperature: f32) -> usize {
    let next;

    let lr = rsv.logits.range.clone();
    let logits = &mut rsv.logits.as_mut()[lr];
    // let logits = self.rsv.logits.as_mut();
    let rng_seed = 100;
    let mut rng = ChaCha20Rng::seed_from_u64(rng_seed);
    if temperature == 0.0 {
        // greedy decoding, choose argmax
        next = logits.iter().enumerate()
            .reduce(|(i1, v1), (i2, v2)| if v1 > v2 { (i1, v1) } else { (i2, v2) })
            .map(|(i, _)| i).unwrap();
    } else {
        // temperature scaling
        if temperature < 1.0 {
            logits.iter_mut().for_each(|z| *z /= temperature);
        }
        // compute probabilities
        device.softmax_num(logits, 0);
        // next = sample(&transformer.rsv.logits, &mut rng);
        next = sample_top_q(logits, cfg.vocab_size, 0.9, &mut rng);

    }
    next
}

impl<'a> Transformer for TransformerCPU {
    fn cpu_state(&self) -> &RunState<Vec<f32>> {
        &self.state
    }
    fn get_config(&self) -> Config {
        self.config.clone()
    }


    fn from_file(cp_path: &str) -> Self {
        let rd = &mut BufReader::new(File::open(cp_path).unwrap());
        let config = Config::from_file(rd);
        let weights = TransformerWeights::from_file(rd, &config);
        let state = RunState::from_config(&config);

        let cpu = CPU {  };
        Self {
            config: config,
            weights: weights,
            state: state,
            device: cpu,
        }
    }

}

fn sample_top_q(probabilities: &[f32], num: usize, topp: f32, rng: &mut ChaCha20Rng) -> usize {
    let cutoff = (1.0f32 - topp) / ((num - 1) as f32);
    let mut prob_index = probabilities.iter().enumerate().filter(
        |(_, &p)| p > cutoff
    ).collect::<Vec<(usize, &f32)>>();
    prob_index.sort_by(
        |(_, &a2), (_, &b2)|
        b2.partial_cmp(&a2).unwrap()
    );

    let mut cum_prob = 0.0f32;
    let mut last_index = prob_index.len() - 1;
    for i in 0..prob_index.len() {
        cum_prob += prob_index[i].1;
        if cum_prob > topp {
            last_index = i;
            break;
        }
    }

    let r = rng.gen::<f32>() * cum_prob;
    let mut cdf = 0.0f32;

    for i in 0..last_index {
        cdf += prob_index[i].1;
        if r < cdf {
            return prob_index[i].0;
        }
    }

    return prob_index[last_index].0;


}