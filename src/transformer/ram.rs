use std::{io::BufReader, fs::File};
use rayon::prelude::*;
use crate::device::cpu::CPU;
use super::{Config, Transformer, CPUStorage, MutView, View};
use rand_chacha::ChaCha20Rng;
use rand::{Rng, SeedableRng};
use crate::utils::read::*;
pub struct TransformerCPU {
    pub config: Config,
    pub weights: TransformerWeights,
    pub state: RunState,
    pub device: CPU
}

impl Transformer for TransformerCPU {
    fn cpu_state(&self) -> &RunState {
        &self.state
    }
    fn get_config(&self) -> Config {
        self.config.clone()
    }
    fn sample(&mut self, temperature: f32) -> usize {
        let next;
        let logits = self.state.logits.as_mut();
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
            self.device.softmax_num(logits, 0);
            // next = sample(&transformer.state.logits, &mut rng);
            next = sample_top_q(logits, self.config.vocab_size, 0.9, &mut rng);

        }
        next
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

    fn forward(&mut self, token: usize, pos: usize) {

        let cfg = &self.config;

        let weights: &TransformerWeights = &self.weights;
        let state = &mut self.state;
        let dim = cfg.dim;

        let hidden_dim = cfg.hidden_dim;
        let head_size = dim / cfg.n_heads;

        // Load the embedding into x.
        state.x.as_mut_slice().as_mut().copy_from_slice(
            &weights.token_embedding_table.slice(
                    (token * cfg.dim)..((token + 1) * cfg.dim)).as_ref());

        // Positional encoding
        let pos_real = &weights.freq_cis_real.slice(pos*(head_size/2)..(pos+1)*(head_size/2));
        let pos_img = &weights.freq_cis_imag.slice(pos*(head_size/2)..(pos+1)*(head_size/2));

        // Forward through the layers.
        for layer in 0..cfg.n_layers {

            // Apply layer normalization. a.k.a we normalize by data items (rows).
            // In comparison, batch normalization normalizes by features (columns);
            // Since we don't need to recenter but only rescaling, we use RMSNorm than
            // actual layer norm, and it worked well in practice.
            self.device.rmsnorm(&mut state.xb.as_mut_slice(), &state.x.as_slice(), &weights.rms_att_weight.slice(layer * dim..(layer + 1) * dim), dim);

            // Calculate Q, K, V
            self.device.matmul_1d(&mut state.q.as_mut_slice(), &weights.wq.slice(layer * dim * dim..(layer + 1) * dim * dim), &state.xb.as_slice(), dim);
            self.device.matmul_1d(&mut state.k.as_mut_slice(), &weights.wk.slice(layer * dim * dim..(layer + 1) * dim * dim), &state.xb.as_slice(), dim);
            self.device.matmul_1d(&mut state.v.as_mut_slice(), &weights.wv.slice(layer * dim * dim..(layer + 1) * dim * dim), &state.xb.as_slice(), dim);

            // RoPE relative positional encoding. https://arxiv.org/pdf/2104.09864.pdf
            // b/c in attention we only care about the distance between two words, we use relative attention.
            // Rotary position embeddings are only applied to the Q & Ks, not the values,
            // also, they are applied after the multi with weights. (The vanilla transformer is different,
            // positional embeddings were applied before multiplication with weights.)
            for h in 0..cfg.n_heads {
                let q = &mut state.q.mut_slice(h * head_size..(h + 1) * head_size);
                let k = &mut state.k.mut_slice(h * head_size..(h + 1) * head_size);

                for i in 0..(head_size / 2) {
                    // Instead of doing matmul, this is a more simplified way (from the paper above)
                    let (fcr, fci) = (pos_real.as_ref()[i], pos_img.as_ref()[i]);
                    (q.as_mut()[i * 2], q.as_mut()[i * 2 + 1]) = (
                        q.as_mut()[i * 2] * fcr - q.as_mut()[i * 2 + 1] * fci,
                        q.as_mut()[i * 2] * fci + q.as_mut()[i * 2 + 1] * fcr);
                    (k.as_mut()[i * 2], k.as_mut()[i * 2 + 1]) = (
                        k.as_mut()[i * 2] * fcr - k.as_mut()[i * 2 + 1] * fci,
                        k.as_mut()[i * 2] * fci + k.as_mut()[i * 2 + 1] * fcr);
                }
            }

            let lo = layer * cfg.seq_len * dim;
            state.key_cache.mut_slice((lo + pos * dim)..(lo + (pos + 1) * dim)).as_mut().copy_from_slice(&state.k.as_slice().as_ref());
            state.value_cache.mut_slice((lo + pos * dim)..(lo + (pos + 1) * dim)).as_mut().copy_from_slice(&state.v.as_slice().as_ref());

            // Multihead attention.
            let mut atts: Vec<&mut [f32]> = state.att.as_mut().chunks_mut(cfg.seq_len).collect();
            let qs: Vec<&mut [f32]> = state.q.as_mut().chunks_mut(head_size).collect();
            let xbs: Vec<&mut [f32]> = state.xb.as_mut().chunks_mut(head_size).collect();
            atts.par_iter_mut().zip(xbs).enumerate().for_each(|(h, (att,xb))| {
                let q = &qs[h];
                for t in 0..(pos + 1) {
                    let ko = lo + t * dim + h * head_size;
                    let bind = state.key_cache.slice(ko..(ko + head_size));
                    let k = bind.as_ref();
                    att[t] = q.iter().zip(k.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>() / (head_size as f32).sqrt();
                }
                self.device.softmax_num(&mut att[..(pos + 1)], pos + 1);
                xb.fill(0.0);
                for t in 0..(pos + 1) {
                    let ko = lo + t * dim + h * head_size;
                    let v = &state.value_cache.slice(ko..(ko + head_size));
                    let a = att[t];
                    xb.iter_mut().zip(v.as_ref()).for_each(|(xbi, &vi)| *xbi += a * vi);
                }

            });
            self.device.matmul_1d(&mut state.xb2.as_mut_slice(), &weights.wo.slice(layer * dim * dim..(layer + 1) * dim * dim), &state.xb.as_slice(), dim);

            state.x.as_mut().iter_mut().zip(state.xb2.as_ref().iter()).for_each(|(a, b)| *a += *b);

            // pre ffn rmsnorm
            self.device.rmsnorm(&mut state.xb.as_mut_slice(), &state.x.as_slice(), &weights.rms_ffn_weight.slice(layer * dim .. (layer + 1) * dim), dim);

            // ffn
            self.device.matmul_1d(&mut state.hb.as_mut_slice(),  &weights.w1.slice(layer * hidden_dim * dim..(layer + 1) * hidden_dim * dim), &state.xb.as_slice(), dim);
            self.device.matmul_1d(&mut state.hb2.as_mut_slice(), &weights.w3.slice(layer * hidden_dim * dim..(layer + 1) * hidden_dim * dim), &state.xb.as_slice(), dim);

            // silu
            state.hb.as_mut().par_iter_mut().for_each(|a|*a = *a * (1.0 / (1.0 + (-*a).exp())));

            state.hb.as_mut().iter_mut().zip(state.hb2.as_ref().iter()).for_each(|(a, &b)| *a *= b);

            self.device.matmul_1d(&mut state.xb.as_mut_slice(),  &&weights.w2.slice(layer *dim*hidden_dim..(layer + 1)*dim*hidden_dim), &state.hb.as_slice(), hidden_dim);

            state.x.as_mut().iter_mut().zip(state.xb.as_ref().iter()).for_each(|(a, &b)| *a += b);
        }

        // final rmsnorm
        state.xb.as_mut().copy_from_slice(&state.x.as_ref());
        self.device.rmsnorm(&mut state.x.as_mut_slice(), &state.xb.as_slice(), &weights.rms_final_weight.as_slice(), 0);

        // compute logits
        let wcls = match &weights.wcls {
            Some(wcls) => wcls,
            None => &weights.token_embedding_table,
        };
        self.device.matmul_1d(&mut state.logits.as_mut_slice(),  &wcls.as_slice(), &state.x.as_slice(), dim);
    }



}

pub struct TransformerWeights {
    pub token_embedding_table: CPUStorage,
    pub rms_att_weight:  CPUStorage,
    pub rms_ffn_weight: CPUStorage,
    pub wq: CPUStorage,
    pub wk: CPUStorage,
    pub wv: CPUStorage,
    pub wo: CPUStorage,
    pub w1: CPUStorage,
    pub w2: CPUStorage,
    pub w3: CPUStorage,
    pub rms_final_weight:CPUStorage,
    pub freq_cis_real: CPUStorage,
    pub freq_cis_imag: CPUStorage,
    pub wcls: Option<CPUStorage>,
}

impl TransformerWeights {
    fn from_file(f: &mut BufReader<File>, c: &Config) -> Self {
        let head_size = c.dim / c.n_heads;
        Self {
            token_embedding_table: CPUStorage { data: read_vec(f, c.vocab_size * c.dim)},
            rms_att_weight: CPUStorage { data: read_vec(f, c.n_layers * c.dim) },
            wq: CPUStorage { data: read_vec(f, c.n_layers * c.dim * c.dim) },
            wk: CPUStorage { data: read_vec(f, c.n_layers * c.dim * c.dim) },
            wv: CPUStorage { data: read_vec(f, c.n_layers * c.dim * c.dim) },
            wo: CPUStorage { data: read_vec(f, c.n_layers * c.dim * c.dim) },
            rms_ffn_weight: CPUStorage { data: read_vec(f, c.n_layers * c.dim) },
            w1: CPUStorage { data: read_vec(f, c.n_layers * c.dim * c.hidden_dim) },
            w2: CPUStorage { data: read_vec(f, c.n_layers * c.dim * c.hidden_dim) },
            w3: CPUStorage { data: read_vec(f, c.n_layers * c.dim * c.hidden_dim) },
            rms_final_weight: CPUStorage { data: read_vec(f, c.dim) },
            freq_cis_real: CPUStorage { data: read_vec(f, c.seq_len * head_size / 2) },
            freq_cis_imag: CPUStorage { data: read_vec(f, c.seq_len * head_size / 2) },
            wcls: {
                if c.shared_weight { None } else {
                    Some(CPUStorage { data: read_vec::<f32>(f, c.vocab_size * c.dim) })
                }
            },
        }
    }

}

#[derive(Default)]
pub struct RunState {
    pub x: CPUStorage,
    pub xb: CPUStorage,
    pub xb2: CPUStorage,
    pub hb: CPUStorage,
    pub hb2: CPUStorage,
    pub q: CPUStorage,
    pub k: CPUStorage,
    pub v: CPUStorage,
    pub att: CPUStorage,
    pub logits: CPUStorage,
    pub key_cache: CPUStorage,
    pub value_cache: CPUStorage,
}

impl RunState {
    fn from_config(cfg: &Config) -> Self {
        let kv_dim = cfg.dim * cfg.n_kv_heads / cfg.n_heads;
        Self {
            x: CPUStorage { data: vec![0.0; cfg.dim as usize] },
            xb: CPUStorage { data: vec![0.0; cfg.dim as usize] },
            xb2: CPUStorage { data: vec![0.0; cfg.dim as usize] },
            hb: CPUStorage { data: vec![0.0; cfg.hidden_dim as usize] },
            hb2: CPUStorage { data: vec![0.0; cfg.hidden_dim as usize] },
            q: CPUStorage { data: vec![0.0; cfg.dim as usize] },
            k: CPUStorage { data: vec![0.0; cfg.dim as usize] },
            v: CPUStorage { data: vec![0.0; cfg.dim as usize] },
            att: CPUStorage { data: vec![0.0; cfg.n_heads * cfg.seq_len as usize] },
            logits: CPUStorage { data: vec![0.0; cfg.vocab_size as usize] },
            key_cache: CPUStorage { data: vec![0.0; cfg.n_layers * cfg.seq_len * kv_dim as usize] },
            value_cache: CPUStorage { data: vec![0.0; cfg.n_layers * cfg.seq_len * kv_dim as usize] },
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