use std::{io::BufReader, fs::File};
use rayon::prelude::*;
use crate::device::{cpu::CPU, device::Device};
use super::{Config, Transformer};
use rand_chacha::ChaCha20Rng;
use rand::{Rng, SeedableRng};
use crate::utils::read::*;
pub struct TransformerCPU {
    config: Config,
    weights: TransformerWeights,
    state: RunState,
    device: CPU
}

impl Transformer for TransformerCPU {
    fn get_config(&self) -> &Config {
        &self.config
    }
    fn sample(&mut self, temperature: f32) -> usize {
        let next;
        let logits = &mut self.state.logits;
        let rng_seed = 100;
        let mut rng = ChaCha20Rng::seed_from_u64(rng_seed);
        if temperature == 0.0 {
            // greedy decoding, choose argmax
            next = self.state.logits.iter().enumerate()
                .reduce(|(i1, v1), (i2, v2)| if v1 > v2 { (i1, v1) } else { (i2, v2) })
                .map(|(i, _)| i).unwrap();
        } else {
            // temperature scaling
            if temperature < 1.0 {
                logits.iter_mut().for_each(|z| *z /= temperature);
            }
            // compute probabilities
            self.device.softmax(logits);
            // next = sample(&transformer.state.logits, &mut rng);
            next = sample_top_q(&logits, self.config.vocab_size, 0.9, &mut rng);

        }
        next
    }

    fn from_file(cp_path: &str) -> Self {
        let rd = &mut BufReader::new(File::open(cp_path).unwrap());
        let config = Config::from_file(rd);
        let weights = TransformerWeights::from_file(rd, &config);
        let state = RunState::from_config(&config);
        let cpu = CPU {};
        Self {
            config: config,
            weights: weights,
            state: state,
            device: cpu,
        }
    }

    fn forward(&mut self, token: usize, pos: usize) {

        // let gpu = GPU::new();
        // let tg = TransformerWeightsGPU::from_hw(&transformer.weights, &gpu);
        // let sg = RunStateGPU::from_state(&transformer.state, &gpu);

        let cfg = &self.config;

        // let size = cfg.n_layers * cfg.dim * cfg.hidden_dim;
        // let mut buf = vec![0.0f32;size];
        // let mut buf2 = vec![0.0f32;cfg.dim];
        // unsafe { let _ = memcpy_dtoh_sync(& mut buf, tg.w1); };
        // unsafe { let _ = memcpy_dtoh_sync(& mut buf2, sg.q); };


        let weights: &TransformerWeights = &self.weights;
        let state = &mut self.state;
        let dim = cfg.dim;

        let hidden_dim = cfg.hidden_dim;
        let head_size = dim / cfg.n_heads;

        // Load the embedding into x.
        state.x.copy_from_slice(
            &weights.token_embedding_table
                    [(token * cfg.dim)..((token + 1) * cfg.dim)],
        );

        // Positional encoding
        let pos_real = &weights.freq_cis_real[pos*(head_size/2)..(pos+1)*(head_size/2)];
        let pos_img = &weights.freq_cis_imag[pos*(head_size/2)..(pos+1)*(head_size/2)];

        // Forward through the layers.
        for layer in 0..cfg.n_layers {

            // Apply layer normalization. a.k.a we normalize by data items (rows).
            // In comparison, batch normalization normalizes by features (columns);
            // Since we don't need to recenter but only rescaling, we use RMSNorm than
            // actual layer norm, and it worked well in practice.
            self.device.rmsnorm(&mut state.xb, &state.x, &weights.rms_att_weight[layer * dim..(layer + 1) * dim].into());

            // Calculate Q, K, V
            self.device.matmul_1d(&mut state.q, &weights.wq[layer * dim * dim..(layer + 1) * dim * dim].into(), &state.xb, dim);
            self.device.matmul_1d(&mut state.k, &weights.wk[layer * dim * dim..(layer + 1) * dim * dim].into(), &state.xb, dim);
            self.device.matmul_1d(&mut state.v, &weights.wv[layer * dim * dim..(layer + 1) * dim * dim].into(), &state.xb, dim);

            // RoPE relative positional encoding. https://arxiv.org/pdf/2104.09864.pdf
            // b/c in attention we only care about the distance between two words, we use relative attention.
            // Rotary position embeddings are only applied to the Q & Ks, not the values,
            // also, they are applied after the multi with weights. (The vanilla transformer is different,
            // positional embeddings were applied before multiplication with weights.)
            for h in 0..cfg.n_heads {
                let q = &mut state.q[h * head_size..(h + 1) * head_size];
                let k = &mut state.k[h * head_size..(h + 1) * head_size];

                for i in 0..(head_size / 2) {
                    // Instead of doing matmul, this is a more simplified way (from the paper above)
                    let (fcr, fci) = (pos_real[i], pos_img[i]);
                    (q[i * 2], q[i * 2 + 1]) = (
                        q[i * 2] * fcr - q[i * 2 + 1] * fci,
                        q[i * 2] * fci + q[i * 2 + 1] * fcr);
                    (k[i * 2], k[i * 2 + 1]) = (
                        k[i * 2] * fcr - k[i * 2 + 1] * fci,
                        k[i * 2] * fci + k[i * 2 + 1] * fcr);
                }
            }

            let lo = layer * cfg.seq_len * dim;
            state.key_cache[(lo + pos * dim)..(lo + (pos + 1) * dim)].copy_from_slice(&state.k);
            state.value_cache[(lo + pos * dim)..(lo + (pos + 1) * dim)].copy_from_slice(&state.v);

            // Multihead attention.
            let mut atts: Vec<&mut [f32]> = state.att.chunks_mut(cfg.seq_len).collect();
            let qs: Vec<&mut [f32]> = state.q.chunks_mut(head_size).collect();
            let xbs: Vec<&mut [f32]> = state.xb.chunks_mut(head_size).collect();
            atts.par_iter_mut().zip(xbs).enumerate().for_each(|(h, (att,xb))| {
                let q = &qs[h];
                for t in 0..(pos + 1) {
                    let ko = lo + t * dim + h * head_size;
                    let k = &state.key_cache[ko..(ko + head_size)];
                    att[t] = q.iter().zip(k.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>() / (head_size as f32).sqrt();
                }
                self.device.softmax(&mut att[..(pos + 1)].into());
                xb.fill(0.0);
                for t in 0..(pos + 1) {
                    let ko = lo + t * dim + h * head_size;
                    let v = &state.value_cache[ko..(ko + head_size)];
                    let a = att[t];
                    xb.iter_mut().zip(v).for_each(|(xbi, &vi)| *xbi += a * vi);
                }

            });
            self.device.matmul_1d(&mut state.xb2, &weights.wo[layer * dim * dim..(layer + 1) * dim * dim].into(), &state.xb, dim);

            state.x.iter_mut().zip(state.xb2.iter()).for_each(|(a, b)| *a += *b);

            // pre ffn rmsnorm
            self.device.rmsnorm(&mut state.xb, &state.x, &weights.rms_ffn_weight[layer * dim .. (layer + 1) * dim].into());

            // ffn
            self.device.matmul_1d(&mut state.hb,  &weights.w1[layer * hidden_dim * dim..(layer + 1) * hidden_dim * dim].into(), &state.xb, dim);
            self.device.matmul_1d(&mut state.hb2, &weights.w3[layer * hidden_dim * dim..(layer + 1) * hidden_dim * dim].into(), &state.xb, dim);

            // silu
            state.hb.par_iter_mut().for_each(|a|*a = *a * (1.0 / (1.0 + (-*a).exp())));

            state.hb.iter_mut().zip(state.hb2.iter()).for_each(|(a, &b)| *a *= b);

            self.device.matmul_1d(&mut state.xb,  &weights.w2[layer *dim*hidden_dim..(layer + 1)*dim*hidden_dim].into(), &state.hb, hidden_dim);

            state.x.iter_mut().zip(state.xb.iter()).for_each(|(a, &b)| *a += b);
        }

        // final rmsnorm
        state.xb.copy_from_slice(&state.x);
        self.device.rmsnorm(&mut state.x, &state.xb, &weights.rms_final_weight);

        // compute logits
        let wcls = match &weights.wcls {
            Some(wcls) => wcls,
            None => &weights.token_embedding_table,
        };
        self.device.matmul_1d(&mut state.logits,  wcls,&state.x, dim);
    }



}

struct TransformerWeights {
    token_embedding_table: Vec<f32>,
    rms_att_weight: Vec<f32>,
    rms_ffn_weight: Vec<f32>,

    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,

    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,

    rms_final_weight: Vec<f32>,
    freq_cis_real: Vec<f32>,
    freq_cis_imag: Vec<f32>,
    wcls: Option<Vec<f32>>,
}

impl TransformerWeights {
    fn from_file(f: &mut BufReader<File>, c: &Config) -> Self {
        let head_size = c.dim / c.n_heads;
        Self {
            token_embedding_table: read_vec(f, c.vocab_size * c.dim),
            rms_att_weight: read_vec(f, c.n_layers * c.dim),
            wq: read_vec(f, c.n_layers * c.dim * c.dim),
            wk: read_vec(f, c.n_layers * c.dim * c.dim),
            wv: read_vec(f, c.n_layers * c.dim * c.dim),
            wo: read_vec(f, c.n_layers * c.dim * c.dim),
            rms_ffn_weight: read_vec(f, c.n_layers * c.dim),
            w1: read_vec(f, c.n_layers * c.dim * c.hidden_dim),
            w2: read_vec(f, c.n_layers * c.dim * c.hidden_dim),
            w3: read_vec(f, c.n_layers * c.dim * c.hidden_dim),
            rms_final_weight: read_vec(f, c.dim),
            freq_cis_real: read_vec(f, c.seq_len * head_size / 2),
            freq_cis_imag: read_vec(f, c.seq_len * head_size / 2),
            wcls: {
                if c.shared_weight { None } else {
                    Some(read_vec::<f32>(f, c.vocab_size * c.dim))
                }
            },
        }
    }

}

struct RunState {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl RunState {
    fn from_config(cfg: &Config) -> Self {
        let kv_dim = cfg.dim * cfg.n_kv_heads / cfg.n_heads;
        Self {
            x: vec![0.0; cfg.dim as usize],
            xb: vec![0.0; cfg.dim as usize],
            xb2: vec![0.0; cfg.dim as usize],
            hb: vec![0.0; cfg.hidden_dim as usize],
            hb2: vec![0.0; cfg.hidden_dim as usize],
            q: vec![0.0; cfg.dim as usize],
            k: vec![0.0; cfg.dim as usize],
            v: vec![0.0; cfg.dim as usize],
            att: vec![0.0; cfg.n_heads * cfg.seq_len as usize],
            logits: vec![0.0; cfg.vocab_size as usize],
            key_cache: vec![0.0; cfg.n_layers * cfg.seq_len * kv_dim as usize],
            value_cache: vec![0.0; cfg.n_layers * cfg.seq_len * kv_dim as usize],
        }
    }

}



fn sample_top_q(probabilities: &Vec<f32>, num: usize, topp: f32, rng: &mut ChaCha20Rng) -> usize {
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