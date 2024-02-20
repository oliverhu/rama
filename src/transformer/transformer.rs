use std::marker::PhantomData;

use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::device::device::Device;

use super::{Config, ram::{TransformerWeights, RunState}, MutView, View};

pub struct Transformer<'a, D, MT, T> where D: Device<'a, MT, T>, MT: MutView<'a>, T: View<'a> {
    pub config: Config,
    pub weights: TransformerWeightsTMP<'a, T>,
    pub state: RunStateTMP<'a, MT>,
    pub device: D,
    phantom: PhantomData<&'a MT>,
    phantom2: PhantomData<&'a T>,
}

pub struct RunStateTMP<'a, MT> where MT: MutView<'a> {
    pub x: MT,
    pub xb: MT,
    pub xb2: MT,
    pub hb: MT,
    pub hb2: MT,
    pub q: MT,
    pub k: MT,
    pub v: MT,
    pub att: MT,
    pub logits: MT,
    pub key_cache: MT,
    pub value_cache: MT,
    phantom: PhantomData<&'a MT>
}
pub struct TransformerWeightsTMP<'a, T> where T: View<'a> {
    pub token_embedding_table: T,
    pub rms_att_weight:  T,
    pub rms_ffn_weight: T,
    pub wq: T,
    pub wk: T,
    pub wv: T,
    pub wo: T,
    pub w1: T,
    pub w2: T,
    pub w3: T,
    pub rms_final_weight:T,
    pub freq_cis_real: T,
    pub freq_cis_imag: T,
    pub wcls: Option<T>,
    phantom: PhantomData<&'a T>,
}

impl<'a, D, MT, T> Transformer<'a, D, MT, T>
where D: Device<'a, MT, T>, MT: MutView<'a, MV = MT, V = T>, T: View<'a, V = T> {
    fn forward(&mut self, token: usize, pos: usize) {
        let gpu_weights = &self.weights;
        let cfg = &self.config;
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim;
        let head_size = dim / cfg.n_heads;
        let gpu = &self.device;
        let gpu_state = &mut self.state;

        self.device.copy_from_slice(&mut gpu_state.x, &gpu_weights.token_embedding_table.slice(token * dim..((token + 1) * cfg.dim)), dim);

        let pos_real = gpu_weights.freq_cis_real.slice(pos * (head_size / 2)..(pos+1)*(head_size/2));
        let pos_img = gpu_weights.freq_cis_imag.slice(pos * (head_size / 2)..(pos+1)*(head_size/2));


        for layer in 0..cfg.n_layers {

            gpu.rmsnorm(&mut gpu_state.xb, &gpu_state.x.into_view(), &gpu_weights.rms_att_weight.slice(layer * dim..(layer + 1) * dim), dim);

            gpu.matmul(&mut gpu_state.q, &gpu_weights.wq.slice(layer * dim * dim..(layer + 1) * dim * dim), &gpu_state.xb.into_view(), dim, dim, 1);
            // gpu.matmul_cublas(&mut gpu_state.q, &gpu_weights.wq.slice(layer * dim * dim..), &gpu_state.xb, dim, dim, 1);
            gpu.matmul(&mut gpu_state.q, &gpu_weights.wq.slice(layer * dim * dim..(layer + 1) * dim * dim), &gpu_state.xb.into_view(), dim, dim, 1);

            gpu.matmul(&mut gpu_state.k, &gpu_weights.wk.slice(layer * dim * dim..(layer + 1) * dim * dim), &gpu_state.xb.into_view(), dim, dim, 1);

            gpu.matmul(&mut gpu_state.v, &gpu_weights.wv.slice(layer * dim * dim..(layer + 1) * dim * dim), &gpu_state.xb.into_view(), dim, dim, 1);

            for h in 0..cfg.n_heads {

                let q = &gpu_state.q.mut_slice(h * head_size..(h + 1) * head_size);
                let k = &gpu_state.k.mut_slice(h * head_size..(h + 1) * head_size);
                gpu.apply_position(q, k, &pos_real, &pos_img, cfg.n_heads as i32, head_size as i32);
            }

            // -- %%
            let lo = layer * cfg.seq_len * dim;
            gpu.copy_from_slice(&mut gpu_state.key_cache.mut_slice(lo + pos * dim..(lo + (pos + 1) * dim)), &gpu_state.k.into_view(), dim);
            gpu.copy_from_slice(&mut gpu_state.value_cache.mut_slice(lo + pos * dim..(lo + (pos + 1) * dim)), &gpu_state.v.into_view(), dim);
            // self.state.into_state(&mut self._cpu_state);
            gpu.multi_head_attention(&gpu_state, &cfg, layer, pos);
            // self.state.into_state(&mut self._cpu_state);

            gpu.matmul(&mut gpu_state.xb2, &gpu_weights.wo.slice(layer * dim * dim..(layer + 1) * dim * dim), &gpu_state.xb.into_view(), dim, dim, 1);

            gpu.array_add(&mut gpu_state.x, &gpu_state.xb2.into_view(), dim);

            gpu.rmsnorm(&mut gpu_state.xb, &gpu_state.x.into_view(), &gpu_weights.rms_ffn_weight.slice(layer * dim..(layer + 1) * dim), dim);


            gpu.matmul(&mut gpu_state.hb, &gpu_weights.w1.slice(layer * hidden_dim * dim..(layer + 1) * hidden_dim * dim), &gpu_state.xb.into_view(), dim, hidden_dim, 1);
            gpu.matmul(&mut gpu_state.hb2, &gpu_weights.w3.slice(layer * hidden_dim * dim..(layer + 1) * hidden_dim * dim), &gpu_state.xb.into_view(), dim, hidden_dim, 1);

            //---
            gpu.sinu(&mut gpu_state.hb, hidden_dim);


            gpu.array_mult(&mut gpu_state.hb, &gpu_state.hb2.into_view(), hidden_dim);


            gpu.matmul(&mut gpu_state.xb, &gpu_weights.w2.slice(layer * dim * hidden_dim..(layer + 1)*dim*hidden_dim), &gpu_state.hb.into_view(), hidden_dim, dim, 1);

            gpu.array_add(&mut gpu_state.x, &gpu_state.xb.into_view(), dim);

        }

        gpu.copy_from_slice(&mut gpu_state.xb, &gpu_state.x.into_view(), dim);

        gpu.rmsnorm(&mut gpu_state.x, &gpu_state.xb.into_view(), &gpu_weights.rms_final_weight, dim);

        gpu.matmul(&mut gpu_state.logits, &gpu_weights.wcls.unwrap(), &gpu_state.x.into_view(), dim, cfg.vocab_size, 1);

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
            softmax_num(logits, 0);
            // next = sample(&transformer.state.logits, &mut rng);
            next = sample_top_q(logits, self.config.vocab_size, 0.9, &mut rng);

        }
        next
    }
}

pub fn softmax_num(x: &mut [f32], _n: usize) {
    let max = x.par_iter().copied().reduce(|| x[0], |a, b| a.max(b));
    x.par_iter_mut().for_each(|a| *a=(*a-max).exp());
    let sum = x.par_iter().sum::<f32>();
    x.par_iter_mut().for_each(|a| *a /= sum);
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