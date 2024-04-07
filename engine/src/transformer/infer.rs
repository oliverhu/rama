use rand::Rng;
use rand_chacha::ChaCha20Rng;

use crate::device::device::{Device, QuantDevice};

use super::{state::{RunStateView, TransformerWeightsView}, Config, Storage};

pub fn forward<'a, T: Storage, Q: Storage, D: Device<T, T>>(cfg: &Config, wv: &TransformerWeightsView<'a, T, T>, rsv: &mut RunStateView<'a, T, T>, token: usize, pos: usize,  device: &D) {
    // let mut cpu_state = RunState::from_config(&cfg); // for debugging
    let dim: usize = cfg.dim;
    let hidden_dim = cfg.hidden_dim;
    let head_size = dim / cfg.n_heads;
    device.copy_from_slice(&mut rsv.x, &wv.token_embedding_table.slice(token * dim..((token + 1) * cfg.dim)), dim);

    let pos_real = wv.freq_cis_real.slice(pos * (head_size / 2)..);
    let pos_img = wv.freq_cis_imag.slice(pos * (head_size / 2)..);

    for layer in 0..cfg.n_layers {
        device.rmsnorm(&mut rsv.xb, &rsv.x.as_view(), &wv.rms_att_weight.slice(layer * dim..), dim);
        device.matmul(&mut rsv.q, &wv.wq.slice(layer * dim * dim..), &rsv.xb.as_view(), dim, dim, 1);
        device.matmul(&mut rsv.k, &wv.wk.slice(layer * dim * dim..), &rsv.xb.as_view(), dim, dim, 1);
        device.matmul(&mut rsv.v, &wv.wv.slice(layer * dim * dim..), &rsv.xb.as_view(), dim, dim, 1);

        for h in 0..cfg.n_heads {
            let q = &mut rsv.q.mut_slice(h * head_size..);
            let k = &mut rsv.k.mut_slice(h * head_size..);
            device.apply_position(q, k, &pos_real, &pos_img, head_size);
        }

        let lo = layer * cfg.seq_len * dim;
        device.copy_from_slice(&mut rsv.key_cache.mut_slice(lo + pos * dim..(lo + (pos + 1) * dim)), &rsv.k.as_view(), dim);
        device.copy_from_slice(&mut rsv.value_cache.mut_slice(lo + pos * dim..(lo + (pos + 1) * dim)), &rsv.v.as_view(), dim);
        device.multi_head_attention(rsv, &cfg, layer, pos);
        device.matmul(&mut rsv.xb2, &wv.wo.slice(layer * dim * dim..), &rsv.xb.as_view(), dim, dim, 1);

        device.array_add(&mut rsv.x, &rsv.xb2.as_view(), dim);

        device.rmsnorm(&mut rsv.xb, &rsv.x.as_view(), &wv.rms_ffn_weight.slice(layer * dim..), dim);

        device.matmul(&mut rsv.hb, &wv.w1.slice(layer * hidden_dim * dim..), &rsv.xb.as_view(), dim, hidden_dim, 1);
        device.matmul(&mut rsv.hb2, &wv.w3.slice(layer * hidden_dim * dim..), &rsv.xb.as_view(), dim, hidden_dim, 1);

        device.sinu(&mut rsv.hb, hidden_dim);
        device.array_mult(&mut rsv.hb, &rsv.hb2.as_view(), hidden_dim);
        device.matmul(&mut rsv.xb, &wv.w2.slice(layer * dim * hidden_dim..), &rsv.hb.as_view(), hidden_dim, dim, 1);
        device.array_add(&mut rsv.x, &rsv.xb.as_view(), dim);
    }
    device.copy_from_slice(&mut rsv.xb, &rsv.x.as_view(), dim);
    device.rmsnorm(&mut rsv.x, &rsv.xb.as_view(), &wv.rms_final_weight, dim);
    device.matmul(&mut rsv.logits, &wv.wcls, &rsv.x.as_view(), dim, cfg.vocab_size, 1);

}

// Quantized version of forward pass. The difference is mostly around quantization/dequantization
// before matrix multiplication. Likely can be refactored later.
pub fn forward_q<'a, T: Storage, Q: Storage, D: Device<T, Q> + QuantDevice<T, Q>>(cfg: &Config, wv: &TransformerWeightsView<'a, T, Q>, rsv: &mut RunStateView<'a, T, Q>, token: usize, pos: usize,  device: &D) {
    // let mut cpu_state = RunState::from_config(&cfg); // for debugging
    let dim: usize = cfg.dim;
    let hidden_dim = cfg.hidden_dim;
    let head_size = dim / cfg.n_heads;
    device.copy_from_slice(&mut rsv.x, &wv.token_embedding_table.slice(token * dim..((token + 1) * cfg.dim)), dim);

    let pos_real = wv.freq_cis_real.slice(pos * (head_size / 2)..);
    let pos_img = wv.freq_cis_imag.slice(pos * (head_size / 2)..);

    for layer in 0..cfg.n_layers {
        device.rmsnorm(&mut rsv.xb, &rsv.x.as_view(), &wv.rms_att_weight.slice(layer * dim..), dim);

        // TODO QUANTIZE
        device.quantize(&mut rsv.xq, &rsv.xb.as_view(), dim);

        device.matmul_q(&mut rsv.q, &wv.wq.slice(layer..layer + 1), &rsv.xq.as_view(), dim, dim, 1);
        device.matmul_q(&mut rsv.k, &wv.wk.slice(layer * dim * dim..), &rsv.xq.as_view(), dim, dim, 1);
        device.matmul_q(&mut rsv.v, &wv.wv.slice(layer * dim * dim..), &rsv.xq.as_view(), dim, dim, 1);


        for h in 0..cfg.n_heads {
            let q = &mut rsv.q.mut_slice(h * head_size..);
            let k = &mut rsv.k.mut_slice(h * head_size..);
            device.apply_position(q, k, &pos_real, &pos_img, head_size);
        }

        let lo = layer * cfg.seq_len * dim;
        device.copy_from_slice(&mut rsv.key_cache.mut_slice(lo + pos * dim..(lo + (pos + 1) * dim)), &rsv.k.as_view(), dim);
        device.copy_from_slice(&mut rsv.value_cache.mut_slice(lo + pos * dim..(lo + (pos + 1) * dim)), &rsv.v.as_view(), dim);
        device.multi_head_attention(rsv, &cfg, layer, pos);

        // Quantize xb for output of the attention
        device.quantize(&mut rsv.xq, &rsv.xb.as_view(), dim);

        device.matmul_q(&mut rsv.xb2, &wv.wo.slice(layer * dim * dim..), &rsv.xq.as_view(), dim, dim, 1);

        device.array_add(&mut rsv.x, &rsv.xb2.as_view(), dim);

        device.rmsnorm(&mut rsv.xb, &rsv.x.as_view(), &wv.rms_ffn_weight.slice(layer * dim..), dim);

        // Quantize again
        device.quantize(&mut rsv.xq, &rsv.xb.as_view(), dim);

        device.matmul_q(&mut rsv.hb, &wv.w1.slice(layer * hidden_dim * dim..), &rsv.xq.as_view(), dim, hidden_dim, 1);
        device.matmul_q(&mut rsv.hb2, &wv.w3.slice(layer * hidden_dim * dim..), &rsv.xq.as_view(), dim, hidden_dim, 1);

        device.sinu(&mut rsv.hb, hidden_dim);

        // Quantize before matmul
        device.quantize(&mut rsv.hq, &rsv.hb.as_view(), hidden_dim);

        device.array_mult(&mut rsv.hb, &rsv.hb2.as_view(), hidden_dim);
        device.matmul_q(&mut rsv.xb, &wv.w2.slice(layer * dim * hidden_dim..), &rsv.hq.as_view(), hidden_dim, dim, 1);
        device.array_add(&mut rsv.x, &rsv.xb.as_view(), dim);
    }
    device.copy_from_slice(&mut rsv.xb, &rsv.x.as_view(), dim);
    device.rmsnorm(&mut rsv.x, &rsv.xb.as_view(), &wv.rms_final_weight, dim);

    // Quantize
    device.quantize(&mut rsv.xq, &rsv.x.as_view(), dim);
    device.matmul_q(&mut rsv.logits, &wv.wcls, &rsv.xq.as_view(), dim, cfg.vocab_size, 1);

}

pub fn sample_top_q(probabilities: &[f32], num: usize, topp: f32, rng: &mut ChaCha20Rng) -> usize {
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
    prob_index[last_index].0
}