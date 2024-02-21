use crate::device::device::Device;

use super::{state::{RunStateView, TransformerWeightsView}, Config, Storage};

pub fn forward<'a, T: Storage, D: Device<T>>(cfg: &Config, wv: &TransformerWeightsView<'a, T>, rsv: &mut RunStateView<'a, T>, token: usize, pos: usize,  device: &D) {
    let dim = cfg.dim;
    let hidden_dim = cfg.hidden_dim;
    let head_size = dim / cfg.n_heads;
    device.copy_from_slice(&mut rsv.x, &wv.token_embedding_table.slice(token * dim..), dim);

    let pos_real = wv.freq_cis_real.slice(pos * (head_size / 2)..);
    let pos_img = wv.freq_cis_imag.slice(pos * (head_size / 2)..);


    for layer in 0..cfg.n_layers {

        device.rmsnorm(&mut rsv.xb, &rsv.x.as_view(), &wv.rms_att_weight.slice(layer * dim..), dim);

        device.matmul(&mut rsv.q, &wv.wq.slice(layer * dim * dim..), &rsv.xb.as_view(), dim, dim, 1);
        // device.matmul(&mut rsv.q, &wv.wq.slice(layer * dim * dim..), &rsv.xb, dim, dim, 1);
        device.matmul(&mut rsv.q, &wv.wq.slice(layer * dim * dim..), &rsv.xb.as_view(), dim, dim, 1);
        device.matmul(&mut rsv.k, &wv.wk.slice(layer * dim * dim..), &rsv.xb.as_view(), dim, dim, 1);
        device.matmul(&mut rsv.v, &wv.wv.slice(layer * dim * dim..), &rsv.xb.as_view(), dim, dim, 1);

        for h in 0..cfg.n_heads {

            let q = &mut rsv.q.mut_slice(h * head_size..);
            let k = &mut rsv.k.mut_slice(h * head_size..);
            device.apply_position(q, k, &pos_real, &pos_img, head_size);
        }

        // -- %%
        let lo = layer * cfg.seq_len * dim;
        device.copy_from_slice(&mut rsv.key_cache.mut_slice(lo + pos * dim..), &rsv.k.as_view(), dim);
        device.copy_from_slice(&mut rsv.value_cache.mut_slice(lo + pos * dim..), &rsv.v.as_view(), dim);
        // self.state.into_state(&mut self._cpu_state);
        device.multi_head_attention(rsv, &cfg, layer, pos);
        // self.state.into_state(&mut self._cpu_state);

        device.matmul(&mut rsv.xb2, &wv.wo.slice(layer * dim * dim..), &rsv.xb.as_view(), dim, dim, 1);

        device.array_add(&mut rsv.x, &rsv.xb2.as_view(), dim);

        device.rmsnorm(&mut rsv.xb, &rsv.x.as_view(), &wv.rms_ffn_weight.slice(layer * dim..), dim);

        device.matmul(&mut rsv.hb, &wv.w1.slice(layer * hidden_dim * dim..), &rsv.xb.as_view(), dim, hidden_dim, 1);
        device.matmul(&mut rsv.hb2, &wv.w3.slice(layer * hidden_dim * dim..), &rsv.xb.as_view(), dim, hidden_dim, 1);

        //---
        device.sinu(&mut rsv.hb, hidden_dim);


        device.array_mult(&mut rsv.hb, &rsv.hb2.as_view(), hidden_dim);


        device.matmul(&mut rsv.xb, &wv.w2.slice(layer * dim * hidden_dim..), &rsv.hb.as_view(), hidden_dim, dim, 1);

        device.array_add(&mut rsv.x, &rsv.xb.as_view(), dim);

    }

    device.copy_from_slice(&mut rsv.xb, &rsv.x.as_view(), dim);
    device.rmsnorm(&mut rsv.x, &rsv.xb.as_view(), &wv.rms_final_weight, dim);
    device.matmul(&mut rsv.logits, &wv.wcls, &rsv.x.as_view(), dim, cfg.vocab_size, 1);

}