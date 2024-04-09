use std::{fs::File, io::BufReader};

use super::{ram_q80::QuantizedTensor, read_vec, state::{RunState, TransformerWeights}, Config};


impl RunState<Vec<f32>, Vec<QuantizedTensor>> {
    pub fn from_config(cfg: &Config) -> Self {
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
            xq: vec![QuantizedTensor { q: vec![0; cfg.dim], s: vec![0.0; cfg.dim / 64] }; 1],
            hq: vec![QuantizedTensor { q: vec![0; cfg.dim], s: vec![0.0; cfg.hidden_dim / 64] }; 1],
        }
    }

}

impl TransformerWeights<Vec<f32>, Vec<f32>> {
    pub fn from_file(f: &mut BufReader<File>, c: &Config) -> TransformerWeights<Vec<f32>, Vec<f32>> {
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
            wcls_exists: !c.shared_weight,
            wcls: {
                if c.shared_weight { vec![1.0] } else {
                    read_vec::<f32>(f, c.vocab_size * c.dim)
                }
            },
            q_token: vec![1.0; c.vocab_size * c.dim],
        }
    }

}