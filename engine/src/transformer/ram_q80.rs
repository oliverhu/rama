use std::{fs::File, io::{BufReader, Read}};

use super::{read_vec, state::TransformerWeights, Config, FromBytes, Storage};

// Each group (64) of tensors is stored in [i8; 64] + one f32 as scaling factor. Compression rate is about 4 * 64 / (1 * 64 + 4) ~= 4.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct QuantizedTensor {
    pub q: Vec<i8>, // quantized value
    pub s: Vec<f32>, // scaling factor
}

impl Storage for Vec<QuantizedTensor> {
    fn length(&self) -> usize {
        //TODO - MUST FIX
        self.len()
    }
}

const GS: i32 = 64;
fn read_q80_tensor(rd: &BufReader<File>, size_each: usize) -> QuantizedTensor {
    let mut qt = QuantizedTensor::default();
    let s_size: i32 = size_each as i32 / GS;

    qt.q = (0..size_each).map(|_| {
            let mut buffer = [0u8; 1];
            rd.read_exact(&mut buffer).expect("error reading file");
            i8::from_le_bytes(buffer)
        }
    ).collect();
    qt.s = (0..size_each).map(|_| {
        let mut buffer = [0u8; 4];
        rd.read_exact(&mut buffer).expect("error reading file");
        f32::from_le_bytes(buffer)
    }
    ).collect();
    qt
}

fn read_q80_vec(rd: &BufReader<File>, n_layer: usize, size_each: usize) -> Vec<QuantizedTensor> {
    (0..n_layer).map(|_| read_q80_tensor(rd, size_each)).collect()

}

impl TransformerWeights<Vec<f32>, Vec<QuantizedTensor>> {
    pub fn from_file(f: &mut BufReader<File>, c: &Config) -> Self {
        let head_size = c.dim / c.n_heads;
        Self {
            token_embedding_table: read_vec(f, c.vocab_size * c.dim),
            rms_att_weight: read_vec(f, c.n_layers * c.dim),
            wq: read_q80_vec(f, c.n_layers, c.dim * c.dim),
            wk: read_q80_vec(f, c.n_layers, c.dim * c.dim),
            wv: read_q80_vec(f, c.n_layers, c.dim * c.dim),
            wo: read_q80_vec(f, c.n_layers, c.dim * c.dim),
            rms_ffn_weight: read_vec(f, c.n_layers * c.dim),
            w1: read_q80_vec(f, c.n_layers, c.dim * c.hidden_dim),
            w2: read_q80_vec(f, c.n_layers, c.dim * c.hidden_dim),
            w3: read_q80_vec(f, c.n_layers, c.dim * c.hidden_dim),
            rms_final_weight: read_vec(f, c.dim),
            freq_cis_real: read_vec(f, c.seq_len * head_size / 2),
            freq_cis_imag: read_vec(f, c.seq_len * head_size / 2),
            wcls_exists: !c.shared_weight,
            wcls: {
                if c.shared_weight { vec![] } else {
                    read_vec(f, c.vocab_size * c.dim)
                }
            },
        }
    }

}