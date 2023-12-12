#[cfg(feature = "gpu")]
pub mod gpu;
pub mod cpu;

use std::{io::BufReader, fs::File};
use crate::utils::read::*;

// TODO: probably rename CPU/GPU to CPU/GPU "storage" later. The only difference
// here is where we store the weights, CPU RAM or GPU HBM.
pub trait Transformer {
    fn forward(&mut self, token: usize, pos: usize);
    fn from_file(cp_path: &str) -> Self;
    fn get_config(&self) -> Config;
    fn sample(&mut self, temperature: f32) -> usize;
}

#[derive(Clone)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub shared_weight: bool,
}

impl Config {
    fn from_file(f: &mut BufReader<File>) -> Self {
        let mut shared_weight = false;
        let c  = Self {

            dim: read::<i32>(f) as usize,
            hidden_dim: read::<i32>(f) as usize,
            n_layers: read::<i32>(f) as usize,
            n_heads: read::<i32>(f) as usize,
            n_kv_heads: read::<i32>(f) as usize,
            vocab_size: {
                let vocab = read::<i32>(f);
                if vocab > 0 {
                    shared_weight = true;
                    vocab as usize
                } else {
                    vocab.abs() as usize
                }
            },
            seq_len: read::<i32>(f) as usize,
            shared_weight: false,
        };
        Self {
            shared_weight: shared_weight,
            ..c
        }
    }
}