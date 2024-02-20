#[cfg(feature = "gpu")]
pub mod hbm;
pub mod ram;

use std::{fs::File, io::BufReader, ops::Range};
use crate::utils::read::*;

use self::ram::RunState;

// TODO: probably rename CPU/GPU to CPU/GPU "storage" later. The only difference
// here is where we store the weights, CPU RAM or GPU HBM.
pub trait Transformer {
    // fn forward(&mut self, token: usize, pos: usize);
    fn from_file(cp_path: &str) -> Self;
    fn get_config(&self) -> Config;
    // fn sample(&mut self, temperature: f32) -> usize;
    fn cpu_state(&self) -> &RunState;
}

pub struct View<'a, T: Storage> {
    pub data: &'a T,
    pub range: Range<usize>,
}

pub struct MutView<'a, MT> where MT: Storage {
    pub data: &'a mut MT,
    pub range: Range<usize>,
}

impl<'a, T: Storage> View<'a, T> {
    pub fn slice(&self, range: Range<usize>) -> View<'_, T> {
        View {
            data: self.data,
            range,
        }
    }

    pub fn new(storage: &'a T) -> View<'a, T> {
        View {
            data: storage,
            range: 0..storage.len()
        }
    }
}

pub trait Storage {
    fn len(&self) -> usize;
}

impl<'a, MT: Storage> MutView<'a, MT> {
    pub fn as_view(&self) -> View<'_, MT> {
        View {
            data: self.data,
            range: self.range.clone(),
        }
    }
    pub fn slice(&self, range: Range<usize>) -> View<'_, MT> {
        View {
            data: self.data,
            range,
        }
    }

    pub fn mut_slice(&mut self, range: Range<usize>) -> MutView<'_, MT> {
        MutView {
            data: self.data,
            range,
        }
    }

    pub fn new(storage: &mut MT) -> MutView<'_, MT> {
        let le = storage.len();
        MutView {
            data: storage,
            range: 0..le
        }
    }

}

impl<'a, T: Storage> AsRef<T> for View<'a, T> {
    fn as_ref(&self) -> &T {
        &self.data
    }
}

impl<'a, T: Storage> AsRef<T> for MutView<'a, T> {
    fn as_ref(&self) -> &T {
        &self.data
    }
}

impl<'a, T: Storage> AsMut<T> for MutView<'a, T> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl Storage for Vec<i32> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl Storage for Vec<f32> {
    fn len(&self) -> usize {
        self.len()
    }
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