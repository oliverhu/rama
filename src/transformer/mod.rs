#[cfg(feature = "gpu")]
pub mod hbm;
pub mod ram;

use std::{io::BufReader, fs::File, ops::Range};
use crate::utils::read::*;

use self::ram::RunState;

// TODO: probably rename CPU/GPU to CPU/GPU "storage" later. The only difference
// here is where we store the weights, CPU RAM or GPU HBM.
pub trait Transformer {
    fn forward(&mut self, token: usize, pos: usize);
    fn from_file(cp_path: &str) -> Self;
    fn get_config(&self) -> Config;
    fn sample(&mut self, temperature: f32) -> usize;
    fn cpu_state(&self) -> &RunState;
}


#[derive(Default, Debug)]
pub struct CPUStorage {
    data: Vec<f32>
}

pub trait MutView<'a> {
    type MV;

    fn as_mut_slice(&'a mut self) -> Self::MV;
    fn mut_slice(&'a mut self, range: Range<usize>) -> Self::MV;
}

pub trait View<'a> {
    type V;

    fn as_slice(&'a self) -> Self::V;
    fn slice(&'a self, range: Range<usize>) -> Self::V;
}

impl<'a> MutView<'a> for CPUStorage {
    type MV = CPUMutView<'a>;

    fn as_mut_slice(&'a mut self) -> CPUMutView<'a> {
        CPUMutView::new(&mut self.data)
    }

    fn mut_slice(&'a mut self, range: Range<usize>) -> CPUMutView<'a> {
        CPUMutView {
            data: &mut self.data,
            range: range,
        }
    }

}

impl<'a> View<'a> for CPUStorage {
    type V  = CPUView<'a>;

    fn as_slice(&'a self) -> CPUView<'a> {
        CPUView::new(&self.data)
    }

    fn slice(&'a self, range: Range<usize>) -> CPUView<'a> {
        CPUView {
            data: &self.data,
            range: range,
        }
    }

}

#[derive(Debug)]
pub struct CPUView<'a> {
    data: &'a Vec<f32>,
    range: Range<usize>,
}

impl<'a> CPUView<'a> {
    fn new(data: &'a Vec<f32>) -> Self {
        let l = data.len();
        Self {
            data: data,
            range: 0..l
        }
    }

}

impl<'a> View<'a> for CPUView<'_> {
    type V = Self;
    fn slice(&self, range: Range<usize>) -> Self {
        Self {
            data: self.data,
            range: (self.range.start + range.start)..(self.range.end + range.end)
        }
    }

    fn as_slice(&self) -> Self::V {
        Self {
            data: self.data,
            range: (self.range.start)..(self.range.end)
        }
    }

}

impl<'a> MutView<'a> for CPUMutView<'a> {
    type MV = Self;

    fn as_mut_slice(&'a mut self) -> Self::MV {
        todo!()
    }

    fn mut_slice(&'a mut self, range: Range<usize>) -> Self::MV {
        Self {
            data: self.data,
            range: (self.range.start + range.start)..(self.range.end + range.end)
        }
    }

}

#[derive(Debug)]
pub struct CPUMutView<'a> {
    data: &'a mut Vec<f32>,
    range: Range<usize>,
}

impl<'a> CPUMutView<'a> {
    fn new(data: &'a mut Vec<f32>) -> Self {
        let l = data.len();
        Self {
            data: data,
            range: 0..l
        }
    }
    #[allow(dead_code)]
    fn slice(&'a self, range: Range<usize>) -> CPUView {
        CPUView {
            data: self.data,
            range: (self.range.start + range.start)..(self.range.end + range.end)
        }
    }


}

impl AsRef<[f32]> for CPUStorage {
    fn as_ref(&self) -> &[f32] {
        &self.data.as_slice()
    }
}

impl AsMut<[f32]> for CPUStorage {
    fn as_mut(&mut self) -> &mut [f32] {
        self.data.as_mut_slice()
    }
}

impl AsRef<[f32]> for CPUView<'_> {
    fn as_ref(&self) -> &[f32] {
        &self.data[self.range.start..self.range.end]
    }
}

impl AsMut<[f32]> for CPUMutView<'_> {
    fn as_mut(&mut self) -> &mut [f32] {
        &mut self.data[self.range.start..self.range.end]
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