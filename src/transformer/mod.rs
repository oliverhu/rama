#[cfg(feature = "gpu")]
pub mod hbm;
pub mod ram;
pub mod state;

use std::{fs::File, io::BufReader, ops::{Range, RangeBounds, Bound}};
use crate::utils::read::*;

use self::state::RunState;


// TODO: probably rename CPU/GPU to CPU/GPU "storage" later. The only difference
// here is where we store the weights, CPU RAM or GPU HBM.
pub trait Transformer {
    // fn forward(&mut self, token: usize, pos: usize);
    fn from_file(cp_path: &str) -> Self;
    fn get_config(&self) -> Config;
    // fn sample(&mut self, temperature: f32) -> usize;
    fn cpu_state(&self) -> &RunState<Vec<f32>>;
}

pub struct View<'a, T: Storage> {
    pub data: &'a T,
    pub range: Range<usize>,
}

pub struct MutView<'a, MT> where MT: Storage {
    pub data: &'a mut MT,
    pub range: Range<usize>,
}

pub fn range_from(range: impl RangeBounds<usize>, max_len: usize) -> Range<usize> {
    let start = range.start_bound();
    let end = range.end_bound();

    match (start, end) {
        (Bound::Included(s), Bound::Included(e)) => *s..*e + 1,
        (Bound::Included(s), Bound::Excluded(e)) => *s..*e,
        (Bound::Included(s), Bound::Unbounded) => *s..max_len,
        (Bound::Excluded(s), Bound::Included(e)) => *s + 1..*e + 1,
        (Bound::Excluded(s), Bound::Excluded(e)) => *s + 1..*e,
        (Bound::Excluded(s), Bound::Unbounded) => *s + 1..max_len,
        (Bound::Unbounded, Bound::Included(e)) => 0..max_len,
        (Bound::Unbounded, Bound::Excluded(e)) => 0..max_len - 1,
        (Bound::Unbounded, Bound::Unbounded) => 0..max_len,
    }
}

impl<'a, T: Storage> View<'a, T> {
    pub fn slice(&self, range: impl RangeBounds<usize>) -> View<'_, T> {
        let r = range_from(range, self.data.len());

        View {
            data: self.data,
            range: r,
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
    pub fn slice(&self, range: impl RangeBounds<usize>) -> View<'_, MT> {
        let r = range_from(range, self.data.len());
        View {
            data: self.data,
            range: r,
        }
    }

    pub fn mut_slice(&mut self, range: impl RangeBounds<usize>) -> MutView<'_, MT> {
        let r = range_from(range, self.data.len());
        MutView {
            data: self.data,
            range: r,
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