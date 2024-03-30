#[cfg(feature = "gpu")]
pub mod hbm;
pub mod ram;
pub mod state;
pub mod infer;
pub mod ram_q80;

use std::{convert::Infallible, fs::File, io::{self, BufReader}, ops::{Bound, Range, RangeBounds}, time::Duration};
use async_channel::Sender;
use axum::response::sse::Event;

use crate::{device::device::Device, tokenizer::bpe::{decode, Tokenizer}, transformer::infer::forward, utils::read::*};
use std::io::{prelude::*, stdout};

use self::state::{RunStateView, TransformerWeightsView};

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
        (Bound::Unbounded, Bound::Included(e)) => 0..*e + 1,
        (Bound::Unbounded, Bound::Excluded(e)) => 0..*e,
        (Bound::Unbounded, Bound::Unbounded) => 0..max_len,
    }
}

impl<'a, T: Storage> View<'a, T> {
    pub fn slice(&self, range: impl RangeBounds<usize>) -> View<'_, T> {
        let r = range_from(range, self.data.length());

        View {
            data: self.data,
            range: r,
        }
    }

    pub fn new(storage: &'a T) -> View<'a, T> {
        View {
            data: storage,
            range: 0..storage.length()
        }
    }
}

pub trait Storage {
    fn length(&self) -> usize;
}

impl<'a, MT: Storage> MutView<'a, MT> {
    pub fn as_view(&self) -> View<'_, MT> {
        View {
            data: self.data,
            range: self.range.clone(),
        }
    }
    pub fn slice(&self, range: impl RangeBounds<usize>) -> View<'_, MT> {
        let r = range_from(range, self.data.length());
        View {
            data: self.data,
            range: r,
        }
    }

    pub fn mut_slice(&mut self, range: impl RangeBounds<usize>) -> MutView<'_, MT> {
        let r = range_from(range, self.data.length());
        MutView {
            data: self.data,
            range: r,
        }
    }

    pub fn new(storage: &mut MT) -> MutView<'_, MT> {
        let le = storage.length();
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
    fn length(&self) -> usize {
        self.len()
    }
}

impl Storage for Vec<f32> {
    fn length(&self) -> usize {
        self.len()
    }
}

#[derive(Clone, Debug)]
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
    pub fn from_file(f: &mut BufReader<File>) -> Self {
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

pub fn generate<'a, T: Storage, D: Device<T>>(cfg: &Config,
    tokenizer: &Tokenizer,
    prompt: String,
    temperature: f32,
    steps: usize,
    topp: f32,
    wv: &TransformerWeightsView<'a, T>,
    rsv: &mut RunStateView<'a, T>,
    device: &D
) -> io::Result<String>
    {
    let prompt_tokens = if prompt.len() > 0 { tokenizer.encode(&prompt) } else { Vec::new() };

    let mut token = 1;
    let mut pos = 0;
    let mut next;
    let mut response = "".to_owned();

    while pos < steps {
        forward(cfg, wv, rsv, token, pos, device);

        if pos < prompt_tokens.len() {
            next = prompt_tokens[pos];
        } else {
            next = device.sample(cfg, rsv, temperature, topp);
        }

        let mut token_str = tokenizer.vocab[next].clone();
        token_str = decode(token_str);
        response += &token_str;
        print!("{}", token_str);
        stdout().flush()?;

        token = next;
        pos += 1;
    };
    Ok(response)
}

#[allow(dead_code)]
pub async fn generate_stream<'a, T: Storage, D: Device<T>>(cfg: &Config,
    tokenizer: &Tokenizer,
    prompt: String,
    temperature: f32,
    steps: usize,
    topp: f32,
    wv: &TransformerWeightsView<'a, T>,
    rsv: &mut RunStateView<'a, T>,
    device: &D,
    sender: Sender<Result<Event, Infallible>>
) {
    let prompt_tokens = if prompt.len() > 0 { tokenizer.encode(&prompt) } else { Vec::new() };

    let mut token = 1;
    let mut pos = 0;
    let mut next;
    let mut response = "".to_owned();

    while pos < steps {
        forward(cfg, wv, rsv, token, pos, device);

        if pos < prompt_tokens.len() {
            next = prompt_tokens[pos];
        } else {
            next = device.sample(cfg, rsv, temperature, topp);
        }

        let mut token_str = tokenizer.vocab[next].clone();
        token_str = decode(token_str);
        response += &token_str;
        print!("{}", token_str);
        let _ = stdout().flush();

        token = next;
        pos += 1;
        let _ = sender.send(Ok(Event::default().data(token_str.replace("\n", "\\n")))).await;
        tokio::time::sleep(Duration::from_millis(1)).await;

    };
}
