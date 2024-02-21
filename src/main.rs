use clap::Parser;
use device::cpu::CPU;
use device::device::Device;
#[cfg(feature="gpu")]
use device::gpu::GPU;
use transformer::state::{RunState, RunStateView, TransformerWeights, TransformerWeightsView};
use transformer::{Config, Storage};

use core::f32;
use std::collections::HashMap;
use std::fs::File;
use std::time::SystemTime;
use std::io::{prelude::*, BufReader, Result, stdout, stdin};
use std::path::Path;
use byteorder::ByteOrder;
mod device;
mod transformer;
mod utils;
use utils::read::read_n;

use crate::transformer::infer::forward;

#[derive(Parser, Debug)]
#[command(long_about = None)]
struct Args {
    /// Path to the model checkpoint file
    #[arg(short, long)]
    model: String,

    /// Path to the model tokenizer file
    #[arg(short, long)]
    tokenizer: String,

    /// Initial prompt string
    #[arg(short, long, default_value = "")]
    prompt: String,

    /// Number of steps to run
    #[arg(short, long, default_value_t = 255)]
    step: u16,

    /// (optional) The temperature [0, inf], default is 1.
    #[arg(short('r'), long, default_value_t = 1.0)]
    temperature: f32,

    /// (optional) p value in top-p sampling, default is 0.9.
    #[arg(short('l'), long, default_value_t = 0.9)]
    topp: f32,

    /// (optional) Mode: generate or chat.
    #[arg(short('o'), long, default_value = "generate")]
    mode: String,
}

// Options:
//   -m, --model <MODEL>              Path to the model checkpoint file
//   -t, --tokenizer <TOKENIZER>      Path to the model checkpoint file
//   -p, --prompt <PROMPT>            Initial prompt string
//   -s, --step <STEP>                Number of steps to run [default: 1]
//   -r, --temperature <TEMPERATURE>  Path to the model checkpoint file [default: 0.9]
//   -h, --help                       Print help
//   -V, --version                    Print version

fn main() {
    let args = Args::parse();
    let path = &args.model;
    let token_path = &args.tokenizer;

    let rd = &mut BufReader::new(File::open(path).unwrap());
    let config = Config::from_file(rd);
    let weights = TransformerWeights::from_file(rd, &config);
    let mut state = RunState::from_config(&config);

    // #[cfg(not(feature="gpu"))]
    // #[cfg(feature="gpu")]
    let wv = TransformerWeightsView::from_ws(&weights);
    #[cfg(feature="gpu")]
    let wv = TransformerWeightsView::from_gpu_ws(&weights);
    let mut rsv = RunStateView::from_rs(&mut state);

    let step = args.step;
    let prompt = args.prompt;
    let temperature = args.temperature;
    let tokenizer = Tokenizer::new(token_path, config.vocab_size).unwrap();

    let start: SystemTime = SystemTime::now();

    let device = CPU {};
    #[cfg(feature="gpu")]
    let device = GPU::new();

    let _ = generate(&config, &tokenizer, prompt, temperature, step.into(), &wv, &mut rsv, &device);
    let elapsed = start.elapsed().unwrap();
    println!("\n--------------------------------");
    println!("elapsed: {}.{:03} s, avg tok/s: {}",
            elapsed.as_secs(), elapsed.subsec_millis(),
            (step - 1) as f32 / elapsed.as_secs_f32());

}

fn generate<'a, T: Storage, D: Device<T>>(cfg: &Config,
    tokenizer: &Tokenizer,
    prompt: String,
    temperature: f32,
    steps: usize,
    wv: &TransformerWeightsView<'a, T>,
    rsv: &mut RunStateView<'a, T>,
    device: &D
) -> Result<()>
    {
    let prompt_tokens = if prompt.len() > 0 { tokenizer.encode(&prompt) } else { Vec::new() };

    let mut token = 1;
    let mut pos = 0;
    let mut next;

    while pos < steps {
        forward(cfg, wv, rsv, token, pos, device);

        if pos < prompt_tokens.len() {
            next = prompt_tokens[pos];
        } else {
            next = device.sample(cfg, rsv, temperature);
        }

        let mut token_str = tokenizer.vocab[next].clone();
        token_str = decode(token_str);
        print!("{}", token_str);
        stdout().flush()?;

        token = next;
        pos += 1;
    };
    Ok(())


}

// Tokenizer
#[derive(Debug, Default)]
struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    word_token_map: HashMap<String, usize>,

    max_token_length: usize,
}

impl Tokenizer {
    fn new(tokenizer_path: &str, vocab_size: usize) -> Result<Self> {
        let path = Path::new(tokenizer_path);
        let file = match File::open(&path) {
            Err(er) => panic!("couldn't open {}: {}", tokenizer_path, er),
            Ok(file) => file,
        };
        // Read max_token_length
        let mut reader = BufReader::new(file);
        let mut vocab = Tokenizer::default();

        let max_token_length_buffer = read_n(&mut reader, std::mem::size_of::<u32>())?;
        vocab.max_token_length = byteorder::LittleEndian::read_u32(&max_token_length_buffer) as usize;
        for idx in 0..vocab_size {
            let vocab_score_buffer = read_n(&mut reader, std::mem::size_of::<f32>())?;
            let score = byteorder::LittleEndian::read_f32(&vocab_score_buffer);
            vocab.vocab_scores.push(score);

            let length_buffer = read_n(&mut reader, std::mem::size_of::<i32>())?;
            let string_length = byteorder::LittleEndian::read_i32(&length_buffer);

            let string_buffer = read_n(&mut reader, string_length as usize)?;
            let string = String::from_utf8(string_buffer).unwrap();
            vocab.vocab.push(string.clone());
            vocab.word_token_map.insert(string, idx);
        }
        Ok(vocab)
    }

    /// Encode the input string. For generation, this is only the prompt.
    /// Here BPE is used reference:
    /// https://towardsdatascience.com/sentencepiece-tokenizer-demystified-d0a3aac19b15
    fn encode(&self, s: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        tokens.reserve(s.len());
        for c in s.trim().chars() {
            if c == '\n' { continue; }
            let tid = self.word_token_map.get(&(c.to_string())).unwrap();
            tokens.push(*tid);
        }

        let mut str_buffer = String::new();

        loop {
            let mut best_score = -1e10;
            let mut best_token_id = usize::MAX;
            let mut best_idx = usize::MAX;

            for idx in 0..tokens.len() - 1 {
                // reset buffer
                str_buffer.clear();

                let tok = tokens[idx];
                str_buffer.push_str(&self.vocab[tok]);
                str_buffer.push_str(&self.vocab[tokens[idx + 1]]);

                match self.word_token_map.get(&str_buffer) {
                    Some(tid) => {
                        if self.vocab_scores[*tid] > best_score {
                            best_score = self.vocab_scores[*tid];
                            best_token_id = *tid;
                            best_idx = idx;
                        }
                    },
                    None => (),
                }
            }
            // We have merged all the pairs.
            if best_idx == usize::MAX {
                break;
            }

            // merge i and i + 1 token, remove the i + 1 token.
            tokens[best_idx] = best_token_id;
            tokens.remove(best_idx + 1);


        }
        tokens

    }

}

///
/// Utility functions
///
fn decode(str: String) -> String {
    let lower_case = str.to_ascii_lowercase();
    let raw_bytes = lower_case.as_bytes();

    // handle tokens in the form of "<0xAB>"
    if str.contains("<s>") { return String::new(); }
    if str.len() > 0
        && char::from(raw_bytes[0]) == '<'
        && char::from(raw_bytes[raw_bytes.len() - 1]) == '>' {
            let c = u8::from_str_radix(&str[3..5], 16).unwrap();
            char::from(c).to_string()
    } else {
        str
    }

}
