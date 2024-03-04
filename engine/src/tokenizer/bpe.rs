use std::{collections::HashMap, fs::File, io::BufReader, path::Path};
use std::io::Result;

use byteorder::ByteOrder;

use crate::utils::read::read_n;

// Tokenizer
#[derive(Debug, Default)]
pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub vocab_scores: Vec<f32>,
    pub word_token_map: HashMap<String, usize>,

    pub max_token_length: usize,
}

impl Tokenizer {
    pub fn new(tokenizer_path: &str, vocab_size: usize) -> Result<Self> {
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
    pub fn encode(&self, s: &str) -> Vec<usize> {
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

// Utility functions
pub fn decode(str: String) -> String {
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
