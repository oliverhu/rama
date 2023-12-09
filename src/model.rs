
// // pub mod model {
//     use std::{io::{prelude::*, BufReader, Read, Result, stdout, stdin}, fs::File, collections::HashMap, path::Path};

//     use byteorder::ByteOrder;

//     pub struct TransformerWeights {
//         token_embedding_table: Vec<f32>,
//         rms_att_weight: Vec<f32>,
//         rms_ffn_weight: Vec<f32>,

//         wq: Vec<f32>,
//         wk: Vec<f32>,
//         wv: Vec<f32>,
//         wo: Vec<f32>,

//         w1: Vec<f32>,
//         w2: Vec<f32>,
//         w3: Vec<f32>,

//         rms_final_weight: Vec<f32>,
//         freq_cis_real: Vec<f32>,
//         freq_cis_imag: Vec<f32>,
//         wcls: Option<Vec<f32>>,
//     }

//     impl TransformerWeights {
//         pub fn from_file(f: &mut BufReader<File>, c: &Config) -> Self {
//             let head_size = c.dim / c.n_heads;
//             Self {
//                 token_embedding_table: read_vec(f, c.vocab_size * c.dim),
//                 rms_att_weight: read_vec(f, c.n_layers * c.dim),
//                 wq: read_vec(f, c.n_layers * c.dim * c.dim),
//                 wk: read_vec(f, c.n_layers * c.dim * c.dim),
//                 wv: read_vec(f, c.n_layers * c.dim * c.dim),
//                 wo: read_vec(f, c.n_layers * c.dim * c.dim),
//                 rms_ffn_weight: read_vec(f, c.n_layers * c.dim),
//                 w1: read_vec(f, c.n_layers * c.dim * c.hidden_dim),
//                 w2: read_vec(f, c.n_layers * c.dim * c.hidden_dim),
//                 w3: read_vec(f, c.n_layers * c.dim * c.hidden_dim),
//                 rms_final_weight: read_vec(f, c.dim),
//                 freq_cis_real: read_vec(f, c.seq_len * head_size / 2),
//                 freq_cis_imag: read_vec(f, c.seq_len * head_size / 2),
//                 wcls: {
//                     if c.shared_weight { None } else {
//                         Some(read_vec::<f32>(f, c.vocab_size * c.dim))
//                     }
//                 },
//             }
//         }

//         // move the weights into GPU.
//         // fn into_gpu(&self, gpu: &GPU) {

//         // }
//     }

//     pub struct RunState {
//         x: Vec<f32>,
//         xb: Vec<f32>,
//         xb2: Vec<f32>,
//         hb: Vec<f32>,
//         hb2: Vec<f32>,
//         q: Vec<f32>,
//         k: Vec<f32>,
//         v: Vec<f32>,
//         att: Vec<f32>,
//         logits: Vec<f32>,
//         key_cache: Vec<f32>,
//         value_cache: Vec<f32>,
//     }

//     impl RunState {
//         pub fn from_config(cfg: &Config) -> Self {
//             let kv_dim = cfg.dim * cfg.n_kv_heads / cfg.n_heads;
//             Self {
//                 x: vec![0.0; cfg.dim as usize],
//                 xb: vec![0.0; cfg.dim as usize],
//                 xb2: vec![0.0; cfg.dim as usize],
//                 hb: vec![0.0; cfg.hidden_dim as usize],
//                 hb2: vec![0.0; cfg.hidden_dim as usize],
//                 q: vec![0.0; cfg.dim as usize],
//                 k: vec![0.0; cfg.dim as usize],
//                 v: vec![0.0; cfg.dim as usize],
//                 att: vec![0.0; cfg.n_heads * cfg.seq_len as usize],
//                 logits: vec![0.0; cfg.vocab_size as usize],
//                 key_cache: vec![0.0; cfg.n_layers * cfg.seq_len * kv_dim as usize],
//                 value_cache: vec![0.0; cfg.n_layers * cfg.seq_len * kv_dim as usize],
//             }
//         }

//     }

//     pub struct Transformer {
//         config: Config,
//         weights: TransformerWeights,
//         state: RunState,
//         // fd: u16,
//         // data: Vec<f32>,
//         // file_size: usize,
//     }

//     impl Transformer {

//         pub fn from_file(cp_path: &str) -> Self {
//             let rd = &mut BufReader::new(File::open(cp_path).unwrap());
//             let config = Config::from_file(rd);
//             let weights = TransformerWeights::from_file(rd, &config);
//             let state = RunState::from_config(&config);
//             Self {
//                 config: config,
//                 weights: weights,
//                 state: state,
//             }
//         }

//     }


//     pub struct Config {
//         dim: usize,
//         hidden_dim: usize,
//         n_layers: usize,
//         n_heads: usize,
//         n_kv_heads: usize,
//         vocab_size: usize,
//         seq_len: usize,
//         shared_weight: bool,
//     }

// impl Config {
//     pub fn from_file(f: &mut BufReader<File>) -> Self {
//         let mut shared_weight = false;
//         let c  = Self {

//             dim: read::<i32>(f) as usize,
//             hidden_dim: read::<i32>(f) as usize,
//             n_layers: read::<i32>(f) as usize,
//             n_heads: read::<i32>(f) as usize,
//             n_kv_heads: read::<i32>(f) as usize,
//             vocab_size: {
//                 let vocab = read::<i32>(f);
//                 if vocab > 0 {
//                     shared_weight = true;
//                     vocab as usize
//                 } else {
//                     vocab.abs() as usize
//                 }
//             },
//             seq_len: read::<i32>(f) as usize,
//             shared_weight: false,
//         };
//         Self {
//             shared_weight: shared_weight,
//             ..c
//         }
//     }
// }

// trait FromBytes {
//     fn from_bytes(bytes: [u8; 4]) -> Self;
// }

// impl FromBytes for f32 {
//     fn from_bytes(bytes: [u8; 4]) -> Self {
//         f32::from_le_bytes(bytes)
//     }
// }

// impl FromBytes for u32 {
//     fn from_bytes(bytes: [u8; 4]) -> Self {
//         u32::from_le_bytes(bytes)
//     }
// }

// impl FromBytes for i32 {
//     fn from_bytes(bytes: [u8; 4]) -> Self {
//         i32::from_le_bytes(bytes)
//     }
// }

// fn read<T: FromBytes>(rd:&mut BufReader<File>) -> T {
//     let mut buffer = [0u8; 4];
//     rd.read_exact(&mut buffer).expect("error reading file");
//     T::from_bytes(buffer)
// }

// fn read_vec<T: FromBytes>(rd: &mut BufReader<File>, size: usize) -> Vec<T> {
//     (0..size).map(|_| read::<T>(rd)).collect()
// }

// fn read_n<R>(reader: R, bytes_to_read: usize) -> Result<Vec<u8>>
// where
//     R: Read,
// {
//     let mut buf = vec![];
//     let mut chunk = reader.take(bytes_to_read as u64);
//     let n = chunk.read_to_end(&mut buf)?;
//     assert_eq!(bytes_to_read, n);
//     Ok(buf)
// }


// ///
// /// Tokenizer
// ///
// #[derive(Debug, Default)]
// struct Tokenizer {
//     vocab: Vec<String>,
//     vocab_scores: Vec<f32>,
//     word_token_map: HashMap<String, usize>,

//     max_token_length: usize,
// }

// impl Tokenizer {
//     fn new(tokenizer_path: &str, vocab_size: usize) -> Result<Self> {
//         let path = Path::new(tokenizer_path);
//         let file = match File::open(&path) {
//             Err(er) => panic!("couldn't open {}: {}", tokenizer_path, er),
//             Ok(file) => file,
//         };
//         // Read max_token_length
//         let mut reader = BufReader::new(file);
//         let mut vocab = Tokenizer::default();

//         let max_token_length_buffer = read_n(&mut reader, std::mem::size_of::<u32>())?;
//         vocab.max_token_length = byteorder::LittleEndian::read_u32(&max_token_length_buffer) as usize;
//         for idx in 0..vocab_size {
//             let vocab_score_buffer = read_n(&mut reader, std::mem::size_of::<f32>())?;
//             let score = byteorder::LittleEndian::read_f32(&vocab_score_buffer);
//             vocab.vocab_scores.push(score);

//             let length_buffer = read_n(&mut reader, std::mem::size_of::<i32>())?;
//             let string_length = byteorder::LittleEndian::read_i32(&length_buffer);

//             let string_buffer = read_n(&mut reader, string_length as usize)?;
//             let string = String::from_utf8(string_buffer).unwrap();
//             vocab.vocab.push(string.clone());
//             vocab.word_token_map.insert(string, idx);
//         }
//         Ok(vocab)
//     }

//     /// Encode the input string. For generation, this is only the prompt.
//     /// Here BPE is used reference:
//     /// https://towardsdatascience.com/sentencepiece-tokenizer-demystified-d0a3aac19b15
//     fn encode(&self, s: &str) -> Vec<usize> {
//         let mut tokens = Vec::new();
//         tokens.reserve(s.len());
//         for c in s.trim().chars() {
//             if c == '\n' { continue; }
//             let tid = self.word_token_map.get(&(c.to_string())).unwrap();
//             tokens.push(*tid);
//         }

//         let mut str_buffer = String::new();

//         loop {
//             let mut best_score = -1e10;
//             let mut best_token_id = usize::MAX;
//             let mut best_idx = usize::MAX;

//             for idx in 0..tokens.len() - 1 {
//                 // reset buffer
//                 str_buffer.clear();

//                 let tok = tokens[idx];
//                 str_buffer.push_str(&self.vocab[tok]);
//                 str_buffer.push_str(&self.vocab[tokens[idx + 1]]);

//                 match self.word_token_map.get(&str_buffer) {
//                     Some(tid) => {
//                         if self.vocab_scores[*tid] > best_score {
//                             best_score = self.vocab_scores[*tid];
//                             best_token_id = *tid;
//                             best_idx = idx;
//                         }
//                     },
//                     None => (),
//                 }
//             }
//             // We have merged all the pairs.
//             if best_idx == usize::MAX {
//                 break;
//             }

//             // merge i and i + 1 token, remove the i + 1 token.
//             tokens[best_idx] = best_token_id;
//             tokens.remove(best_idx + 1);


//         }
//         tokens

//     }

// }


// // }