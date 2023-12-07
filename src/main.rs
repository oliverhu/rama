use clap::Parser;
use rayon::prelude::*;
use core::f32;
use std::collections::HashMap;
use std::fs::File;
use std::time::SystemTime;
use std::io::{prelude::*, BufReader, Read, Result, stdout, stdin};
use std::path::Path;
use byteorder::ByteOrder;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
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
    let transformer = &mut Transformer::from_file(path);

    let config = &transformer.config;
    let step = args.step;
    let prompt = args.prompt;
    let temperature = args.temperature;
    let tokenizer = Tokenizer::new(token_path, config.vocab_size).unwrap();
    if args.mode == "generate" {
        let start: SystemTime = SystemTime::now();
        let _ = generate(transformer, &tokenizer, prompt, temperature, step.into());
        let elapsed = start.elapsed().unwrap();
        println!("\n--------------------------------");
        println!("elapsed: {}.{:03} s, avg tok/s: {}",
             elapsed.as_secs(), elapsed.subsec_millis(),
             (step - 1) as f32 / elapsed.as_secs_f32());
    } else {
        let _ = chat(transformer, &tokenizer, step.into());
    }
}

fn chat(transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    steps: usize) -> Result<()>
{
    let mut user_turn;
    let mut next = 0;
    let mut user_index = 0;
    let mut token;
    let mut pos;
    let rng_seed = 100;
    let mut rng = ChaCha20Rng::seed_from_u64(rng_seed);

    let mut system_prompt = String::new();
    let mut user_prompt = String::new();
    let mut rendered_prompt;
    let mut prompt_tokens = Vec::new();

    print!("Enter system prompt (optional): ");
                stdout().lock().flush()?;
                stdin().read_line(&mut system_prompt)?;

    loop {
        pos = 0;
        user_turn = true;
        while pos < steps {
            if user_turn {
                print!("\nUser: ");
                stdout().lock().flush()?;
                stdin().read_line(&mut user_prompt)?;

                if user_prompt.contains("EOS") {
                    println!("Assistant: ending chat...");
                    return Ok(());
                }

                // Render user/system prompts into llama2 chat schema
                if pos == 0 && system_prompt.len() > 0 {
                    rendered_prompt = format!("[INST] <<SYS>>{}<</SYS>>{} [/INST]", system_prompt, user_prompt);
                } else {
                    rendered_prompt = format!("[INST] {} [/INST]", user_prompt);
                }

                prompt_tokens = tokenizer.encode(&rendered_prompt);
                user_turn = false;
                user_index = 0;

                print!("\nAssistant: ");
                stdout().lock().flush()?;
            }

            if user_index < prompt_tokens.len() {
                token = prompt_tokens[user_index];
                user_index += 1;
            } else {
                token = next;
            }

            if token == 2 { user_turn = true; }

            forward(transformer, token, pos);
            next = sample_top_q(&transformer.state.logits, transformer.config.vocab_size, 0.9, &mut rng);
            pos += 1;

            if user_index >= prompt_tokens.len() && next != 2 {
                let mut token_str = tokenizer.vocab[next].clone();
                token_str = decode(token_str);
                print!("{}", token_str);
                stdout().flush()?;
            }
            if next == 2 {
                println!("")
            };
        }
    }

}

fn generate(mut transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    prompt: String,
    temperature: f32,
    steps: usize) -> Result<()> {
    let prompt_tokens = if prompt.len() > 0 { tokenizer.encode(&prompt) } else { Vec::new() };

    let mut token = 1;
    let mut pos = 0;
    let mut next;
    let rng_seed = 100;
    let mut rng = ChaCha20Rng::seed_from_u64(rng_seed);

    while pos < steps {
        forward(&mut transformer, token, pos);
        if pos < prompt_tokens.len() {
            next = prompt_tokens[pos];
        } else {
            if temperature == 0.0 {
                // greedy decoding, choose argmax
                next = transformer.state.logits.iter().enumerate()
                    .reduce(|(i1, v1), (i2, v2)| if v1 > v2 { (i1, v1) } else { (i2, v2) })
                    .map(|(i, _)| i).unwrap();
            } else {
                // temperature scaling
                if temperature < 1.0 {
                    transformer.state.logits.iter_mut().for_each(|z| *z /= temperature);
                }
                // compute probabilities
                softmax(&mut transformer.state.logits);
                // next = sample(&transformer.state.logits, &mut rng);
                next = sample_top_q(&transformer.state.logits, transformer.config.vocab_size, 0.9, &mut rng);

            }
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


fn sample_top_q(probabilities: &Vec<f32>, num: usize, topp: f32, rng: &mut ChaCha20Rng) -> usize {
    let cutoff = (1.0f32 - topp) / ((num - 1) as f32);
    let mut prob_index = probabilities.iter().enumerate().filter(
        |(_, &p)| p > cutoff
    ).collect::<Vec<(usize, &f32)>>();
    prob_index.sort_by(
        |(_, &a2), (_, &b2)|
        b2.partial_cmp(&a2).unwrap()
    );

    let mut cum_prob = 0.0f32;
    let mut last_index = prob_index.len() - 1;
    for i in 0..prob_index.len() {
        cum_prob += prob_index[i].1;
        if cum_prob > topp {
            last_index = i;
            break;
        }
    }

    let r = rng.gen::<f32>() * cum_prob;
    let mut cdf = 0.0f32;

    for i in 0..last_index {
        cdf += prob_index[i].1;
        if r < cdf {
            return prob_index[i].0;
        }
    }

    return prob_index[last_index].0;


}

// Naive sampling worked well for tinystories but performs really bad for llama model inference.
// fn sample(probabilities: &Vec<f32>, rng: &mut ChaCha20Rng) -> usize {
//     let r = rng.gen::<f32>();
//     let mut cdf = 0.0;
//     for (i, &p) in probabilities.iter().enumerate() {
//         cdf += p;
//         if r < cdf {
//             return i;
//         }
//     }
//     probabilities.len() - 1
// }

///
/// LLaMA architecture explained:
/// https://www.youtube.com/watch?v=Mn_9W1nCFLo
///
fn forward(transformer: &mut Transformer, token: usize, pos: usize) {

    let cfg = &transformer.config;
    let weights: &TransformerWeights = &transformer.weights;
    let state = &mut transformer.state;
    let dim = cfg.dim;

    let hidden_dim = cfg.hidden_dim;
    let head_size = dim / cfg.n_heads;

    // Load the embedding into x.
    state.x.copy_from_slice(
        &weights.token_embedding_table
                [(token * cfg.dim)..((token + 1) * cfg.dim)],
    );

    // Positional encoding
    let pos_real = &weights.freq_cis_real[pos*(head_size/2)..(pos+1)*(head_size/2)];
    let pos_img = &weights.freq_cis_imag[pos*(head_size/2)..(pos+1)*(head_size/2)];

    // Forward through the layers.
    for layer in 0..cfg.n_layers {

        // Apply layer normalization. a.k.a we normalize by data items (rows).
        // In comparison, batch normalization normalizes by features (columns);
        // Since we don't need to recenter but only rescaling, we use RMSNorm than
        // actual layer norm, and it worked well in practice.
        rmsnorm(&mut state.xb, &state.x, &weights.rms_att_weight[layer * dim..(layer + 1) * dim]);

        // Calculate Q, K, V
        matmul(&mut state.q, &weights.wq[layer * dim * dim..(layer + 1) * dim * dim], &state.xb, dim);
        matmul(&mut state.k, &weights.wk[layer * dim * dim..(layer + 1) * dim * dim], &state.xb, dim);
        matmul(&mut state.v, &weights.wv[layer * dim * dim..(layer + 1) * dim * dim], &state.xb, dim);

        // RoPE relative positional encoding. https://arxiv.org/pdf/2104.09864.pdf
        // b/c in attention we only care about the distance between two words, we use relative attention.
        // Rotary position embeddings are only applied to the Q & Ks, not the values,
        // also, they are applied after the multi with weights. (The vanilla transformer is different,
        // positional embeddings were applied before multiplication with weights.)
        for h in 0..cfg.n_heads {
            let q = &mut state.q[h * head_size..(h + 1) * head_size];
            let k = &mut state.k[h * head_size..(h + 1) * head_size];

            for i in 0..(head_size / 2) {
                // Instead of doing matmul, this is a more simplified way (from the paper above)
                let (fcr, fci) = (pos_real[i], pos_img[i]);
                (q[i * 2], q[i * 2 + 1]) = (
                    q[i * 2] * fcr - q[i * 2 + 1] * fci,
                    q[i * 2] * fci + q[i * 2 + 1] * fcr);
                (k[i * 2], k[i * 2 + 1]) = (
                    k[i * 2] * fcr - k[i * 2 + 1] * fci,
                    k[i * 2] * fci + k[i * 2 + 1] * fcr);
            }
        }

        let lo = layer * cfg.seq_len * dim;
        state.key_cache[(lo + pos * dim)..(lo + (pos + 1) * dim)].copy_from_slice(&state.k);
        state.value_cache[(lo + pos * dim)..(lo + (pos + 1) * dim)].copy_from_slice(&state.v);

        // Multihead attention.
        let mut atts: Vec<&mut [f32]> = state.att.chunks_mut(cfg.seq_len).collect();
        let qs: Vec<&mut [f32]> = state.q.chunks_mut(head_size).collect();
        let xbs: Vec<&mut [f32]> = state.xb.chunks_mut(head_size).collect();
        atts.par_iter_mut().zip(xbs).enumerate().for_each(|(h, (att,xb))| {
            let q = &qs[h];
            for t in 0..(pos + 1) {
                let ko = lo + t * dim + h * head_size;
                let k = &state.key_cache[ko..(ko + head_size)];
                att[t] = q.iter().zip(k.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>() / (head_size as f32).sqrt();
            }
            softmax(&mut att[..(pos + 1)]);
            xb.fill(0.0);
            for t in 0..(pos + 1) {
                let ko = lo + t * dim + h * head_size;
                let v = &state.value_cache[ko..(ko + head_size)];
                let a = att[t];
                xb.iter_mut().zip(v).for_each(|(xbi, &vi)| *xbi += a * vi);
            }

        });
        matmul(&mut state.xb2, &weights.wo[layer * dim * dim..(layer + 1) * dim * dim], &state.xb, dim);

        state.x.iter_mut().zip(state.xb2.iter()).for_each(|(a, b)| *a += *b);

        // pre ffn rmsnorm
        rmsnorm(&mut state.xb, &state.x, &weights.rms_ffn_weight[layer * dim .. (layer + 1) * dim]);

        // ffn
        matmul(&mut state.hb,  &weights.w1[layer * hidden_dim * dim..(layer + 1) * hidden_dim * dim], &state.xb, dim);
        matmul(&mut state.hb2, &weights.w3[layer * hidden_dim * dim..(layer + 1) * hidden_dim * dim], &state.xb, dim);

        // silu
        state.hb.par_iter_mut().for_each(|a|*a = *a * (1.0 / (1.0 + (-*a).exp())));

        state.hb.iter_mut().zip(state.hb2.iter()).for_each(|(a, &b)| *a *= b);

        matmul(&mut state.xb,  &weights.w2[layer *dim*hidden_dim..(layer + 1)*dim*hidden_dim], &state.hb, hidden_dim);

        state.x.iter_mut().zip(state.xb.iter()).for_each(|(a, &b)| *a += b);
    }

    // final rmsnorm
    state.xb.copy_from_slice(&state.x);
    rmsnorm(&mut state.x, &state.xb, &weights.rms_final_weight);

    // compute logits
    let wcls = match &weights.wcls {
        Some(wcls) => wcls,
        None => &weights.token_embedding_table,
    };
    matmul(&mut state.logits,  wcls,&state.x, dim);
}

struct Config {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
    shared_weight: bool,
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

struct TransformerWeights {
    token_embedding_table: Vec<f32>,
    rms_att_weight: Vec<f32>,
    rms_ffn_weight: Vec<f32>,

    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,

    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,

    rms_final_weight: Vec<f32>,
    freq_cis_real: Vec<f32>,
    freq_cis_imag: Vec<f32>,
    wcls: Option<Vec<f32>>,
}

impl TransformerWeights {
    fn from_file(f: &mut BufReader<File>, c: &Config) -> Self {
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
            wcls: {
                if c.shared_weight { None } else {
                    Some(read_vec::<f32>(f, c.vocab_size * c.dim))
                }
            },
        }

    }
}

struct RunState {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl RunState {
    fn from_config(cfg: &Config) -> Self {
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
        }
    }

}

struct Transformer {
    config: Config,
    weights: TransformerWeights,
    state: RunState,
    // fd: u16,
    // data: Vec<f32>,
    // file_size: usize,
}

impl Transformer {

    fn from_file(cp_path: &str) -> Self {
        let rd = &mut BufReader::new(File::open(cp_path).unwrap());
        let config = Config::from_file(rd);
        let weights = TransformerWeights::from_file(rd, &config);
        let state = RunState::from_config(&config);
        Self {
            config: config,
            weights: weights,
            state: state,
        }
    }

}

///
/// MATH FUNCTIONS
///

// RMSNORM
fn rmsnorm(o: &mut Vec<f32>, x: &Vec<f32>, weight: &[f32]) {
    let v: f32 = 1.0f32 / (x.iter().map(|x| x * x ).sum::<f32>() / x.len() as f32 + 1e-5f32).sqrt();
    for i in 0..o.len() {
        o[i] = weight[i] * (v * x[i]);
    }
}

fn softmax(x: &mut [f32]) {
    let max = x.par_iter().copied().reduce(|| x[0], |a, b| a.max(b));
    x.par_iter_mut().for_each(|a| *a=(*a-max).exp());
    let sum = x.par_iter().sum::<f32>();
    x.par_iter_mut().for_each(|a| *a /= sum);
}

// Matrix Multiplication
// W (d,n) @ x (n,) -> xout (d,)
fn matmul(o: &mut Vec<f32>, w: &[f32], x: &Vec<f32>, n: usize) {
    o.par_iter_mut().enumerate().for_each(|(idx, o)| {
        let mut val = 0.0f32;
        for j in 0..n {
            val += w[idx * n + j] * x[j]
        }
        *o = val;
    });
}

///
/// Tokenizer
///
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

trait FromBytes {
    fn from_bytes(bytes: [u8; 4]) -> Self;
}

impl FromBytes for f32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        f32::from_le_bytes(bytes)
    }
}

impl FromBytes for u32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        u32::from_le_bytes(bytes)
    }
}

impl FromBytes for i32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        i32::from_le_bytes(bytes)
    }
}

fn read<T: FromBytes>(rd:&mut BufReader<File>) -> T {
    let mut buffer = [0u8; 4];
    rd.read_exact(&mut buffer).expect("error reading file");
    T::from_bytes(buffer)
}

fn read_vec<T: FromBytes>(rd: &mut BufReader<File>, size: usize) -> Vec<T> {
    (0..size).map(|_| read::<T>(rd)).collect()
}

fn read_n<R>(reader: R, bytes_to_read: usize) -> Result<Vec<u8>>
where
    R: Read,
{
    let mut buf = vec![];
    let mut chunk = reader.take(bytes_to_read as u64);
    let n = chunk.read_to_end(&mut buf)?;
    assert_eq!(bytes_to_read, n);
    Ok(buf)
}

// print the time of a function.
// fn benchmark<G>(f: G, name: String)
// where
// G: FnOnce()
// {
//     let start = Instant::now();
//     f();
//     let duration: Duration = start.elapsed();
//     println!("Time elapsed in {} is: {:?}", name, duration);
// }


#[cfg(test)]
mod tests {
    // use super::*;
    #[test]
    fn test_matrix_mul() {

    }

    // fn test_softmax() {}
    // fn test_rmsnorm() {}
}
