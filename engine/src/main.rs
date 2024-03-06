use clap::Parser;
use device::cpu::CPU;

use tokenizer::bpe::Tokenizer;
#[cfg(feature="gpu")]
use device::gpu::GPU;
use transformer::state::{RunState, RunStateView, TransformerWeights, TransformerWeightsView};
use transformer::{Config};

use core::f32;
use std::fs::File;
use std::time::SystemTime;
use std::io::{BufReader};
mod device;
mod transformer;
mod utils;
pub mod tokenizer;



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
    let topp = args.topp;

    let rd = &mut BufReader::new(File::open(path).unwrap());
    let config = Config::from_file(rd);

    #[allow(unused_variables)]
    let device = CPU {};
    #[cfg(feature="gpu")]
    let device = GPU::new();

    #[allow(unused_mut)]
    let mut weights = TransformerWeights::from_file(rd, &config);
    #[cfg(feature="gpu")]
    let weights = TransformerWeights::from_weight(&mut weights, &device);
    let mut state = RunState::from_config(&config);

    #[cfg(feature="gpu")]
    let mut state = RunState::from_state(&mut state, &device);

    #[cfg(not(feature="gpu"))]
    let wv = TransformerWeightsView::from_ws(&weights);
    #[cfg(feature="gpu")]
    let wv = TransformerWeightsView::from_gpu_ws(&weights);

    let mut rsv = RunStateView::from_rs(&mut state);

    let step = args.step;
    let prompt = args.prompt;
    let temperature = args.temperature;
    let tokenizer = Tokenizer::new(token_path, config.vocab_size).unwrap();

    let start: SystemTime = SystemTime::now();

    let _ = transformer::generate(&config, &tokenizer, prompt, temperature, step.into(), topp, &wv, &mut rsv, &device);
    let elapsed = start.elapsed().unwrap();
    println!("\n--------------------------------");
    println!("elapsed: {}.{:03} s, avg tok/s: {}",
            elapsed.as_secs(), elapsed.subsec_millis(),
            (step - 1) as f32 / elapsed.as_secs_f32());

}

