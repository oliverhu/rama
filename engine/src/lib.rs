use std::{convert::Infallible, fs::File, io::{self, BufReader}, time::SystemTime};

use axum::response::sse::Event;
use clap::builder::Str;

use crate::{device::cpu::CPU, tokenizer::bpe::Tokenizer, transformer::{state::{RunState, RunStateView, TransformerWeights, TransformerWeightsView}, Config}};
use std::io::{prelude::*, stdout};
pub mod transformer;
pub mod device;
pub mod tokenizer;
pub mod utils;
use tokio::sync::mpsc::Sender;
pub struct EngineConfig {
    /// Path to the model file
    pub model: String,

    /// Path to the model tokenizer file
    pub tokenizer: String,

    /// Initial prompt string
    pub prompt: String,

    /// Number of steps to run
    pub step: u16,

    /// (optional) The temperature [0, inf], default is 1.
    pub temperature: f32,

    /// (optional) p value in top-p sampling, default is 0.9.
    pub topp: f32,

    /// (optional) Mode: generate or chat.
    pub mode: String,

}

impl EngineConfig {
    pub fn from_model_tokenizer(model: String, tokenizer: String) -> Self {
        Self {
            model,
            tokenizer,
            prompt: "".to_string(),
            step: 255,
            temperature: 1.0,
            topp: 0.9,
            mode: "generate".to_string(),
        }
    }
}

pub fn generate(config: EngineConfig) -> io::Result<String> {
    let path = config.model;
    let token_path = config.tokenizer;

    let rd = &mut BufReader::new(File::open(path).unwrap());
    let transformer_config = Config::from_file(rd);

    #[allow(unused_variables)]
    let device = CPU {};
    #[cfg(feature="gpu")]
    let device = GPU::new();

    #[allow(unused_mut)]
    let mut weights = TransformerWeights::from_file(rd, &transformer_config);
    #[cfg(feature="gpu")]
    let weights = TransformerWeights::from_weight(&mut weights, &device);
    let mut state = RunState::from_config(&transformer_config);

    #[cfg(feature="gpu")]
    let mut state = RunState::from_state(&mut state, &device);

    #[cfg(not(feature="gpu"))]
    let wv = TransformerWeightsView::from_ws(&weights);
    #[cfg(feature="gpu")]
    let wv = TransformerWeightsView::from_gpu_ws(&weights);

    let mut rsv = RunStateView::from_rs(&mut state);

    let step = config.step;
    let prompt = config.prompt;
    let temperature = config.temperature;
    let topp = config.topp;
    let tokenizer = Tokenizer::new(&token_path, transformer_config.vocab_size).unwrap();

    let start: SystemTime = SystemTime::now();

    let response = transformer::generate(&transformer_config, &tokenizer, prompt, temperature, step.into(), topp, &wv, &mut rsv, &device);
    let elapsed = start.elapsed().unwrap();
    println!("\n--------------------------------");
    println!("elapsed: {}.{:03} s, avg tok/s: {}",
            elapsed.as_secs(), elapsed.subsec_millis(),
            (step - 1) as f32 / elapsed.as_secs_f32());
    response
}

// Streaming version of generate
pub async fn generate_stream(config: EngineConfig, sender: Sender<Result<Event, Infallible>>) {
    let path = config.model;
    let token_path = config.tokenizer;

    let rd = &mut BufReader::new(File::open(path).unwrap());
    let transformer_config = Config::from_file(rd);

    #[allow(unused_variables)]
    let device = CPU {};
    #[cfg(feature="gpu")]
    let device = GPU::new();

    #[allow(unused_mut)]
    let mut weights = TransformerWeights::from_file(rd, &transformer_config);
    #[cfg(feature="gpu")]
    let weights = TransformerWeights::from_weight(&mut weights, &device);
    let mut state = RunState::from_config(&transformer_config);

    #[cfg(feature="gpu")]
    let mut state = RunState::from_state(&mut state, &device);

    #[cfg(not(feature="gpu"))]
    let wv = TransformerWeightsView::from_ws(&weights);
    #[cfg(feature="gpu")]
    let wv = TransformerWeightsView::from_gpu_ws(&weights);

    let mut rsv = RunStateView::from_rs(&mut state);

    let step = config.step;
    let prompt = config.prompt;
    let temperature = config.temperature;
    let topp = config.topp;
    let tokenizer = Tokenizer::new(&token_path, transformer_config.vocab_size).unwrap();

    let start: SystemTime = SystemTime::now();

    transformer::generate_stream(&transformer_config, &tokenizer, prompt, temperature, step.into(), topp, &wv, &mut rsv, &device, sender).await;
    let elapsed = start.elapsed().unwrap();
    println!("\n--------------------------------");
    println!("elapsed: {}.{:03} s, avg tok/s: {}",
            elapsed.as_secs(), elapsed.subsec_millis(),
            (step - 1) as f32 / elapsed.as_secs_f32());

}