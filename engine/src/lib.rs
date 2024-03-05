use std::{fs::File, io::BufReader, time::SystemTime};

use crate::{device::cpu::CPU, tokenizer::bpe::Tokenizer, transformer::{state::{RunState, RunStateView, TransformerWeights, TransformerWeightsView}, Config}};
pub mod transformer;
pub mod device;
pub mod tokenizer;
pub mod utils;
pub struct EngineConfig {
    model: String,

    /// Path to the model tokenizer file
    tokenizer: String,

    /// Initial prompt string
    prompt: String,

    /// Number of steps to run
    step: u16,

    /// (optional) The temperature [0, inf], default is 1.
    temperature: f32,

    /// (optional) p value in top-p sampling, default is 0.9.
    topp: f32,

    /// (optional) Mode: generate or chat.
    mode: String,

}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: "".to_string(),
            tokenizer: "".to_string(),
            prompt: "hello!".to_string(),
            step: 255,
            temperature: 1.0,
            topp: 0.9,
            mode: "generate".to_string(),
        }
    }
}

pub fn generate(config: EngineConfig) {
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
    let tokenizer = Tokenizer::new(&token_path, transformer_config.vocab_size).unwrap();

    let start: SystemTime = SystemTime::now();

    let _ = transformer::generate(&transformer_config, &tokenizer, prompt, temperature, step.into(), &wv, &mut rsv, &device);
    let elapsed = start.elapsed().unwrap();
    println!("\n--------------------------------");
    println!("elapsed: {}.{:03} s, avg tok/s: {}",
            elapsed.as_secs(), elapsed.subsec_millis(),
            (step - 1) as f32 / elapsed.as_secs_f32());


}