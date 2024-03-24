use std::{convert::Infallible, fs::File, io::BufReader, sync::OnceLock, time::SystemTime};
use axum::response::sse::Event;
#[cfg(feature="gpu")]
use crate::device::gpu::GPU;
use crate::{device::cpu::CPU, tokenizer::bpe::Tokenizer, transformer::{state::{RunState, RunStateView, TransformerWeights, TransformerWeightsView}, Config}};
pub mod transformer;
pub mod device;
pub mod tokenizer;
pub mod utils;
use async_channel::{Receiver, Sender};

#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Path to the model file
    pub model: String,

    /// Path to the model tokenizer file
    pub tokenizer: String,

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
            step: 255,
            temperature: 1.0,
            topp: 0.9,
            mode: "generate".to_string(),
        }
    }
}

pub struct ClientRequest {
    // The prompt in the request, in the future we need to batch request.
    pub prompt: String,

    // The sender we stream event into.
    pub sender: Sender<Result<Event, Infallible>>,
}

pub static ENGINE_SERVICE: OnceLock<EngineService> = OnceLock::new();

// EngineService is a singleton that process the queued inputs from web server
#[derive(Debug)]
pub struct EngineService {
    // Poll the requests from web server, spawn thread for each stream.
    receiver: Receiver<ClientRequest>,

    // Weights of the model in CPU memory.
    weights: TransformerWeights<Vec<f32>>,

    // Tokenizer
    tokenizer: Tokenizer,

    // configs, need to be cleaned up later.
    eng_config: EngineConfig,
    model_config: Config,
}

impl EngineService {
    pub fn global() -> &'static EngineService {
        ENGINE_SERVICE.get().expect("Engine not initialized")
    }

    pub fn new(
        eng_config: EngineConfig,
        receiver: Receiver<ClientRequest>
    ) -> Self {
        // Initialize the model and ship weights to memory.
        let path = eng_config.model.clone();
        let token_path = eng_config.tokenizer.clone();

        let rd = &mut BufReader::new(File::open(path).unwrap());
        let model_config = Config::from_file(rd);

        #[allow(unused_variables)]
        let device: CPU = CPU {};

        #[allow(unused_mut)]
        let mut weights = TransformerWeights::from_file(rd, &model_config);

        let tokenizer = Tokenizer::new(&token_path, model_config.vocab_size).unwrap();

        Self {
            receiver: receiver.clone(),
            weights,
            tokenizer,
            eng_config,
            model_config,
        }
    }

    pub fn init(&self) {
        tokio::spawn(handler());
    }


}

async fn handler() {
    let es = EngineService::global();
    loop {
        match es.receiver.recv().await {
            Ok(cr) => {
                // allocate RunState space.
                tokio::spawn(async {
                    let mut state = RunState::from_config(&es.model_config);
                    let prompt = cr.prompt;
                    println!("received prompt --> {}", prompt);

                    #[cfg(not(feature="gpu"))]
                    let wv = TransformerWeightsView::from_ws(&es.weights);

                    #[cfg(feature="gpu")]
                    let device = GPU::new();

                    #[cfg(feature="gpu")]
                    let weights = TransformerWeights::from_weight(&mut weights, &device);


                    #[cfg(feature="gpu")]
                    let mut state = RunState::from_state(&mut state, &device);

                    #[cfg(feature="gpu")]
                    let wv = TransformerWeightsView::from_gpu_ws(&weights);

                    let mut rsv = RunStateView::from_rs(&mut state);

                    let step = es.eng_config.step;
                    let temperature = es.eng_config.temperature;
                    let topp = es.eng_config.topp;
                    let device = CPU {};
                    transformer::generate_stream(&es.model_config, &es.tokenizer, prompt, temperature, step.into(), topp, &wv, &mut rsv, &device, cr.sender).await;
                });
            },
            Err(_) => {
                println!("error");
            },
        }
    }
}

// Streaming version of generate
pub async fn generate_stream(config: EngineConfig, prompt: String, sender: Sender<Result<Event, Infallible>>) {
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