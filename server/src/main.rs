use axum::{
    extract::{Query, State}, http::StatusCode, routing::{get, post}, Router
};
use clap::Parser;
use engine::{generate_stream, EngineConfig};

use axum::response::sse::{Event, Sse};
use axum_extra::{headers, TypedHeader};
use futures::stream::Stream;
use std::{collections::HashMap, convert::Infallible};

#[derive(Parser, Debug)]
#[command(long_about = None)]
struct Args {
    /// Path to the model checkpoint file
    #[arg(short, long)]
    model: String,

    /// Path to the model tokenizer file
    #[arg(short, long)]
    tokenizer: String,

    /// (optional) Number of steps to run
    #[arg(short, long, default_value_t = 255)]
    step: u16,

    /// (optional) The temperature [0, inf], default is 1.
    #[arg(short('a'), long, default_value = "0.0.0.0:3000")]
    address: String,

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


#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let model = &args.model;
    let tokenizer = &args.tokenizer;
    let topp = args.topp;
    let step = args.step;
    let temperature = args.temperature;
    let mode = args.mode;
    let address = args.address;

    let cfg = EngineConfig {
        model: model.to_string(),
        tokenizer: tokenizer.to_string(),
        step,
        temperature,
        topp,
        mode,
    };

    let app = Router::new()
        .route("/", get(home))
        .route("/gen", get(gen))
        .route("/chat", post(chat))
        .with_state(cfg);
    let listener = tokio::net::TcpListener::bind(address).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn home() -> &'static str {
    "Hello world! Welcome to Rama!"
}

async fn gen(
    TypedHeader(user_agent): TypedHeader<headers::UserAgent>,
    Query(params): Query<HashMap<String, String>>,
    State(config): State<EngineConfig>
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    println!("`{}` connected", user_agent.as_str());
    let prompt = params.get("prompt").cloned().unwrap_or("".to_owned());
    let (sender, mut receiver) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(16);
    let stream = async_stream::stream! {
        while let Some(event) = receiver.recv().await {
            yield event
        }
    };

    tokio::spawn(async move {
        generate_stream(config, prompt, sender).await;
    });


    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .text("keep-alive-text"),
    )
}

async fn chat(body: String) -> (StatusCode, String) {
    (StatusCode::CREATED, body)
}
