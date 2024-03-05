use axum::{
    routing::{get, post},
    http::StatusCode,
    Json, Router,
};
use engine::{generate, EngineConfig};
use serde::{Deserialize, Serialize};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let app = Router::new()
        .route("/", get(home))
        .route("/generate", post(gen))
        .route("/chat", post(chat));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn home() -> &'static str {
    "Hello world! Welcome to Rama!"
}

async fn gen(body: String) -> (StatusCode, String) {
    let mut config = EngineConfig::from_model_tokenizer(
        "/home/pi/py/llama2.c/stories15M.bin".to_owned(),
        "/home/pi/rama/engine/tokenizer.bin".to_owned()
    );
    config.prompt = body.to_owned();
    let response = generate(config);
    (StatusCode::CREATED, response.unwrap())
}

async fn chat(body: String) -> (StatusCode, String) {
    (StatusCode::CREATED, body)
}
