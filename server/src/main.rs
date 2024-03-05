use axum::{
    routing::{get, post},
    http::StatusCode,
    Json, Router,
};
use engine::EngineConfig;
use serde::{Deserialize, Serialize};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let app = Router::new()
        .route("/", get(home))
        .route("/generate", post(generate))
        .route("/chat", post(chat));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn home() -> &'static str {
    "Hello world! Welcome to Rama!"
}

async fn generate(body: String) -> (StatusCode, String) {
    let config = EngineConfig::default();
    (StatusCode::CREATED, body)
}

async fn chat(body: String) -> (StatusCode, String) {
    (StatusCode::CREATED, body)
}
