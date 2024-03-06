use axum::{
    extract::Query, http::StatusCode, routing::{get, post}, Router
};
use engine::{generate, generate_stream, EngineConfig};

use axum::response::sse::{Event, Sse};
use axum_extra::{headers, TypedHeader};
use futures::stream::Stream;
use std::{collections::HashMap, convert::Infallible, pin::Pin, task::{Context, Poll}};

use tokio::sync::oneshot::error::TryRecvError;
use tokio::sync::mpsc::Receiver;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let app = Router::new()
        .route("/", get(home))
        .route("/generate", post(gen))
        .route("/gen", get(gen2))
        .route("/chat", post(chat));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn home() -> &'static str {
    "Hello world! Welcome to Rama!"
}


async fn gen2(
    TypedHeader(user_agent): TypedHeader<headers::UserAgent>,
    Query(params): Query<HashMap<String, String>>
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    println!("`{}` connected", user_agent.as_str());
    let prompt = params.get("prompt").cloned().unwrap_or("".to_owned());
    let (sender, receiver) = tokio::sync::mpsc::channel(1000);

    tokio::spawn(async move {
        let mut config = EngineConfig::from_model_tokenizer(
            "/home/pi/py/llama2.c/stories15M.bin".to_owned(),
            "/home/pi/rama/engine/tokenizer.bin".to_owned()
        );
        config.prompt = prompt.to_owned();
        generate_stream(config, sender).await;
    });

    let mut reusable_receiver = ReusableReceiver::new(receiver);
    let str = reusable_receiver.stream();
    Sse::new(str).keep_alive(
        axum::response::sse::KeepAlive::new()
            // .interval(Duration::from_secs(1))
            .text("keep-alive-text"),
    )
}

async fn gen(
    body: String,
    // TypedHeader(user_agent): TypedHeader<headers::UserAgent>,
) -> (StatusCode, String) {
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


pub struct ReusableReceiver<T> {
    receiver: Option<Receiver<T>>,
    receiver_oneshot: Option<tokio::sync::oneshot::Receiver<Receiver<T>>>,
}

pub struct ReusableReceiverStream<T> {
    receiver: Option<Receiver<T>>,
    sender_oneshot: Option<tokio::sync::oneshot::Sender<Receiver<T>>>,
}

impl<T> ReusableReceiver<T> where T: Sync + Send + 'static {
    pub fn new(receiver: Receiver<T>) -> Self {
        Self {
            receiver: Some(receiver),
            receiver_oneshot: None,
        }
    }

    pub fn stream(&mut self) -> Pin<Box<dyn Stream<Item=T> + Send + Sync + 'static>> {
        if self.receiver.is_none() {
            self.recover_receiver();
        }

        let (sender_oneshot, receiver_oneshot) = tokio::sync::oneshot::channel();

        self.receiver_oneshot = Some(receiver_oneshot);

        Box::pin(ReusableReceiverStream {
            receiver: std::mem::take(&mut self.receiver),
            sender_oneshot: Some(sender_oneshot),
        })
    }

    fn recover_receiver(&mut self) {
        let mut receiver_oneshot = std::mem::take(&mut self.receiver_oneshot)
            .expect("unexpected situation, need to be fixed");

        loop {
            match receiver_oneshot.try_recv() {
                Err(TryRecvError::Closed) => {
                    return;
                }
                Err(TryRecvError::Empty) => {}
                Ok(receiver) => {
                    self.receiver = Some(receiver);
                    return;
                }
            }
        }
    }
}

impl<T> Drop for ReusableReceiverStream<T> {
    fn drop(&mut self) {
        let receiver = std::mem::take(&mut self.receiver)
            .expect("unexpected situation, need to be fixed");
        let sender_oneshot = std::mem::take(&mut self.sender_oneshot);
        let result = sender_oneshot
            .expect("unexpected situation, need to be fixed")
            .send(receiver);
        if let Err(error) = result {
            println!("{:?}", error);
        }
    }
}

impl<T> Stream for ReusableReceiverStream<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver
            .as_mut()
            .expect("unexpected situation, need to be fixed")
            .poll_recv(cx)
    }
}
