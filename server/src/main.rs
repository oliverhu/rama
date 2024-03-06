use axum::{
    extract::{Query, State}, http::StatusCode, routing::{get, post}, Router
};
use clap::Parser;
use engine::{generate_stream, EngineConfig};

use axum::response::sse::{Event, Sse};
use axum_extra::{headers, TypedHeader};
use futures::stream::Stream;
use std::{collections::HashMap, convert::Infallible, pin::Pin, task::{Context, Poll}};

use tokio::sync::oneshot::error::TryRecvError;
use tokio::sync::mpsc::Receiver;


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
    let (sender, receiver) = tokio::sync::mpsc::channel(1000);

    tokio::spawn(async move {
        generate_stream(config, prompt, sender).await;
    });

    let mut reusable_receiver = ReusableReceiver::new(receiver);
    let str = reusable_receiver.stream();
    Sse::new(str).keep_alive(
        axum::response::sse::KeepAlive::new()
            .text("keep-alive-text"),
    )
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
