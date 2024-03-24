// Batcher batches request from the web server and send over to the engine for batch processing.

use std::time::Duration;

use async_channel::Receiver;

/// batch the request up to a size or wait time before sending out to the engine for reference.
pub async fn get_batch(
    receiver: &Receiver<String>,
    prompts: &mut Vec<String>,
    batch_size: usize,
    wait_time: Duration,
) {
    let _ = tokio::time::timeout(wait_time, batch_helper(receiver, prompts, batch_size)).await;

}

async fn batch_helper(
    receiver: &Receiver<String>,
    prompts: &mut Vec<String>,
    batch_size: usize,
) {
    loop {
        match receiver.recv().await {
            Ok(prompt) => {
                prompts.push(prompt.clone());
                println!("received --> {}", prompt);
            }
            Err(_) => {
                println!("error in proccessing");
            }
        }
        if prompts.len() == batch_size {
            break;
        }
    }

}
