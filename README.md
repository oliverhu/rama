# Rama

<p align="center">
  <img src="assets/rama.png" width="300" height="300" alt="Cute Llama">
</p>

Rama is composed of a Llama inference engine (the forward pass, like TensorRT) and an inference server (the web service layer, like Triton). It started as a port of [llama.c](https://github.com/karpathy/llama2.c) to understand the llama architecture and learn Rust. Later I realized the web service layer is necessary for the repo to be useful for home hosting & further my learning in Rust (async). So far the differential feature of Rama is its support for GPU inference, plus a well integrated web server. The inference server crate (./server) is under heavy construction.

## Usage
### Pre-req
Check out code, install dependenceis & get the models!
```
$ git clone https://github.com/oliverhu/rama
$ wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
$ pip install httpie # if you already have httpie installed, skip this line.
```
### Web Server + Engine Usage

The default tokenizer.bin is under `./engine/tokenizer.bin`
```
$ cargo run --bin server -- --model PATH_TO_stories15M.bin --tokenizer PATH_TO_tokenizer.bin
$ http --stream :3000/gen prompt=='I have a dog' | python server/print.py
```
Output:
```
$ http --stream :3000/gen prompt=='I have a dog' | python server/print.py
I have a dog named Spot. Spot is a good dog. He loves to play with Lily. Lily has a dog named Spot. Spot is a good dog. He is obedient.
One day, Lily and Spot go to the park. They see a big slide. Lily wants to go on the slide. She says to Spot, "Come on, Spot. Let's go on the slide." Spot barks and wags his tail. He likes Lily.
Lily climbs up the ladder. She sits on the slide. She holds Spot's leash. She says to Spot, "Ready, Spot? Let's go!" Spot barks. He jumps off the slide. He runs to the slide. He sees Lily. He runs to the slide. He jumps on the slide. He slides down. He goes very fast. He laughs.
Lily claps. She says, "Good job, Spot! You are brave!" She hugs Spot. She says, "You are a good dog, Spot. You are a good dog." Spot barks.%
```
Alternatively, you can open your browser `localhost:3000/?prompt=I have a dog,` and you will see the generated response!

### Engine Usage
Of course you can skip the inference server and only develop/use the engine!
```
cargo build --bin engine --release
cargo run --bin engine --release -- -m stories15M.bin -t tokenizer.bin -p 'once upon a time'
```
(Note release build is about 10x faster...for my Linux box, debug build gives ~35 tok/s,
release build gives 370 tok/s)

For llama2 model from Meta:
```
$ pip install -r engine/export/requirements.txt
$ python engine/export/export.py llama2_7b.bin --meta-llama path/to/llama/model/7B
$ cargo run --release -- -m llama2_7b.bin -t tokenizer.bin -p 'once upon a time'
```

pass `--features gpu` to use GPU for matrix multiplications
```
$ cargo run --bin engine --features gpu --release -- -m llama2_7b.bin -t tokenizer.bin -p 'once upon a time'
```


Sample output:
```
$ cargo run --bin engine --release -- -m llama2-7b.bin -t tokenizer.bin -p 'once upon a time' -r 0.5

    Finished release [optimized] target(s) in 0.01s
    Running `target/release/rama -m llama2-7b.bin -t tokenizer.bin -p 'once upon a time' -r 0.5`

once upon a time, there was a little boy who lived in a little town. He was a very good boy. He loved his family, his friends, and his little town..
One day, the little boy was walking down the street when he saw a beautiful little house. He had never seen a house like it before. It was so beautiful that he wanted to live in it..
The little boy walked up to the house and knocked on the door. A beautiful lady answered the door. She was the owner of the house..
The little boy told the lady that he wanted to live in her house. The lady told the little boy that he could live in her house if he could find a way to make her house even more beautiful..
The little boy thought about this for a while. He decided that he would make the lady’s house even more beautiful by making it into a castle..
The little boy went to work. He built a castle out of the house. He put a moat around the castle. He put a drawbridge over the moat. He put a drawbridge over the drawbridge. He put a drawbridge over the drawbridge..
The little boy was very happy with his castle...
```

## Attribution
The implementation referenced another llama.c port [here](https://github.com/leo-du/llama2.rs) from [leo-du](https://github.com/leo-du), and [dfdx](https://github.com/coreylowman/dfdx) from [coreylowman](https://github.com/coreylowman).

This repo was created to learn Rust and understand llama2 model architectures by code. The repo is annotated with learning materials and documentations.

Plan is to catch up with the performance of llama.cpp!

## Engine Performance
Command used to get tok/s
```
cargo run --bin engine --release --features gpu -- -m stories110M.bin  -t tokenizer.bin -p "once upon a time" -r 1 -s 200
```
Model           | Platform          | Token/s
:---------------|:------------------|:------------
stories15M.bin  | RTX 4090          | 480.81
stories15M.bin  | Ryzen 7 5700X     | 402.35
stories15M.bin  | Intel i9 13900KF  | 346.75
stories15M.bin  | M1 Macbook Pro    | 196.47
stories15M.bin  | M2 Macbook Pro    | 194.81
stories110M.bin | RTX 4090          | 201.08
stories110M.bin | RTX 4070ti/CUBLAS | 113
stories110M.bin | Intel i9 13900KF  | 86
stories110M.bin | RTX 4070ti        | 80
stories110M.bin | Ryzen 7 5700X     | 68
stories110M.bin | M2 Macbook Pro    | 52
stories110M.bin | M1 Macbook Pro    | 29
llama2-7b.bin   | Intel i9 13900KF  | 2.42
llama2-7b.bin   | Ryzen 7 5700X     | 1.38
llama2-7b.bin   | M2 Macbook Pro    | 0.12
llama2-7b.bin   | M1 Macbook Pro    | 0.02


Running llama2-7b f32 in M1 macbook is extremely slow since it requires 25GB memory but M1 only has 16GB total memory, the amount of swapping is huge.

## Server Performance
TBD. It currently uses Server Sent Event to drive generations, for a chatting experience, need to support WebSocket later.

## TODOs
- [x] Support chat interface.
- [x] Add tok/s.
- [x] Support GPU inference.
- [x] Improve GPU performance to be at least slightly faster than CPU as baseline.
- [x] Support CUBLAS for matmul.
- [x] Support SIMD for CPU.
- [ ] Support quantization.
- [ ] Support flash attention.
- [ ] Support AMD GPUs.
