# Rama

Rama is a port of [llama.c](ttps://github.com/karpathy/llama2.c), some of the code referenced another port [here](https://github.com/leo-du/llama2.rs) from [leo-du](https://github.com/leo-du).

I created this repo to learn Rust from scratch (first piece of code in Rust after Rustling!), and get first hand experience understanding llama2 model architectures.

The repo is also annotated with learning materials and documentations.

Plan is to catch up with the performance of llama.cpp.

## Usage
```
git clone https://github.com/oliverhu/rama
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
cargo build --release
cargo run --release -- -m stories15M.bin -t tokenizer.bin -p 'once upon a time'
```

For llama2 model from Meta:
```
pip install -r requirements.txt
python export.py llama2_7b.bin --meta-llama path/to/llama/model/7B
cargo run --release -- -m llama2_7b.bin -t tokenizer.bin -p 'once upon a time'
```

## TODOs
- [ ] Support GPU inference.
- [ ] Add more comprehensive benchmark.