use femto_gpt::gpt::GPT;
use femto_gpt::tokenizer::{HFTokenizer, SentencepieceTokenizer, SimpleTokenizer, Tokenizer};

use std::fs;

fn main() {
    let mut rng = rand::thread_rng();

    // Create a unique char-to-int mapping for all unique characters inside our dataset
    let dataset_char =
        fs::read_to_string("dataset.txt").expect("Should have been able to read the file");
    // let tokenizer = HFTokenizer::new("gpt2");
    let tokenizer = SentencepieceTokenizer::new();

    let dataset = tokenizer.tokenize(&dataset_char);

    let batch_size = 16;

    let num_tokens = 64;
    let vocab_size = tokenizer.vocab_size();
    let embedding_degree = 64;
    let num_layers = 4;
    let num_heads = 4;
    let head_size = embedding_degree / num_heads;
    let hiddens = 2;
    let dropout = 0.0;

    assert_eq!(num_heads * head_size, embedding_degree);

    println!("Vocab-size: {} unique characters", vocab_size);

    let mut gpt = GPT::new(
        &mut rng,
        vocab_size,
        embedding_degree,
        num_tokens,
        num_layers,
        num_heads,
        head_size,
        hiddens,
        dropout,
        femto_gpt::optimizer::AdamW::new(),
    );

    println!("Number of parameters: {}", gpt.num_params());

    // Load training data from train_data directory (If exists)
    gpt.load();

    println!("Generating text:");

    // Generate 100 character with the currently trained model before
    // starting the training loop.
    for _ in 0..5 {
        // Generate 100 character with the currently trained model before
        // starting the training loop.
        println!(
            "> {}",
            tokenizer.untokenize(&gpt.infer(&tokenizer.tokenize("\n"), 80, |ch| {}))
        );
    }

    println!();
    println!("Starting the training loop... (This make take hours to converge! be patient!)");
    println!();

    let base_lr = 0.001;
    let min_lr = 0.00001;
    let warmup_steps = 100;
    let decay_steps = 50000;

    // Training loop!
    gpt.train(&dataset, 100000, batch_size, |step| {
        if step < warmup_steps {
            (base_lr / warmup_steps as f32) * step as f32
        } else {
            // Fancy LR tuning, thanks to https://github.com/cutoken!
            f32::max(
                min_lr,
                base_lr - (base_lr - min_lr) * (step - warmup_steps) as f32 / decay_steps as f32,
            )
        }
    });
}
