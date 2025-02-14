use femto_gpt::graph::GraphError;

#[cfg(not(feature = "gpu"))]
fn main() -> Result<(), GraphError> {
    use femto_gpt::gpt::{TrainingState, GPT};
    use femto_gpt::optimizer::AdamW;
    use femto_gpt::tokenizer::{SimpleTokenizer, Tokenizer};
    use std::fs;
    use std::io::prelude::*;
    use std::path::Path;

    let training_state_path = Path::new("training_state.dat");

    let mut rng = rand::thread_rng();

    // Create a unique char-to-int mapping for all unique characters inside our dataset
    let dataset_char =
        fs::read_to_string("dataset.txt").expect("Should have been able to read the file");
    let tokenizer = SimpleTokenizer::new(&dataset_char);

    let dataset = tokenizer.tokenize(&dataset_char);

    let batch_size = 32;

    let num_tokens = 64;
    let vocab_size = tokenizer.vocab_size();
    let embedding_degree = 64;
    let num_layers = 4;
    let num_heads = 4;
    let head_size = embedding_degree / num_heads;
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
        dropout,
        AdamW::new(),
    )?;

    println!("Number of parameters: {}", gpt.num_params());

    // Load training data from train_data directory (If exists)
    // If you want to reuse training_data of a smaller model in a bigger model, you may
    // first start again with a new optimizer by setting load_optimizer=false
    // WARN: YOU CAN ONLY REUSE THE WEIGHTS OF A MODEL WITH DIFFERENT NUM-LAYERS!
    // IT'S NOT POSSIBLE TO CHANGE OTHER PROPERTIES ONCE THE MODEL IS TRAINED!
    if training_state_path.is_file() {
        let mut ts_file = fs::File::open(training_state_path).unwrap();
        let mut bytes = Vec::new();
        ts_file.read_to_end(&mut bytes).unwrap();
        let ts: TrainingState<AdamW> = bincode::deserialize(&bytes).unwrap();
        gpt.set_training_state(ts, true)?;
    }

    println!();
    println!("Starting the training loop... (This make take hours to converge! be patient!)");
    println!();

    let base_lr = 0.001;
    let min_lr = 0.00001;
    let warmup_steps = 100;
    let decay_steps = 50000;

    // Training loop!
    gpt.train(
        &dataset,
        100000,
        batch_size,
        None, // or Some(n), limit backward process to last n computations
        |step| {
            if step < warmup_steps {
                (base_lr / warmup_steps as f32) * step as f32
            } else {
                // Fancy LR tuning, thanks to https://github.com/cutoken!
                f32::max(
                    min_lr,
                    base_lr
                        - (base_lr - min_lr) * (step - warmup_steps) as f32 / decay_steps as f32,
                )
            }
        },
        |gpt| {
            let mut rng = rand::thread_rng();
            let inference_temperature = 0.5; // How creative? 0.0 min 1.0 max

            println!("Generating text:");

            let inference = gpt.infer(
                &mut rng,
                &tokenizer.tokenize("\n"),
                200,
                inference_temperature,
                |_ch| {},
            )?;

            // Generate 100 character with the currently trained model before
            // starting the training loop.
            println!("{}", tokenizer.untokenize(&inference));

            println!("Saving the model...");
            let ts = gpt.get_training_state().unwrap();
            let bytes = bincode::serialize(&ts).unwrap();
            fs::write(training_state_path, &bytes).expect("Unable to write file");

            Ok(())
        },
    )?;

    Ok(())
}

#[cfg(feature = "gpu")]
fn main() -> Result<(), GraphError> {
    use femto_gpt::funcs::*;
    use femto_gpt::graph::gpu;
    use femto_gpt::tensor::*;
    let mut graph = gpu::GpuGraph::new()?;
    let a = graph.alloc(Tensor::constant(&[10, 10], 64.), "a".into())?;
    let b = graph.alloc(Tensor::constant(&[10, 10], 36.), "b".into())?;
    let c = graph.call(Add::new(), &[a, b])?;
    graph.forward(false)?;
    println!("{:?}", graph.fetch(c)?);
    Ok(())
}
