use std::collections::{HashMap, HashSet};

pub trait Tokenizer {
    fn vocab_size(&self) -> usize;
    fn tokenize(&self, string: &str) -> Vec<usize>;
    fn untokenize(&self, tokens: &[usize]) -> String;
}

pub struct SimpleTokenizer {
    vocab_size: usize,
    ch_to_int: HashMap<char, usize>,
    int_to_ch: HashMap<usize, char>,
}

impl SimpleTokenizer {
    pub fn new(dataset: &str) -> Self {
        let mut chars = dataset
            .chars()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        chars.sort();
        let int_to_ch = chars
            .iter()
            .enumerate()
            .map(|(i, ch)| (i, *ch))
            .collect::<HashMap<usize, char>>();
        let ch_to_int = chars
            .iter()
            .enumerate()
            .map(|(i, ch)| (*ch, i))
            .collect::<HashMap<char, usize>>();
        Self {
            vocab_size: chars.len(),
            int_to_ch,
            ch_to_int,
        }
    }
}

impl Tokenizer for SimpleTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn tokenize(&self, string: &str) -> Vec<usize> {
        string
            .chars()
            .map(|ch| self.ch_to_int.get(&ch).unwrap().clone())
            .collect()
    }
    fn untokenize(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|tkn| self.int_to_ch.get(tkn).unwrap().clone())
            .collect()
    }
}

pub struct AsciiTokenizer;

impl Tokenizer for AsciiTokenizer {
    fn vocab_size(&self) -> usize {
        128
    }
    fn tokenize(&self, string: &str) -> Vec<usize> {
        string
            .chars()
            .map(|ch| u8::try_from(ch).unwrap() as usize)
            .collect()
    }
    fn untokenize(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|tkn| char::from(TryInto::<u8>::try_into(*tkn).unwrap()))
            .collect()
    }
}

pub mod sentencepiece {
    //! Our in-house implementation of the Unigram SentencePiece tokenizer.
    //! It's able to read SentencePiece-generated models and vocabularies.

    use super::Tokenizer;
    use std::{
        collections::HashMap,
        fs::File,
        io::{BufRead, BufReader},
    };

    // VocabEntry::0 being the ID / index
    // VocabEntry::1 being the score
    type VocabEntry = (usize, f32);

    struct VocabMap {
        tokens: Vec<String>,
        indices: HashMap<String, VocabEntry>,
    }

    impl VocabMap {
        fn new() -> Self {
            Self {
                tokens: Vec::new(),
                indices: HashMap::new(),
            }
        }

        fn add_entry(&mut self, token: String, entry: VocabEntry) {
            self.tokens.push(token.clone());
            self.indices.insert(token.clone(), (entry.0, entry.1));
        }

        fn get_entry(&self, token: &str) -> Option<VocabEntry> {
            self.indices.get(token).copied()
        }

        fn get_token(&self, id: usize) -> Option<&str> {
            self.tokens.get(id).map(String::as_str)
        }
    }

    pub struct SentencePieceTokenizer {
        vocab_map: VocabMap,
    }

    impl SentencePieceTokenizer {
        pub fn new(_model_path: &str, vocab_path: &str) -> Self {
            // let _model = fs::read(model_path).expect("can't read model file");

            // read vocab
            let vocab_file = File::open(vocab_path).expect("can't read vocab file");
            let vocab_reader = BufReader::new(vocab_file);

            let mut vocab_map = VocabMap::new();

            for (idx, line) in vocab_reader.lines().enumerate() {
                let line = line.unwrap();
                let mut split = line.splitn(2, "\t");

                let token = split.nth(0).unwrap();
                let score = split.nth(0).unwrap().parse::<f32>().unwrap();

                vocab_map.add_entry(token.to_string(), (idx, score));
            }

            Self { vocab_map }
        }
    }

    impl Tokenizer for SentencePieceTokenizer {
        fn vocab_size(&self) -> usize {
            self.vocab_map.tokens.len()
        }

        fn tokenize(&self, string: &str) -> Vec<usize> {
            todo!()
        }

        fn untokenize(&self, tokens: &[usize]) -> String {
            todo!()
        }
    }
}
