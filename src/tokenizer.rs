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

    use std::{collections::HashMap, fs};

    use super::Tokenizer;

    pub struct SentencePieceTokenizer {
        token_to_id: HashMap<String, usize>,
        id_to_token: HashMap<usize, String>,
    }

    impl SentencePieceTokenizer {
        pub fn new(model_path: &str, vocab_path: &str) -> Self {
            let _model = fs::read(model_path).expect("can't read model file");
            let vocab = fs::read(vocab_path).expect("can't read vocab file");

            let mut token_to_id: HashMap<String, usize> = HashMap::new();
            let mut id_to_token: HashMap<usize, String> = HashMap::new();

            let mut buffer: Vec<u8> = vec![];
            let mut reading_token = true;

            let mut id = 0;

            for byte in vocab {
                // tab character, now reading score
                if byte == 09 {
                    // from_utf8_lossy maybe?
                    let token = std::str::from_utf8(&buffer).unwrap();

                    // add to both mappings
                    token_to_id.insert(token.to_string(), id);
                    id_to_token.insert(id, token.to_string());

                    id += 1;

                    reading_token = false;
                    buffer.clear();

                    continue;
                }

                // newline, now reading next token
                if byte == 0x0a {
                    reading_token = true;
                    buffer.clear();

                    continue;
                }

                if reading_token {
                    buffer.push(byte);
                }
            }

            assert_eq!(token_to_id.len(), id_to_token.len());

            Self {
                token_to_id,
                id_to_token,
            }
        }
    }

    impl Tokenizer for SentencePieceTokenizer {
        fn vocab_size(&self) -> usize {
            self.token_to_id.len()
        }

        fn tokenize(&self, string: &str) -> Vec<usize> {
            todo!()
        }

        fn untokenize(&self, tokens: &[usize]) -> String {
            todo!()
        }
    }
}
