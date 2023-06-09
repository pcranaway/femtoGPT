use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use sentencepiece::SentencePieceProcessor;

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

pub struct SentencepieceTokenizer {
    model: SentencePieceProcessor,
}

impl SentencepieceTokenizer {
    pub fn new() -> Self {
        let model = SentencePieceProcessor::open("reddit.model").unwrap();

        Self { model }
    }
}

impl Tokenizer for SentencepieceTokenizer {
    fn vocab_size(&self) -> usize {
        self.model.len()
    }
    fn tokenize(&self, string: &str) -> Vec<usize> {
        self.model
            .encode(string)
            .unwrap()
            .iter()
            .map(|p| p.id as usize)
            .collect()
    }
    fn untokenize(&self, tokens: &[usize]) -> String {
        let pieces: Vec<u32> = tokens.into_iter().map(|tkn| *tkn as u32).collect();

        self.model.decode_piece_ids(&pieces).unwrap()
    }
}
