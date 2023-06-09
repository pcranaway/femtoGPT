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
        // self.model
        //     .encode(string)
        //     .unwrap()
        //     .iter()
        //     .map(|p| p.id as usize)
        //     .collect()

        let tokens = self.model.encode(string).unwrap();
        let mut result = Vec::with_capacity(tokens.len());

        result.extend(tokens.iter().map(|p| p.id as usize));

        result
    }

    fn untokenize(&self, tokens: &[usize]) -> String {
        let pieces: Vec<u32> = tokens.into_iter().map(|tkn| *tkn as u32).collect();

        self.model.decode_piece_ids(&pieces).unwrap()
    }
}

pub struct HFTokenizer {
    hf_tokenizer: tokenizers::Tokenizer,
}

impl HFTokenizer {
    pub fn new(model: &str) -> Self {
        Self {
            hf_tokenizer: tokenizers::Tokenizer::from_pretrained(model, None).unwrap(),
        }
    }
}

impl Tokenizer for HFTokenizer {
    fn vocab_size(&self) -> usize {
        self.hf_tokenizer.get_vocab_size(true)
    }

    fn tokenize(&self, string: &str) -> Vec<usize> {
        // encode
        let encoded = self.hf_tokenizer.encode(string, true).unwrap();

        let tokens = encoded.get_ids();
        // or encoded.get_word_ids()
        // not sure which one is correct

        // convert result from HF tokenizers into usize vec instead of a u32 one, because for
        // whatever reason that's what HF tokenizers gives us.
        let mut result = Vec::with_capacity(tokens.len());

        result.extend(tokens.iter().map(|p| *p as usize));

        result
    }

    fn untokenize(&self, tokens: &[usize]) -> String {
        dbg!(tokens);

        // convert token IDs to u32 because HF tokenizers expect that for whatever reason
        let mut ids = Vec::with_capacity(tokens.len());

        ids.extend(tokens.iter().map(|p| *p as u32));

        // actually decode
        let decoded = self.hf_tokenizer.decode(ids, true).unwrap();

        decoded
    }
}
