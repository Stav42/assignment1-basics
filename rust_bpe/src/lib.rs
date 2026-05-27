// Rust BPE: pre-tokenization + parallel pair counting
//
// Strategy:
//   1. Rust reads the file and pre-tokenizes + counts byte pairs (parallel).
//   2. Rust returns the full pre-tokenized text as a flat list of token IDs.
//   3. Python runs the merge loop, incrementally updating the token list.
//
// Pre-tokenization has TWO levels:
//   Level 1: Split file into chunks at <|endoftext|> boundaries
//            → chunks are independent, can be processed in parallel
//   Level 2: Within each chunk, split using GPT-2 PAT regex
//            → this ensures byte pairs are only counted WITHIN "words"
//            → prevents merges from crossing word/document boundaries
//
// GPT-2 pre-tokenization PAT regex (splits text into "words"):
//   '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//   This captures:
//     - Contractions ('s, 't, 'll, etc.)
//     - Words (letters with optional leading space)
//     - Numbers (digits with optional leading space)
//     - Punctuation (non-alphanumeric with optional leading space)
//     - Whitespace
//
// TODO list:
// 1. Import pyo3 and set up the Python bindings
// 2. Define the function signature:
//    - Input:  file_path (str), special_tokens (list[str])
//    - Output: (token_ids, pair_counts, encoded_special_tokens)
//      - token_ids: flat list of ints — the full pre-tokenized corpus as byte IDs
//      - pair_counts: dict {(int, int): int} — initial adjacent pair frequencies
//      - encoded_special_tokens: list of bytes — special tokens encoded for Python
// 3. Read the file from disk directly (memory-mapped for large files)
// 4. Split the file into chunks at <|endoftext|> token boundaries
//    (for parallel processing across CPU cores)
// 5. Within each chunk, apply the GPT-2 PAT regex to get pre-tokens
//    Each pre-token is a "word" — a sequence of bytes
// 6. Convert each byte into its ID (0-255) to build the token_ids list
// 7. Count adjacent byte pairs across all pre-tokens:
//    - Slide a window of size 2 over token_ids and count (id_a, id_b) occurrences
//    - Pairs do NOT cross pre-token boundaries
// 8. Return to Python:
//    - token_ids: list[int] — the full corpus as a flat list of token IDs
//    - pair_counts: dict {(int, int): int} — initial pair frequencies
//    - encoded_special_tokens: list of encoded special tokens

use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs;
use fancy_regex::Regex;

#[pyfunction]
fn pre_tokenize(file_path: String, special_tokens: Vec<String>) -> PyResult<PyObject> {
    let text = fs::read_to_string(&file_path)?;
    let pat = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap();
    // let pre_tokens: Vec<&str> = pat.find_iter(&text).map(|m| m.as_str()).collect();
    let pre_tokens: Vec<&str> = pat.find_iter(&text).filter_map(|m| m.ok()).map(|m| m.as_str()).collect();
    let mut token_ids: Vec<usize> = Vec::new();
    for pre_token in &pre_tokens {
        for byte in pre_token.as_bytes() {
            token_ids.push(*byte as usize);
        }
    }

    let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
    for window in token_ids.windows(2) {
        let pair = (window[0], window[1]);
        *pair_counts.entry(pair).or_insert(0) += 1;
    }

    let encoded_special: Vec<Vec<u8>> = special_tokens.iter().map(|t| t.as_bytes().to_vec()).collect();

    Python::with_gil(|py| {
        Ok((token_ids, pair_counts, encoded_special).into_pyobject(py)?.into())
    })
}

#[pymodule]
fn rust_bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pre_tokenize, m)?)?;
    Ok(())
}