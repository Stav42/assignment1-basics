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
use std::time::Instant;
use rayon::prelude::*;
use fancy_regex::Regex;

#[pyfunction]
fn pre_tokenize(file_path: String, special_tokens: Vec<String>) -> PyResult<PyObject> {
    let t_total = Instant::now();
    
    let t_read = Instant::now();
    let text = fs::read_to_string(&file_path)?;
    println!("  [Rust 1/6] File read: {:.1}s ({} bytes)", t_read.elapsed().as_secs_f64(), text.len());
    
    let t_regex = Instant::now();
    let pat = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap();
    println!("  [Rust 2/6] Regex compiled: {:.3}s", t_regex.elapsed().as_secs_f64());
    // let pre_tokens: Vec<&str> = pat.find_iter(&text).map(|m| m.as_str()).collect();

    let mut segments: Vec<&str> = vec![&text];

    let t_split = Instant::now();
    for special_token in &special_tokens {
        let new_segments = segments.into_par_iter().map( |segment| {
            if segment == special_token {
                return vec![segment];
            } else
            {
                let parts: Vec<&str> = segment.split(special_token).collect();
                let mut result = Vec::new();
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 {
                        result.push(special_token.as_str());
                    }
                    if !part.is_empty() {
                        result.push(*part);
                    }
                }
                return result;
            }
        } 
        ).flatten().collect();

        segments = new_segments;
    }
    println!("  [Rust 3/6] Special token split: {:.1}s ({} segments)", t_split.elapsed().as_secs_f64(), segments.len());

    let mut token_ids: Vec<usize> = Vec::new();
    let mut boundary: Vec<bool> = Vec::new();

    let t_tokenize = Instant::now();
    for seg in segments{
        let idx = special_tokens.iter().position(|s| s == seg);

        if idx.is_some() {
            token_ids.push(256 + idx.unwrap()); // Special token IDs start after byte tokens (0-255)
            boundary.push(true);
        } else {
            let mut seg_pre_tokens: Vec<&str> = pat.find_iter(seg).filter_map(|m| m.ok()).map(|m| m.as_str()).collect();
            for pre_token in seg_pre_tokens {
                let bytes = pre_token.as_bytes();
                for (j, byte) in bytes.iter().enumerate() {
                    token_ids.push(*byte as usize);
                    boundary.push(j == bytes.len() - 1);
                }
            }
        }
    }
    println!("  [Rust 4/6] Tokenization: {:.1}s ({} token_ids)", t_tokenize.elapsed().as_secs_f64(), token_ids.len());

    let t_pairs = Instant::now();
    let mut pair_counts: HashMap<(usize, usize), usize> = (0..token_ids.len().saturating_sub(1)).into_par_iter().fold(
        || HashMap::new(), // Initial empty pair count for each thread
        |mut pair_counts, token_idx| {
            if boundary[token_idx] { // Don't count pairs that cross pre-token boundaries
                return pair_counts;
            }
            let pair = (token_ids[token_idx], token_ids[token_idx + 1]);
            if (pair.0 >= 256 && pair.0 < 256 + special_tokens.len()) || 
            (pair.1 >= 256 && pair.1 < 256 + special_tokens.len()) {
                return pair_counts; // Skip pairs involving special tokens
            }
            *pair_counts.entry(pair).or_insert(0) += 1;
            pair_counts
        }).reduce(
            || HashMap::new(),
            |mut a, b| {
                for (pair, count) in b {
                    *a.entry(pair).or_insert(0) += count;
                }
                a
            }
        );
    println!("  [Rust 5/6] Pair counting: {:.1}s ({} unique pairs)", t_pairs.elapsed().as_secs_f64(), pair_counts.len());
    

    // for i in 0..token_ids.len().saturating_sub(1) {
    //     if boundary[i] { // Don't count pairs that cross pre-token boundaries
    //         continue;
    //     }
    //     let pair = (token_ids[i], token_ids[i + 1]);
    //     if (pair.0 >= 256 && pair.0 < 256 + special_tokens.len()) || 
    //     (pair.1 >= 256 && pair.1 < 256 + special_tokens.len()) {
    //         continue;
    //     }
    //     *pair_counts.entry(pair).or_insert(0) += 1;
    // }


    let encoded_special: Vec<Vec<u8>> = special_tokens.iter().map(|t| t.as_bytes().to_vec()).collect();

    println!("  [Rust 6/6] Total Rust time: {:.1}s", t_total.elapsed().as_secs_f64());
    
    Python::with_gil(|py| {
        Ok((token_ids, pair_counts, encoded_special, boundary).into_pyobject(py)?.into())
    })
}

#[pymodule]
fn rust_bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pre_tokenize, m)?)?;
    Ok(())
}