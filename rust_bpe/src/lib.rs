// Rust BPE: pre-tokenization + parallel pair counting
//
// Strategy:
//   1. Rust reads the file, pre-tokenizes, and counts word frequencies (parallel).
//   2. Rust returns a HashMap<Vec<u8>, u32> mapping each pre-token's bytes to its count.
//   3. Python builds pair_counts and pair_to_words from word_counts, then runs the merge loop.
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
//    - Output: word_counts (dict[bytes, int])
// 3. Read the file from disk
// 4. Split the file into chunks at <|endoftext|> token boundaries
//    (for parallel processing across CPU cores)
// 5. Within each chunk, apply the GPT-2 PAT regex to get pre-tokens
//    Each pre-token is a "word" — a sequence of bytes
// 6. Count word frequencies in parallel using fold/reduce
// 7. Return to Python:
//    - word_counts: dict[bytes, int] — mapping each pre-token's bytes to its count

use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;
use rayon::prelude::*;
use fancy_regex::Regex;
use std::fs::File;
use memmap2::Mmap;


#[pyfunction]
fn pre_tokenize(file_path: String, special_tokens: Vec<String>) -> PyResult<PyObject> {
    let t_total = Instant::now();
    
    let t_read = Instant::now();

    let t_read = Instant::now();
    let file = File::open(&file_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let text: &str = unsafe { std::str::from_utf8_unchecked(&mmap) };
    println!("  [Rust 1/6] File mmapped: {:.1}s ({} bytes)", t_read.elapsed().as_secs_f64(), text.len());

    // let text = fs::read_to_string(&file_path)?;
    // println!("  [Rust 1/6] File read: {:.1}s ({} bytes)", t_read.elapsed().as_secs_f64(), text.len());
    
    let t_regex = Instant::now();
    let pat = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap();
    println!("  [Rust 2/6] Regex compiled: {:.3}s", t_regex.elapsed().as_secs_f64());
    // let pre_tokens: Vec<&str> = pat.find_iter(&text).map(|m| m.as_str()).collect();

    let mut segments: Vec<&str> = vec![text];

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

    // Parallel word counting: each segment produces a local HashMap<Vec<u8>, u32>,
    // then all local maps are merged by summing values.
    let t_words = Instant::now();
    let word_counts: HashMap<Vec<u8>, u32> = segments.par_iter().fold(
        || HashMap::new(),
        |mut local_counts, seg| {
            // Skip special tokens — they are not pre-tokenized by the regex
            let is_special = special_tokens.iter().any(|s| s == *seg);
            if is_special {
                return local_counts;
            }
            for m in pat.find_iter(seg).filter_map(|m| m.ok()) {
                let word_bytes = m.as_str().as_bytes().to_vec();
                *local_counts.entry(word_bytes).or_insert(0) += 1;
            }
            local_counts
        }
    ).reduce(
        || HashMap::new(),
        |mut a, b| {
            for (word, count) in b {
                *a.entry(word).or_insert(0) += count;
            }
            a
        }
    );
    println!("  [Rust 4/4] Word counting (parallel): {:.1}s ({} unique words)", t_words.elapsed().as_secs_f64(), word_counts.len());

    println!("  [Rust] Total Rust time: {:.1}s", t_total.elapsed().as_secs_f64());
    
    Python::with_gil(|py| {
        Ok(word_counts.into_pyobject(py)?.into())
    })
}

#[pymodule]
fn rust_bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pre_tokenize, m)?)?;
    Ok(())
}