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
use std::collections::{HashMap, HashSet};
use std::fs;
use std::time::Instant;
use std::sync::OnceLock;
use rayon::prelude::*;
use fancy_regex::Regex;
use std::fs::File;
use memmap2::Mmap;

// GPT-2 pre-tokenization PAT. Compiled once on first use and reused across all
// calls — fancy_regex compilation is expensive, so recompiling per call was a
// major hot-path cost in the per-document encode path.
static PAT: OnceLock<Regex> = OnceLock::new();

fn get_pat() -> &'static Regex {
    PAT.get_or_init(|| {
        Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
            .unwrap()
    })
}


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
            if special_tokens.iter().any(|s| s == segment) {
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

/// Shared pre-tokenizer (sequential, compile-once regex). Splits on the given
/// special tokens (caller passes them longest-first to handle overlaps), then
/// applies the GPT-2 PAT regex within each non-special segment. Returns the
/// ordered list of pre-token byte sequences, with special tokens kept intact.
fn pretokenize_to_bytes(text: &str, special_tokens: &[String]) -> Vec<Vec<u8>> {
    let pat = get_pat();

    // Split on special tokens (sequential).
    let mut segments: Vec<&str> = vec![text];
    for special_token in special_tokens {
        let mut new_segments: Vec<&str> = Vec::new();
        for segment in segments {
            if special_tokens.iter().any(|s| s == segment) {
                new_segments.push(segment);
            } else {
                let parts: Vec<&str> = segment.split(special_token.as_str()).collect();
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 {
                        new_segments.push(special_token.as_str());
                    }
                    if !part.is_empty() {
                        new_segments.push(*part);
                    }
                }
            }
        }
        segments = new_segments;
    }

    // Apply the PAT regex within each segment, preserving order.
    let mut pre_token_list: Vec<Vec<u8>> = Vec::new();
    for seg in &segments {
        if special_tokens.iter().any(|s| s == *seg) {
            pre_token_list.push(seg.as_bytes().to_vec());
        } else {
            for m in pat.find_iter(seg).filter_map(|m| m.ok()) {
                pre_token_list.push(m.as_str().as_bytes().to_vec());
            }
        }
    }

    pre_token_list
}

#[pyfunction]
fn pre_tokenize_string(text: String, special_tokens: Vec<String>) -> PyResult<PyObject> {
    let pre_token_list = pretokenize_to_bytes(&text, &special_tokens);
    Python::with_gil(|py| {
        Ok(pre_token_list.into_pyobject(py)?.into())
    })
}

// --- RustTokenizer: full encode() in Rust ---
//
// Holds the vocab + merge ranks and runs the entire encode pipeline natively:
//   text -> pre-tokenize -> apply merges -> Vec<u32> token IDs.
// This avoids creating thousands of Python `bytes` objects per document and
// runs the merge loop in compiled code instead of the Python interpreter.
//
// Tokens are represented as u32 IDs throughout the merge loop (not byte
// vectors), so merging is just integer comparison + hashmap lookups — no
// per-step allocation of byte sequences.
#[pyclass]
struct RustTokenizer {
    // (left_id, right_id) -> (merge rank, merged token id)
    merges_map: HashMap<(u32, u32), (u32, u32)>,
    // initial token id for each raw byte value (0..256)
    byte_id: Vec<u32>,
    // special-token bytes -> its token id
    special_to_id: HashMap<Vec<u8>, u32>,
    // special tokens as strings, longest-first (for pre-tokenization splitting)
    special_sorted: Vec<String>,
    // pre-token bytes -> cached token ids (pre-tokens repeat heavily in real text)
    cache: HashMap<Vec<u8>, Vec<u32>>,
}

impl RustTokenizer {
    /// Merge a single (non-special) pre-token into its token IDs.
    fn merge_one(&self, pre_token: &[u8]) -> Vec<u32> {
        let mut parts: Vec<u32> = pre_token.iter().map(|&b| self.byte_id[b as usize]).collect();

        while parts.len() > 1 {
            // Find the lowest-rank adjacent pair in a single pass.
            let mut best_rank = u32::MAX;
            let mut best_i = usize::MAX;
            let mut best_merged = 0u32;
            for i in 0..parts.len() - 1 {
                if let Some(&(rank, merged_id)) = self.merges_map.get(&(parts[i], parts[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = i;
                        best_merged = merged_id;
                    }
                }
            }
            if best_i == usize::MAX {
                break;
            }

            let (a, b) = (parts[best_i], parts[best_i + 1]);
            let mut new_parts: Vec<u32> = Vec::with_capacity(parts.len());
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len() && parts[i] == a && parts[i + 1] == b {
                    new_parts.push(best_merged);
                    i += 2;
                } else {
                    new_parts.push(parts[i]);
                    i += 1;
                }
            }
            parts = new_parts;
        }

        parts
    }
}

#[pymethods]
impl RustTokenizer {
    /// Build from the Python tokenizer state.
    ///   vocab:          dict[int, bytes]      (must include special tokens)
    ///   merges:         list[(bytes, bytes)]  (in merge-priority order)
    ///   special_tokens: list[str]
    #[new]
    fn new(
        vocab: HashMap<u32, Vec<u8>>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        special_tokens: Vec<String>,
    ) -> PyResult<Self> {
        // bytes -> id, for looking up merge children/results and single bytes.
        let mut byte_to_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vocab.len());
        for (id, bytes) in &vocab {
            byte_to_id.insert(bytes.clone(), *id);
        }

        // Single-byte starting ids (BPE vocab always contains all 256 bytes).
        let mut byte_id: Vec<u32> = vec![0; 256];
        for b in 0..256u32 {
            let id = *byte_to_id
                .get(&vec![b as u8])
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                    format!("vocab missing single byte {}", b),
                ))?;
            byte_id[b as usize] = id;
        }

        // (left_id, right_id) -> (rank, merged_id)
        let mut merges_map: HashMap<(u32, u32), (u32, u32)> = HashMap::with_capacity(merges.len());
        for (rank, (left, right)) in merges.iter().enumerate() {
            let left_id = *byte_to_id.get(left).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("merge left token not in vocab")
            })?;
            let right_id = *byte_to_id.get(right).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("merge right token not in vocab")
            })?;
            let mut merged = left.clone();
            merged.extend_from_slice(right);
            let merged_id = *byte_to_id.get(&merged).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("merged token not in vocab")
            })?;
            merges_map.insert((left_id, right_id), (rank as u32, merged_id));
        }

        // Special-token byte -> id, and the longest-first split order.
        let mut special_to_id: HashMap<Vec<u8>, u32> = HashMap::new();
        for s in &special_tokens {
            let bytes = s.as_bytes().to_vec();
            if let Some(&id) = byte_to_id.get(&bytes) {
                special_to_id.insert(bytes, id);
            }
        }
        let mut special_sorted = special_tokens.clone();
        special_sorted.sort_by(|a, b| b.len().cmp(&a.len()));

        Ok(RustTokenizer {
            merges_map,
            byte_id,
            special_to_id,
            special_sorted,
            cache: HashMap::new(),
        })
    }

    /// Encode text into token IDs.
    fn encode(&mut self, text: String) -> Vec<u32> {
        let pretokens = pretokenize_to_bytes(&text, &self.special_sorted);
        let mut ids: Vec<u32> = Vec::with_capacity(pretokens.len());

        for pt in pretokens {
            if let Some(&sid) = self.special_to_id.get(&pt) {
                ids.push(sid);
                continue;
            }
            if let Some(cached) = self.cache.get(&pt) {
                ids.extend_from_slice(cached);
                continue;
            }
            let merged = self.merge_one(&pt);
            ids.extend_from_slice(&merged);
            self.cache.insert(pt, merged);
        }

        ids
    }
}

// --- BpeTrainer: stateful BPE training class ---
//
// Holds all BPE training state (word_counts, pair_counts, pair_to_words, vocab, merges).
// Python calls step() repeatedly to perform one merge at a time, allowing progress logging.
// The hot loop (processing affected words) uses rayon for parallelism.

/// Sparse update produced by processing one affected word in parallel.
/// Contains the old/new word and the net pair-count deltas (only non-zero entries).
struct WordUpdate {
    old_word: Vec<u32>,
    new_word: Vec<u32>,
    count: u64,
    pair_count_deltas: Vec<((u32, u32), i64)>,
}

/// Pure function: given a word and the current best pair, compute the merged word
/// and the sparse pair-count deltas. No shared mutable state — safe for rayon.
fn compute_word_update(
    word: &[u32],
    word_counts: &HashMap<Vec<u32>, u64>,
    best_pair: (u32, u32),
    new_id: u32,
) -> WordUpdate {
    let count = *word_counts.get(word).expect("word must exist in word_counts");

    // Greedy left-to-right merge (matches Python exactly)
    let mut new_word = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && word[i] == best_pair.0 && word[i + 1] == best_pair.1 {
            new_word.push(new_id);
            i += 2;
        } else {
            new_word.push(word[i]);
            i += 1;
        }
    }

    // Compute sparse pair-count deltas
    let mut delta_map: HashMap<(u32, u32), i64> = HashMap::new();

    // Subtract all pairs from old word
    for j in 0..word.len().saturating_sub(1) {
        let pair = (word[j], word[j + 1]);
        *delta_map.entry(pair).or_insert(0) -= count as i64;
    }

    // Add all pairs from new word
    for j in 0..new_word.len().saturating_sub(1) {
        let pair = (new_word[j], new_word[j + 1]);
        *delta_map.entry(pair).or_insert(0) += count as i64;
    }

    // Keep only non-zero deltas
    let pair_count_deltas: Vec<((u32, u32), i64)> = delta_map
        .into_iter()
        .filter(|(_, d)| *d != 0)
        .collect();

    WordUpdate {
        old_word: word.to_vec(),
        new_word,
        count,
        pair_count_deltas,
    }
}

/// Apply a single WordUpdate to the global state (called sequentially).
fn apply_update(
    word_counts: &mut HashMap<Vec<u32>, u64>,
    pair_counts: &mut HashMap<(u32, u32), i64>,
    pair_to_words: &mut HashMap<(u32, u32), HashSet<Vec<u32>>>,
    update: WordUpdate,
) {
    // 1. Apply pair_count deltas
    for (pair, delta) in &update.pair_count_deltas {
        let entry = pair_counts.entry(*pair).or_insert(0);
        *entry += delta;
        if *entry <= 0 {
            pair_counts.remove(pair);
        }
    }

    // 2. Remove old_word from pair_to_words for all its pairs
    let mut empty_pairs = Vec::new();
    for i in 0..update.old_word.len().saturating_sub(1) {
        let pair = (update.old_word[i], update.old_word[i + 1]);
        if let Some(words) = pair_to_words.get_mut(&pair) {
            words.remove(&update.old_word);
            if words.is_empty() {
                empty_pairs.push(pair);
            }
        }
    }
    for pair in empty_pairs {
        pair_to_words.remove(&pair);
    }

    // 3. Add new_word to pair_to_words for all its pairs
    for i in 0..update.new_word.len().saturating_sub(1) {
        let pair = (update.new_word[i], update.new_word[i + 1]);
        pair_to_words
            .entry(pair)
            .or_insert_with(HashSet::new)
            .insert(update.new_word.clone());
    }

    // 4. Update word_counts: remove old, insert new
    word_counts.remove(&update.old_word);
    word_counts.insert(update.new_word, update.count);
}

#[pyclass]
struct BpeTrainer {
    word_counts: HashMap<Vec<u32>, u64>,
    pair_counts: HashMap<(u32, u32), i64>,
    pair_to_words: HashMap<(u32, u32), HashSet<Vec<u32>>>,
    vocab: Vec<Vec<u8>>,
    merges: Vec<(u32, u32)>,
}

#[pymethods]
impl BpeTrainer {
    /// Constructor: initialize vocab (256 bytes + special tokens), convert word_counts
    /// to u32 keys, and build pair_counts + pair_to_words in one pass.
    #[new]
    fn new(word_counts_py: HashMap<Vec<u8>, u64>, special_tokens: Vec<Vec<u8>>) -> PyResult<Self> {
        let t = Instant::now();

        // Initialize vocab: 256 single-byte tokens + special tokens
        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(256 + special_tokens.len());
        for i in 0..256u32 {
            vocab.push(vec![i as u8]);
        }
        for token in &special_tokens {
            vocab.push(token.clone());
        }

        // Convert HashMap<Vec<u8>, u64> -> HashMap<Vec<u32>, u64>
        // Each byte maps to its u32 ID (byte value = token ID for 0-255)
        let mut word_counts: HashMap<Vec<u32>, u64> = HashMap::with_capacity(word_counts_py.len());
        for (word_bytes, count) in word_counts_py {
            let word: Vec<u32> = word_bytes.iter().map(|&b| b as u32).collect();
            word_counts.insert(word, count);
        }

        // Build pair_counts and pair_to_words in one pass over word_counts
        let mut pair_counts: HashMap<(u32, u32), i64> = HashMap::new();
        let mut pair_to_words: HashMap<(u32, u32), HashSet<Vec<u32>>> = HashMap::new();

        for (word, &count) in &word_counts {
            for i in 0..word.len().saturating_sub(1) {
                let pair = (word[i], word[i + 1]);
                *pair_counts.entry(pair).or_insert(0) += count as i64;
                pair_to_words
                    .entry(pair)
                    .or_insert_with(HashSet::new)
                    .insert(word.clone());
            }
        }

        println!(
            "  [Rust BpeTrainer] Init: {:.1}s — {} words, {} pairs, vocab={}",
            t.elapsed().as_secs_f64(),
            word_counts.len(),
            pair_counts.len(),
            vocab.len()
        );

        Ok(BpeTrainer {
            word_counts,
            pair_counts,
            pair_to_words,
            vocab,
            merges: Vec::new(),
        })
    }

    /// Perform one BPE merge step. Returns true if a merge was done, false if no pairs remain.
    fn step(&mut self) -> PyResult<bool> {
        if self.pair_counts.is_empty() {
            return Ok(false);
        }

        // 1. SEQUENTIAL: find best pair (max count, tiebreak: lex-greater vocab bytes wins)
        //    Matches Python: max(pair_counts, key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]])))
        let best_pair = self
            .pair_counts
            .iter()
            .max_by(|(pair_a, count_a), (pair_b, count_b)| {
                count_a
                    .cmp(count_b)
                    .then_with(|| {
                        let (a0, a1) = **pair_a;
                        let (b0, b1) = **pair_b;
                        self.vocab[a0 as usize]
                            .as_slice()
                            .cmp(self.vocab[b0 as usize].as_slice())
                            .then_with(|| {
                                self.vocab[a1 as usize]
                                    .as_slice()
                                    .cmp(self.vocab[b1 as usize].as_slice())
                            })
                    })
            })
            .map(|(pair, _)| *pair)
            .unwrap();

        // 2. SEQUENTIAL: create new vocab entry and record merge
        let new_id = self.vocab.len() as u32;
        let mut new_token = self.vocab[best_pair.0 as usize].clone();
        new_token.extend_from_slice(&self.vocab[best_pair.1 as usize]);
        self.vocab.push(new_token);
        self.merges.push(best_pair);

        // 3. Get affected words and remove from pair_to_words
        let affected: Vec<Vec<u32>> = self
            .pair_to_words
            .remove(&best_pair)
            .unwrap_or_default()
            .into_iter()
            .collect();

        // 4. PARALLEL: compute WordUpdate for each affected word (read-only access to word_counts)
        let word_counts = &self.word_counts;
        let updates: Vec<WordUpdate> = affected
            .par_iter()
            .map(|word| compute_word_update(word, word_counts, best_pair, new_id))
            .collect();

        // 5. SEQUENTIAL: apply all updates to global state
        for update in updates {
            apply_update(
                &mut self.word_counts,
                &mut self.pair_counts,
                &mut self.pair_to_words,
                update,
            );
        }

        Ok(true)
    }

    /// Run step() repeatedly until vocab reaches vocab_size or no pairs remain.
    fn run(&mut self, vocab_size: usize) -> PyResult<()> {
        while self.vocab.len() < vocab_size {
            if !self.step()? {
                break;
            }
        }
        Ok(())
    }

    /// Return (vocab, merges) for Python to consume.
    /// vocab: list[bytes] — one entry per token ID.
    /// merges: list[(bytes, bytes)] — each merge as (left_bytes, right_bytes).
    fn finalize(&self) -> PyResult<(Vec<Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>)> {
        let vocab = self.vocab.clone();
        let merges: Vec<(Vec<u8>, Vec<u8>)> = self
            .merges
            .iter()
            .map(|(a, b)| {
                (
                    self.vocab[*a as usize].clone(),
                    self.vocab[*b as usize].clone(),
                )
            })
            .collect();
        Ok((vocab, merges))
    }

    #[getter]
    fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    #[getter]
    fn merge_count(&self) -> usize {
        self.merges.len()
    }
}

#[pymodule]
fn rust_bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pre_tokenize, m)?)?;
    m.add_class::<BpeTrainer>()?;
    m.add_class::<RustTokenizer>()?;
    m.add_function(wrap_pyfunction!(pre_tokenize_string, m)?)?;
    Ok(())
}