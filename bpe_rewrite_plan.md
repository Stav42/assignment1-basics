# BPE Tokenizer: Dict-of-Pre-Tokens Rewrite Plan

This document describes the architectural rewrite that moves BPE training from
the flat `token_ids` representation to the Sennrich-style dict-of-pre-tokens
representation. The goal is to bring full TinyStories training from many hours
down to ~90 seconds.

---

## Why the current architecture is too slow

The current code represents the corpus as a flat list of ~2 billion byte IDs.
Every operation in the merge loop scales with this length:

- The first merge (e.g. `(h, e)`) iterates over ~50M+ occurrences sequentially.
- Each occurrence involves ~30 µs of Python bytecode (dict ops, set ops, scans).
- That's ~15 minutes for **one** merge.

The fundamental issue is that the work per merge scales with **total corpus
size** rather than with **unique affected pre-tokens**. For TinyStories, the
ratio is ~1000–10000×.

---

## The three data structures

### `word_counts: dict[tuple[int, ...], int]`

The corpus, deduplicated. Each key is a tuple of byte IDs representing one
unique pre-token. The value is how many times that pre-token appears in the
corpus.

Example (from the PDF's `bpe_example`):

```
word_counts = {
  (l, o, w):           5,    # "low" appears 5 times
  (l, o, w, e, r):     2,    # "lower" appears 2 times
  (w, i, d, e, s, t):  3,    # "widest" appears 3 times
  (n, e, w, e, s, t):  6,    # "newest" appears 6 times
}
```

For TinyStories: ~50K–100K unique words vs. hundreds of millions of total
pre-tokens. ~1000× compression.

### `pair_counts: dict[(int, int), int]`

For each pair `(a, b)`, the total number of times that pair appears in the
corpus, **weighted by word frequency**.

For the example: pair `(l, o)` appears once in `(l,o,w)` and once in
`(l,o,w,e,r)`, so `pair_counts[(l, o)] = 5*1 + 2*1 = 7`.

Built initially by iterating `word_counts` once and, for each word, adding
`count` (not 1) to each adjacent pair.

### `pair_to_words: dict[(int, int), set[tuple[int, ...]]]`

The reverse index. For each pair, the set of unique words that contain it.

```
pair_to_words[(l, o)] = {(l, o, w), (l, o, w, e, r)}
pair_to_words[(s, t)] = {(w, i, d, e, s, t), (n, e, w, e, s, t)}
```

This is the structural magic. When `(l, o)` wins the merge, you don't scan the
whole corpus — you look up `pair_to_words[(l, o)]`, get exactly the affected
words, and only touch those.

---

## Rust side: return `word_counts` directly

Pre-tokenization stays parallel via rayon, but the output changes.

### What Rust does

1. Read (or memory-map) the file.
2. Split at `<|endoftext|>` into chunks (current behavior — parallel-safe).
3. For each chunk, in parallel:
   - Apply the PAT regex to get pre-tokens.
   - For each pre-token, encode as `Vec<u32>` of byte IDs.
   - Increment a **local** `HashMap<Vec<u32>, u32>`: increment the entry for
     that pre-token by 1.
4. Merge per-chunk HashMaps via rayon's `fold + reduce` pattern (already used
   for pair_counts in the current code).
5. Special tokens are **skipped** — they don't contribute to `word_counts` at
   all. They get fixed vocab IDs in Python (256, 257, ...).
6. Return `HashMap<Vec<u32>, u32>` to Python. PyO3 converts to a Python `dict`
   automatically.

### What goes away

- No flat `token_ids` list.
- No `boundary` array — boundary information is implicit in the word grouping.
- No initial `pair_counts` computation in Rust (Python does it fast from
  `word_counts`).

### Why this is also a Rust win

- FFI output: ~50K–100K entries instead of ~2 billion. Order-of-magnitude
  smaller PyO3 conversion cost.
- The per-chunk HashMap insertion is the inner loop — Rust handles it at
  native speed.
- No giant `Vec<usize>` allocation for billions of ints.

---

## Python side: initial setup after Rust returns

Three steps:

### Step 1: Build vocab as before

- 256 byte tokens + special tokens. Unchanged.

### Step 2: Build `pair_counts` from `word_counts`

- Iterate `word_counts.items()`. For each `(word, count)`:
  - Walk adjacent pairs in `word`.
  - For each pair, `pair_counts[pair] += count` (using `dict.get(pair, 0)`).
- A single pass over ~50K–100K word tuples. Fast.

### Step 3: Build `pair_to_words` from `word_counts`

- Same loop. For each pair in each word, add `word` to
  `pair_to_words[pair]` (using `setdefault(pair, set()).add(word)`).

Total setup: <1 second.

---

## The merge loop body — the heart of it

### Step A: Pick the winning pair

- `best_pair = max(pair_counts, key=...)` with your existing tiebreak.
  Unchanged.
- `new_id = len(vocab)`. Append merged bytes to `vocab`. Append `best_pair` to
  `merges`. Unchanged.

### Step B: Get affected words

- `affected = pair_to_words.pop(best_pair, set())` — the set of words
  containing `(a, b)`.
- `del pair_counts[best_pair]`.

### Step C: For each affected word, rewrite it

For each `word` in `affected`:

1. `count = word_counts[word]`.
2. **Build `new_word`** by scanning `word` left-to-right and replacing every
   `(a, b)` adjacency with the single token `new_id`. Handle overlapping like
   `(a, a, a, a)` greedily (see "Overlapping subtlety" below).
3. **Compute pair deltas**:
   - For every adjacent pair `p` in the original `word`:
     `pair_counts[p] -= count`. Delete the entry if it goes to zero.
   - For every adjacent pair `p_new` in `new_word`:
     `pair_counts[p_new] += count`.
4. **Update `pair_to_words`**:
   - For every pair `p` in the original `word`, remove `word` from
     `pair_to_words[p]`. If the set becomes empty, delete the entry.
   - For every pair `p_new` in `new_word`, add `new_word` to
     `pair_to_words[p_new]`.
5. **Update `word_counts`**:
   - `del word_counts[word]`
   - `word_counts[new_word] = count`

### Step D: Implicit cleanup

The per-word updates already handle cleanup. No global compaction needed.

---

## Worked example: one merge

Starting state:

```
word_counts:
  (l, o, w):           5
  (l, o, w, e, r):     2
  (w, i, d, e, s, t):  3
  (n, e, w, e, s, t):  6

pair_counts (selected):
  (l, o): 7, (o, w): 7, (w, e): 8, (e, r): 2, (w, i): 3, (i, d): 3,
  (d, e): 3, (e, s): 9, (s, t): 9, (n, e): 6, (e, w): 6

pair_to_words (selected):
  (l, o): {(l,o,w), (l,o,w,e,r)}
  (s, t): {(w,i,d,e,s,t), (n,e,w,e,s,t)}
  (e, s): {(w,i,d,e,s,t), (n,e,w,e,s,t)}
```

**Pick best_pair**: `(s, t)` wins (count 9; lex-greater of the tied pairs).
new_id = `Z_st`.

**Affected**: `{(w,i,d,e,s,t), (n,e,w,e,s,t)}` — just 2 words.

**Process word `(w,i,d,e,s,t)`, count=3**:

- new_word = `(w, i, d, e, Z_st)`.
- Old pairs in word: `(w,i), (i,d), (d,e), (e,s), (s,t)`. Decrement each by 3.
- New pairs in new_word: `(w,i), (i,d), (d,e), (e,Z_st)`. Increment each by 3.
- Net effect: `(w,i), (i,d), (d,e)` cancel out; `(e,s) -= 3`, `(s,t) -= 3`,
  `(e,Z_st) += 3`.
- pair_to_words: remove `(w,i,d,e,s,t)` from `(e,s)`'s and `(s,t)`'s sets; add
  `(w,i,d,e,Z_st)` to `(e,Z_st)`'s set; the unchanged pairs need their
  membership swapped too (`(w,i,d,e,s,t)` → `(w,i,d,e,Z_st)`).
- word_counts: del `(w,i,d,e,s,t)`; set `word_counts[(w,i,d,e,Z_st)] = 3`.

**Process word `(n,e,w,e,s,t)`, count=6**: similar.
new_word = `(n, e, w, e, Z_st)`.
Net: `(e,s) -= 6`, `(s,t) -= 6`, `(e,Z_st) += 6`.

**Final**:

- `pair_counts[(s,t)]` was 9, now 0 — delete.
- `pair_counts[(e,s)]` was 9, now 0 — delete.
- `pair_counts[(e,Z_st)] = 9`.
- `pair_to_words[(e,Z_st)] = {(w,i,d,e,Z_st), (n,e,w,e,Z_st)}`.

**Total Python work for this merge: ~30 dict/set operations.** Not 564,848.

---

## The overlapping subtlety inside a word

When scanning a word like `(a, a, a, a)` for the pair `(a, a)`:

- Greedy left-to-right gives `(Z, Z)` — two non-overlapping merges. **Correct.**
- Replacing every `(a, a)` position naively would give `(Z, Z, Z)`. **Wrong.**

Handle this with a single left-to-right scan: build the new word by walking
position by position. When you see the pair, emit `new_id` and **skip the next
position**. This is the same overlap logic the current code has, just scoped to
one tuple of length ~5 instead of a flat list of 2 billion.

---

## What gets deleted from the current code

- `_apply_merge_at` — entire function gone.
- `_build_pair_locations` — gone.
- Flat `token_ids` and `boundary` arrays — gone.
- Leftward/rightward None-skipping scans — gone (no Nones; tuples are dense).
- Outer-loop overlap check + `sorted(occurrences)` + `last_merged_pos` — gone
  (overlap is now local within one word).

## What stays

- `vocab`, `merges`, `num_special`, special-token IDs.
- The `max(pair_counts, key=...)` selection with the lex-greater tiebreak.
- The main `while len(vocab) < vocab_size:` loop.

---

## Performance expectations on full TinyStories

| Phase                                    | Current             | Rewrite           |
|------------------------------------------|---------------------|-------------------|
| Rust pre_tokenize + word_count           | ~30 min             | ~15 s             |
| Python setup (pair_counts, pair_to_words)| ~50 min             | ~1 s              |
| Merge loop                               | many hours          | ~60–90 s          |
| **Total**                                | **many hours**      | **~90 seconds**   |

The first few merges remain the most expensive (most affected words). Later
merges affect very few words — many touch only 1–2 unique pre-tokens. That's
where the dramatic speedup compounds.

---

## Implementation order

1. **Python-only first.** Keep current Rust output. In Python, immediately
   after Rust returns, convert flat `token_ids + boundary` into `word_counts`
   by walking the corpus once and grouping consecutive non-boundary runs into
   tuples (counting duplicates). This isolates the data-structure change so
   you can debug correctness against the test suite before touching Rust.
2. **Verify on validation set.** Time it. Expected ~30–60 seconds.
3. **Move to Rust output.** Rust returns `word_counts` directly; the Python
   conversion step in (1) goes away. Pure performance change, no correctness
   risk.
4. **Run on full TinyStories.** Expected ~90 seconds.

The "Python-only first" step is the critical de-risking move. Tests pass first,
then performance changes.

---

## Special token handling

In the dict-of-pre-tokens model, special-token segments **never become words**.
They are not subject to merges.

- In Rust, when processing a chunk and encountering a special-token segment
  (e.g. `<|endoftext|>`), skip it entirely — do not insert anything into
  `word_counts` for it.
- In Python, special tokens get fixed vocab IDs (`256, 257, ...,
  256 + num_special - 1`) the same way they do now.
- The merge logic never sees special tokens, never indexes them, never
  includes them in any pair count.

This naturally enforces the rule "special tokens delimit hard segmentation
boundaries" because they're entirely absent from the working data structures.

---

## Correctness invariants to preserve (carried over from current code)

- Tiebreak on `max(pair_counts, ...)`: lex-greater pair wins. Use
  `(pair_counts[p], (vocab[p[0]], vocab[p[1]]))` as the key.
- Merges list ordered by creation: append `best_pair` (as `(int, int)` IDs) to
  `merges` per iteration. Convert to bytes pairs only at function return.
- `vocab[id_a] + vocab[id_b]` is the bytes for the merged token. The merged
  token gets ID `len(vocab)` immediately after vocab append.
- Within each word, overlapping pairs are resolved greedily left-to-right.

If your current code passes `pytest tests/test_train_bpe.py`, the rewrite
needs to preserve these. The tests are the authoritative spec.

---

## Why this is the architecture the PDF expects

PDF page 7 (`bpe_example`) explicitly shows the dict-of-tuples representation:

> It is convenient to represent this as a `dict[tuple[bytes, ...], int]`, e.g.,
> `{(l,o,w): 5, ...}`.

PDF page 8 ("Optimizing the merging step") notes that incrementally updating
counts gives "significant speedups." Combined with the dict-of-pre-tokens
representation, this gives the order-of-magnitude reduction that hits the
2-minute target.

PDF page 10 ("Hint" for `train_bpe_tinystories`) tells you to use
multiprocessing during pre-tokenization plus the fact that
`<|endoftext|>` delimits documents and is handled as a special case before
merges. The Rust + rayon side already covers the multiprocessing; this
rewrite implements the "special case before merges" by structurally
excluding special tokens from `word_counts`.
