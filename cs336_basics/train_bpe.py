# BPE tokenizer trainer
#
# Strategy:
#   Rust handles pre-tokenization + word counting (fast, parallel).
#   Rust returns a dict[bytes, int] mapping each pre-token's bytes to its count.
#   Python builds pair_counts and pair_to_words from word_counts, then runs the merge loop.
#
# Pre-tokenization has TWO levels:
#   1. Split file at <|endoftext|> boundaries (Rust, parallelizable)
#   2. Within each chunk, split using GPT-2 PAT regex (Rust)
#      This ensures byte pairs are only counted WITHIN "words",
#      preventing BPE merges from crossing word/document boundaries.
#
# Special tokens (like <|endoftext|>) are:
#   - Added to vocabulary upfront before any merges
#   - Never split or merged by the BPE algorithm
#   - In the final vocab, they have IDs AFTER the 256 base byte tokens
#
# Characters like space, colon, apostrophe are NOT special tokens —
# they're regular characters split by the PAT regex into their own pre-tokens.
#
# TODO list:
# 1. Import the compiled Rust module (e.g., `import rust_bpe`)
# 2. Define the main function:
#    train_bpe(input_path, vocab_size, special_tokens) -> (vocab, merges)
# 3. Call Rust to pre-tokenize and get word counts:
#    word_counts = rust_bpe.pre_tokenize(input_path, special_tokens)
#    - word_counts: dict[bytes, int] — mapping each pre-token's bytes to its count
# 4. Initialize vocabulary:
#    - Tokens 0..255: each byte value is its own token
#      (e.g., vocab[97] = b'a', vocab[32] = b' ')
#    - Tokens 256+: special tokens, assigned sequentially
#      (e.g., vocab[256] = b'<|endoftext|>')
#    - So initial vocab size = 256 + len(special_tokens)
# 5. The merge loop (run until len(vocab) reaches vocab_size):
#    a. Find the pair (id_a, id_b) with the highest count in pair_counts
#       Skip if no pairs remain or max count is 0
#    b. Create the new merged token bytes: vocab[id_a] + vocab[id_b]
#    c. Assign the new token the next available ID: len(vocab)
#    d. Add the new token to vocab
#    e. Add (id_a, id_b) to the merges list
#    f. Update word_counts, pair_counts, and pair_to_words for affected words:
#       - For each word containing the merged pair, create a new word with the pair replaced
#       - Decrement old pair counts and remove old pair_to_words entries
#       - Increment new pair counts and add new pair_to_words entries
#       - Update word_counts to reflect the new word
# 6. Return (vocab, merges) to the adapter


import rust_bpe
import time


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    VERBOSE = True
    PROGRESS_INTERVAL = 1  # Print progress every N merges

    t_total = time.perf_counter()
    if VERBOSE: print(f"[BPE] === Training started ===", flush=True)
    if VERBOSE: print(f"[BPE] input_path={input_path}", flush=True)
    if VERBOSE: print(f"[BPE] vocab_size={vocab_size}  special_tokens={special_tokens}", flush=True)

    # Step 1: Rust pre-tokenization (parallel, memory-mapped)
    if VERBOSE: print(f"[BPE] Calling Rust pre_tokenize ...", flush=True)
    t_phase = time.perf_counter()
    raw_word_counts = rust_bpe.pre_tokenize(input_path, special_tokens)
    t_rust = time.perf_counter() - t_phase
    if VERBOSE: print(f"[BPE] Rust pre_tokenize done in {t_rust:.2f}s", flush=True)
    if VERBOSE: print(f"[BPE]   unique words: {len(raw_word_counts):,}", flush=True)

    # Step 2: Initialize Rust BpeTrainer (builds vocab, pair_counts, pair_to_words)
    num_special = len(special_tokens)
    special_tokens_bytes = [t.encode('utf-8') for t in special_tokens]
    if VERBOSE: print(f"[BPE] Initializing Rust BpeTrainer ...", flush=True)
    t_phase = time.perf_counter()
    trainer = rust_bpe.BpeTrainer(raw_word_counts, special_tokens_bytes)
    t_init = time.perf_counter() - t_phase
    if VERBOSE: print(f"[BPE] BpeTrainer init done in {t_init:.2f}s (vocab={trainer.vocab_len})", flush=True)

    # Step 3: Merge loop with progress reporting
    total_merges_target = vocab_size - trainer.vocab_len
    if VERBOSE: print(f"[BPE] Starting merge loop: {total_merges_target:,} merges to perform", flush=True)
    t_loop_start = time.perf_counter()

    while trainer.vocab_len < vocab_size:
        trainer.step()
        if VERBOSE and trainer.merge_count % PROGRESS_INTERVAL == 0:
            elapsed = time.perf_counter() - t_loop_start
            remaining = total_merges_target - trainer.merge_count
            eta_s = (elapsed / trainer.merge_count) * remaining if trainer.merge_count > 0 else 0
            print(
                f"[BPE] merge {trainer.merge_count:>5}/{total_merges_target}  "
                f"vocab={trainer.vocab_len:>5}  "
                f"elapsed={elapsed:6.1f}s  ETA={eta_s:6.1f}s",
                flush=True,
            )

    t_loop = time.perf_counter() - t_loop_start

    # Step 4: Finalize — get vocab and merges from Rust
    vocab_list, merges_bytes = trainer.finalize()
    vocab_dict = {i: b for i, b in enumerate(vocab_list)}

    t_grand = time.perf_counter() - t_total
    if VERBOSE: print(f"[BPE] === Training complete ===", flush=True)
    if VERBOSE: print(f"[BPE]   Rust pre_tokenize : {t_rust:8.2f}s", flush=True)
    if VERBOSE: print(f"[BPE]   BpeTrainer init   : {t_init:8.2f}s", flush=True)
    if VERBOSE: print(f"[BPE]   merge loop        : {t_loop:8.2f}s ({trainer.merge_count} merges)", flush=True)
    if VERBOSE: print(f"[BPE]   TOTAL             : {t_grand:8.2f}s", flush=True)
    if VERBOSE: print(f"[BPE]   final vocab size  : {len(vocab_dict):,}", flush=True)

    return vocab_dict, merges_bytes


if __name__ == "__main__":
    import tracemalloc
    tracemalloc.start()
    print("Training BPE tokenizer...")
    t0 = time.perf_counter()
    vocab_dict, merges_bytes = train_bpe('/Users/stav.42/courses/assignment1-basics/TinyStories-train.txt', 10000, ["<|endoftext|>"])
    print(f"Total training time: {time.perf_counter() - t0:.3f}s")
    # Track max memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB; Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    tracemalloc.stop()
    #Get longest string in vocab
    longest_string = max(vocab_dict.values(), key=len)
    print(f"Longest string in vocab: {longest_string.decode('utf-8')}")

    # Save in pickle files
    import pickle
    with open('vocab_dict.pkl', 'wb') as f:
        pickle.dump(vocab_dict, f)
    with open('merges_bytes.pkl', 'wb') as f:
        pickle.dump(merges_bytes, f)

    print("Vocabulary and merges saved to vocab_dict.pkl and merges_bytes.pkl")