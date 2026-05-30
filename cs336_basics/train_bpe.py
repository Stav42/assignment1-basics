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
    DEBUG = True    # Set to True to enable verbose per-merge debug output.
    VERBOSE = True    # Set to True to enable [BPE] progress/timing output.
    PROGRESS_INTERVAL = 1  # Print progress every N merges. Lower = more frequent.

    t_total = time.perf_counter()
    if VERBOSE: print(f"[BPE] === Training started ===", flush=True)
    if VERBOSE: print(f"[BPE] input_path={input_path}", flush=True)
    if VERBOSE: print(f"[BPE] vocab_size={vocab_size}  special_tokens={special_tokens}", flush=True)
    t_phase = time.perf_counter()

    # Build initial vocabulary: bytes 0-255, then special tokens
    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    for token in special_tokens:
        vocab.append(token.encode('utf-8'))

    merges: list[tuple] = []
    if VERBOSE: print(f"[BPE] Vocab setup done in {time.perf_counter() - t_phase:.3f}s "
          f"(initial size = {len(vocab)})", flush=True)

    if VERBOSE: print(f"[BPE] Calling Rust pre_tokenize ...", flush=True)
    t_phase = time.perf_counter()
    raw_word_counts = rust_bpe.pre_tokenize(input_path, special_tokens)
    t_rust = time.perf_counter() - t_phase

    # Convert bytes keys to tuple[int, ...] for Python-side processing
    if VERBOSE: print(f"[BPE] Converting word_counts ...", flush=True)
    word_counts = {tuple(b): count for b, count in raw_word_counts.items()}

    # Build pair_counts from word_counts
    if VERBOSE: print(f"[BPE] Building pair_counts ...", flush=True)
    pair_counts = {}
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count

    # Build pair_to_words from word_counts
    if VERBOSE: print(f"[BPE] Building pair_to_words ...", flush=True)
    pair_to_words = {}
    for word in word_counts.keys():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word)

    if VERBOSE: print(f"[BPE] Rust pre_tokenize done in {t_rust:.2f}s", flush=True)
    if VERBOSE: print(f"[BPE]   unique words: {len(word_counts):,}", flush=True)
    if VERBOSE: print(f"[BPE]   initial distinct pairs: {len(pair_counts):,}", flush=True)
    if VERBOSE and pair_counts:
        max_pair = max(pair_counts, key=pair_counts.get)
        print(f"[BPE]   most common pair: {max_pair} -> {pair_counts[max_pair]:,}", flush=True)

    total_merges_target = vocab_size - len(vocab)
    if VERBOSE: print(f"[BPE] Starting merge loop: {total_merges_target:,} merges to perform", flush=True)
    t_loop_start = time.perf_counter()
    merge_count = 0
    rolling_window_t = 0.0  # accumulated time since last progress print

    while len(vocab) < vocab_size:
        t_iter_start = time.perf_counter()

        t0 = time.perf_counter()
        if DEBUG: print("[DEBUG] Finding best pair ...", flush=True)
        best_pair = max(
            pair_counts,
            key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]]))
        )
        if DEBUG: print(f"[DEBUG] Best pair found: {best_pair} with count {pair_counts[best_pair]:,}", flush=True)
        t_find = time.perf_counter() - t0

        t0 = time.perf_counter()
        new_id = len(vocab)
        if DEBUG: print(f"[DEBUG] Merging pair {best_pair} into new token ID {new_id} ...", flush=True)
        vocab.append(vocab[best_pair[0]] + vocab[best_pair[1]])
        merges.append(best_pair)

        ### New implementation
        affected = pair_to_words.get(best_pair, set())
        del pair_to_words[best_pair]

        for word in affected:
            count = word_counts[word]
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)
            
            # Update pair_deltas
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] -= count
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

                # Remove old pair from pair_to_words
                if pair in pair_to_words:
                    pair_to_words[pair].discard(word)
                    if not pair_to_words[pair]:
                        del pair_to_words[pair]

                
                
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count

                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(new_word)

            del word_counts[word]
            word_counts[new_word] = count

        if DEBUG: print(f"[DEBUG] Applying merge to all occurrences of {best_pair} ...", flush=True)

        t_apply = time.perf_counter() - t0
        t_iter = time.perf_counter() - t_iter_start
        merge_count += 1
        rolling_window_t += t_iter

        if VERBOSE and merge_count % PROGRESS_INTERVAL == 0:
            elapsed = time.perf_counter() - t_loop_start
            avg_ms = (rolling_window_t / PROGRESS_INTERVAL) * 1000
            remaining = total_merges_target - merge_count
            eta_s = (elapsed / merge_count) * remaining if merge_count > 0 else 0
            print(
                f"[BPE] merge {merge_count:>5}/{total_merges_target}  "
                f"vocab={len(vocab):>5}  "
                f"best_pair={best_pair}  "
                f"iter={t_iter*1000:6.1f}ms (avg {avg_ms:5.1f}ms)  "
                f"find={t_find*1000:5.1f}ms apply={t_apply*1000:6.1f}ms  "
                f"pairs={len(pair_counts):,}  "
                f"elapsed={elapsed:6.1f}s  ETA={eta_s:6.1f}s",
                flush=True,
            )
            rolling_window_t = 0.0

    t_loop = time.perf_counter() - t_loop_start
    t_grand = time.perf_counter() - t_total
    if VERBOSE: print(f"[BPE] === Training complete ===", flush=True)
    if VERBOSE: print(f"[BPE]   Rust pre_tokenize : {t_rust:8.2f}s", flush=True)
    if VERBOSE: print(f"[BPE]   merge loop        : {t_loop:8.2f}s ({merge_count} merges)", flush=True)
    if VERBOSE: print(f"[BPE]   TOTAL             : {t_grand:8.2f}s", flush=True)
    if VERBOSE: print(f"[BPE]   final vocab size  : {len(vocab):,}", flush=True)

    vocab_dict = {i: vocab[i] for i in range(len(vocab))}
    merges_bytes = [(vocab[a], vocab[b]) for a, b in merges]

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