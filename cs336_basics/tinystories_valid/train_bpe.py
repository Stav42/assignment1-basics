# BPE tokenizer trainer
#
# Strategy:
#   Rust handles pre-tokenization + initial pair counting (fast, parallel).
#   Rust returns the full pre-tokenized text as a flat list of token IDs.
#   Python runs the merge loop, incrementally updating the token list in place.
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
# 3. Call Rust to pre-tokenize and get initial data:
#    token_ids, pair_counts, special_token_bytes = rust_bpe.pre_tokenize(input_path, special_tokens)
#    - token_ids: list[int] — the full corpus as a flat list of byte IDs (0-255)
#    - pair_counts: dict {(int, int): int} — initial adjacent pair frequencies
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
#    f. Walk through token_ids and INCREMENTALLY update:
#       - Find every occurrence of the sequence [id_a, id_b] in token_ids
#       - For each occurrence at position i:
#         i.   Get the left neighbor (token_ids[i-1]) and right neighbor (token_ids[i+2])
#         ii.  Decrement counts: (left, id_a) and (id_b, right)
#         iii. Increment counts: (left, new_id) and (new_id, right)
#         iv.  Replace [id_a, id_b] at positions i,i+1 with [new_id, None]
#              (mark i+1 for removal, i takes the new_id)
#    g. Clean up the None entries from token_ids (compact the list)
#    h. Remove (id_a, id_b) from pair_counts entirely
# 6. Return (vocab, merges) to the adapter


import rust_bpe
import time


def _is_special(token_id: int, num_special: int) -> bool:
    """Return True if token_id is a special token (IDs 256 .. 256+num_special-1)."""
    return 256 <= token_id < 256 + num_special


def _build_pair_locations(token_ids: list, special_tokens: list, boundary: list) -> dict:
    """
    Build a mapping from each adjacent pair to the list of positions (anchored at
    the left element) where that pair appears in token_ids.

    Pairs that cross a word/document boundary or involve a special token are excluded.
    """
    num_special = len(special_tokens)
    pair_locations: dict = {}
    for i in range(len(token_ids) - 1):
        if boundary[i]:
            continue
        if _is_special(token_ids[i], num_special) or _is_special(token_ids[i + 1], num_special):
            continue
        pair = (token_ids[i], token_ids[i + 1])
        pair_locations.setdefault(pair, set([])).add(i)
    return pair_locations


def _apply_merge_at(
    i: int,
    new_id: int,
    best_pair: tuple,
    token_ids: list,
    boundary: list,
    pair_counts: dict,
    pair_locations: dict,
    num_special: int,
):
    """
    Apply a single merge occurrence at position i:
      - Find the actual positions of best_pair[1] (i2) and the right neighbor.
      - Null out i2, write new_id at i, update boundary.
      - Decrement/remove stale pairs (left, best_pair[0]) and (best_pair[1], right).
      - Increment/add new pairs (left, new_id) and (new_id, right).
    """
    # --- Scan left: find the nearest non-None token before i ---
    left_val = None
    left_pos = None
    for pos in range(i - 1, -1, -1):
        if token_ids[pos] is not None:
            left_val = token_ids[pos]
            left_pos = pos
            break

    # --- Scan right (1st hit): position of best_pair[1] ---
    i2 = None
    for pos in range(i + 1, len(token_ids)):
        if token_ids[pos] is not None:
            i2 = pos
            break

    # --- Scan right (2nd hit): the right neighbor ---
    right_val = None
    right_pos = None
    if i2 is not None:
        for pos in range(i2 + 1, len(token_ids)):
            if token_ids[pos] is not None:
                right_val = token_ids[pos]
                right_pos = pos
                break

    # --- Apply the merge in-place ---
    boundary[i] = boundary[i2]
    token_ids[i] = new_id
    token_ids[i2] = None

    # --- Update left-side pairs ---
    if left_val is not None and not _is_special(left_val, num_special) and not boundary[left_pos]:
        old_left_pair = (left_val, best_pair[0])
        if old_left_pair in pair_counts:
            pair_counts[old_left_pair] -= 1
            if pair_counts[old_left_pair] == 0:
                del pair_counts[old_left_pair]
            pair_locations[old_left_pair].remove(left_pos)
            if not pair_locations[old_left_pair]:
                del pair_locations[old_left_pair]

        new_left_pair = (left_val, new_id)
        pair_counts[new_left_pair] = pair_counts.get(new_left_pair, 0) + 1
        pair_locations.setdefault(new_left_pair, set([])).add(left_pos)

    # --- Update right-side pairs ---
    if right_val is not None and not _is_special(right_val, num_special) and not boundary[i]:
        old_right_pair = (best_pair[1], right_val)
        if old_right_pair in pair_counts:
            pair_counts[old_right_pair] -= 1
            if pair_counts[old_right_pair] == 0:
                del pair_counts[old_right_pair]
            pair_locations[old_right_pair].remove(i2)
            if not pair_locations[old_right_pair]:
                del pair_locations[old_right_pair]

        new_right_pair = (new_id, right_val)
        pair_counts[new_right_pair] = pair_counts.get(new_right_pair, 0) + 1
        pair_locations.setdefault(new_right_pair, set([])).add(i)


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
    num_special = len(special_tokens)
    if VERBOSE: print(f"[BPE] Vocab setup done in {time.perf_counter() - t_phase:.3f}s "
          f"(initial size = {len(vocab)})", flush=True)

    if VERBOSE: print(f"[BPE] Calling Rust pre_tokenize ...", flush=True)
    t_phase = time.perf_counter()
    token_ids, pair_counts, special_token_bytes, boundary = rust_bpe.pre_tokenize(input_path, special_tokens)
    t_rust = time.perf_counter() - t_phase

    # Construct the word_counts array
    if VERBOSE: print(f"[BPE] Constructing word_counts ...", flush=True)
    word_counts = {}
    start_indx = 0
    end_indx = 0
    for index, token_id in enumerate(token_ids):
        end_indx = index
        if boundary[index]:
            if end_indx > start_indx:
                word = tuple(token_ids[start_indx:end_indx+1])
                if word_counts.get(word) is None:
                    word_counts[word] = 0
                word_counts[word] += 1
            start_indx = index + 1
            continue

    # Construct the pair_counts dict from the word_counts
    pair_counts_from_words = {}
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts_from_words[pair] = pair_counts_from_words.get(pair, 0) + count
    
    assert pair_counts == pair_counts_from_words, "Mismatch between Rust pair_counts and Python word_counts-derived pair_counts"
    if VERBOSE: print(f"[BPE] word_counts and pair_counts consistency check passed ({len(word_counts):,} words)", flush=True)
    # Construct the pair_to_words dict
    pair_to_words = {}
    for word in word_counts.keys():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word)

    


    if VERBOSE: print(f"[BPE] Rust pre_tokenize done in {t_rust:.2f}s", flush=True)
    if VERBOSE: print(f"[BPE]   token_ids length: {len(token_ids):,}", flush=True)
    if VERBOSE: print(f"[BPE]   initial distinct pairs: {len(pair_counts):,}", flush=True)
    if VERBOSE and pair_counts:
        max_pair = max(pair_counts, key=pair_counts.get)
        print(f"[BPE]   most common pair: {max_pair} -> {pair_counts[max_pair]:,}", flush=True)

    if VERBOSE: print(f"[BPE] Building pair_locations index ...", flush=True)
    t_phase = time.perf_counter()
    # pair_locations = _build_pair_locations(token_ids, special_tokens, boundary)
    t_index = time.perf_counter() - t_phase
    # if VERBOSE: print(f"[BPE] pair_locations built in {t_index:.2f}s "
    #       f"({len(pair_locations):,} keys)", flush=True)

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
        # occurrences = pair_locations.pop(best_pair, [])
        # del pair_counts[best_pair]

        # last_merged_pos = None
        # for i in sorted(occurrences):
        #     if last_merged_pos is not None and i < last_merged_pos + 2:
        #         continue
        #     last_merged_pos = i
        #     _apply_merge_at(
        #         i, new_id, best_pair,
        #         token_ids, boundary,
        #         pair_counts, pair_locations,
        #         num_special,
        #     )
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
    if VERBOSE: print(f"[BPE]   build pair index  : {t_index:8.2f}s", flush=True)
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
    vocab_dict, merges_bytes = train_bpe('/Users/stav.42/courses/assignment1-basics/TinyStories-valid.txt', 10000, ["<|endoftext|>"])
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