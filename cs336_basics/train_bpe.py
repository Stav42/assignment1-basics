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
        pair_locations.setdefault(pair, []).append(i)
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
        pair_locations.setdefault(new_left_pair, []).append(left_pos)

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
        pair_locations.setdefault(new_right_pair, []).append(i)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    t0 = time.perf_counter()

    # Build initial vocabulary: bytes 0-255, then special tokens
    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    for token in special_tokens:
        vocab.append(token.encode('utf-8'))

    merges: list[tuple] = []
    num_special = len(special_tokens)

    # print(f"Block A Clock: {time.perf_counter() - t0:.3f}s")
    t2 = time.perf_counter()

    token_ids, pair_counts, special_token_bytes, boundary = rust_bpe.pre_tokenize(input_path, special_tokens)
    pair_locations = _build_pair_locations(token_ids, special_tokens, boundary)

    # print(f"Block B Clock: {time.perf_counter() - t2:.3f}s")
    t3 = time.perf_counter()

    while len(vocab) < vocab_size:
        # t0 = time.perf_counter()
        # print(f"Merge iteration, vocab size: {len(vocab)}")

        # Pick the most frequent pair; break ties lexicographically by token bytes
        best_pair = max(
            pair_counts,
            key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]]))
        )

        # print("Best pair clock: ", time.perf_counter() - t0)
        t1 = time.perf_counter()

        new_id = len(vocab)
        vocab.append(vocab[best_pair[0]] + vocab[best_pair[1]])
        merges.append(best_pair)

        occurrences = pair_locations.pop(best_pair, [])
        del pair_counts[best_pair]

        last_merged_pos = None
        for i in sorted(occurrences):
            # Skip if this occurrence overlaps with a previous merge in this round
            if last_merged_pos is not None and i < last_merged_pos + 2:
                continue
            last_merged_pos = i

            _apply_merge_at(
                i, new_id, best_pair,
                token_ids, boundary,
                pair_counts, pair_locations,
                num_special,
            )

        # print("Occurrences clock: ", time.perf_counter() - t1)
        t2 = time.perf_counter()

        # print("Cleanup clock: ", time.perf_counter() - t2)
        t3 = time.perf_counter()

    print(f"Total clock: {time.perf_counter() - t0:.3f}s")

    vocab_dict = {i: vocab[i] for i in range(len(vocab))}
    merges_bytes = [(vocab[a], vocab[b]) for a, b in merges]

    return vocab_dict, merges_bytes


if __name__ == "__main__":
    train_bpe('tests/fixtures/tiny_debug.txt', 500, ["<|endoftext|>"])