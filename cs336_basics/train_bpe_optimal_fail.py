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


def find_best_pair(pair_counts, vocab):

    # Find pair with max count:
    best_pair = max(
        pair_counts,
        key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]]))
    )
    return best_pair

def find_occurances(token_ids, pair, boundary, best_pair, vocab):
    occurances = []
    for i in range(len(token_ids)-1):
            if boundary[i]:    # Don't merge across pre-token boundary
                continue
            if token_ids[i] == best_pair[0] and token_ids[i+1] == best_pair[1]:
                occurances.append(i)

    for i in occurances:
        boundary[i] = boundary[i+1]
        token_ids[i] = len(vocab)-1
        token_ids[i+1] = None

    return occurances, boundary, token_ids

def find_occurances_optimal(token_ids, boundary, best_pair, vocab, pair_occurances):

    occurances = pair_occurances.get(best_pair, [])

    for i in occurances:
        next_element = find_nonNone_right(token_ids, i+1)
        if next_element is not None:
            boundary[i] = boundary[next_element]
            token_ids[next_element] = None
        token_ids[i] = len(vocab)-1
    
    return boundary, token_ids, pair_occurances




def compact_token_ids(token_ids, boundary):
    new_token_ids = []
    new_boundary = []
    for t, b in zip(token_ids, boundary):
        if t is not None:
            new_token_ids.append(t)
            new_boundary.append(b)
    return new_token_ids, new_boundary

def compact_token_ids_optimal(token_ids, boundary):
    new_token_ids = []
    new_boundary = []
    for t, b in zip(token_ids, boundary):
        if t is not None:
            new_token_ids.append(t)
            new_boundary.append(b)
    return new_token_ids, new_boundary

def recompute_pair_counts(token_ids, boundary, special_tokens):
    pair_counts = {}
    for i in range(len(token_ids)-1):
        if token_ids[i] is None:
            continue
        if boundary[i]:
            continue

        rightElement = find_nonNone_right(token_ids, i+1)
        if rightElement is None:
            continue
        pair = (token_ids[i], token_ids[rightElement])
        if (pair[0] >= 256 and pair[0] < 256 + len(special_tokens)) or \
        (pair[1] >= 256 and pair[1] < 256 + len(special_tokens)):
            continue
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts

def recompute_pair_counts_optimal(token_ids, boundary, special_tokens):
    return NotImplementedError

# def update_pair_occurances(pair_occurances, best_pair, token_ids, vocab, occurances):
#     # Update pair_occurances for the new pairs created by the merge
#     # Look at the neighbours of each occurance, and find the pairs in the 
#     # pair_occurances dict, and update them accordingly. Since we 
#     # were already taking care of boundaries while constructing the pair_occurances dict, we don't need to worry about boundaries here.
#     # So, look at the left and right neighbours of each occurance, and update the pair_occurances dict accordingly.
#     consumed_positions = {k+1 for k in occurances}  # positions that will be consumed (set to None)
#     occurrences_set = set(occurances)  # for quick lookup

#     for i in occurances:
#         left_element = None
#         for j in range(i-1, -1, -1):
#             if token_ids[j] is not None and j not in consumed_positions:
#                 left_element = j
#                 break
        
#         # Find right neighbor, skipping consumed positions  
#         right_element = None
#         for j in range(i+2, len(token_ids)):
#             if token_ids[j] is not None and j not in consumed_positions:
#                 right_element = j
#                 break

#         if left_element is not None and left_element not in occurrences_set:
#             left_pair = (token_ids[left_element], best_pair[0])
#             if left_pair and left_pair in pair_occurances:
#                 pair_occurances[left_pair] = [idx for idx in pair_occurances[left_pair] if idx != left_element]
#                 new_left_pair = (token_ids[left_element], len(vocab)-1)
#                 pair_occurances[new_left_pair] = pair_occurances.get(new_left_pair, []) + [left_element]
        
#             if left_pair and left_pair not in pair_occurances:
#                 new_left_pair = (token_ids[left_element], len(vocab)-1)
#                 pair_occurances[new_left_pair] = [left_element]

#         if right_element is not None:
#             right_pair = (best_pair[1], token_ids[right_element])
#             if right_pair and right_pair in pair_occurances:
#                 pair_occurances[right_pair] = [idx for idx in pair_occurances[right_pair] if idx != i+1]
#                 new_right_pair = (len(vocab)-1, token_ids[right_element])
#                 pair_occurances[new_right_pair] = pair_occurances.get(new_right_pair, []) + [i]
#             if right_pair and right_pair not in pair_occurances:
#                 new_right_pair = (len(vocab)-1, token_ids[right_element])
#                 pair_occurances[new_right_pair] = [i] 

            



#     # best_pair_occurances = pair_occurances.get(best_pair, [])
#     # for i in best_pair_occurances:
#     #     left_element = find_nonNone_left(token_ids, i-1)
#     #     right_element = find_nonNone_right(token_ids, i+2)
#     #     left_pair = (token_ids[left_element], best_pair[0]) if left_element is not None else None
#     #     right_pair = (best_pair[1], token_ids[right_element]) if right_element is not None else None

#     #     if left_pair and left_pair in pair_occurances:
#     #         pair_occurances[left_pair] = [idx for idx in pair_occurances[left_pair] if idx != left_element]
#     #         new_left_pair = (token_ids[left_element], len(vocab)-1)
#     #         pair_occurances[new_left_pair] = pair_occurances.get(new_left_pair, []) + [left_element]

#     #     if right_pair and right_pair in pair_occurances:
#     #         pair_occurances[right_pair] = [idx for idx in pair_occurances[right_pair] if idx != i+1]
#     #         new_right_pair = (len(vocab)-1, token_ids[right_element])
#     #         pair_occurances[new_right_pair] = pair_occurances.get(new_right_pair, []) + [i]

#     #     # what if the new left pair or the new right pair is not in the pair_occurances dict? We can just add it with the current index as the only occurance.
#     #     if left_pair and left_pair not in pair_occurances:
#     #         new_left_pair = (token_ids[left_element], len(vocab)-1)
#     #         pair_occurances[new_left_pair] = [left_element]

#     #     if right_pair and right_pair not in pair_occurances:
#     #         new_right_pair = (len(vocab)-1, token_ids[right_element])
#     #         pair_occurances[new_right_pair] = [i] 

#     return pair_occurances


def update_pair_occurances(pair_occurances, best_pair, token_ids, vocab, occurances):
    consumed_positions = {k+1 for k in occurances}
    occurrences_set = set(occurances)
    new_id = len(vocab) - 1
    for i in occurances:
        left_element = None
        for j in range(i-1, -1, -1):
            if token_ids[j] is not None and j not in consumed_positions:
                left_element = j
                break
        right_element = None
        for j in range(i+2, len(token_ids)):
            if token_ids[j] is not None and j not in consumed_positions:
                right_element = j
                break
        if left_element is not None:
            left_is_occurrence = left_element in occurrences_set
            left_token = new_id if left_is_occurrence else token_ids[left_element]
            old_left_pair = (token_ids[left_element], best_pair[0])
            new_left_pair = (left_token, new_id)
            if old_left_pair in pair_occurances:
                pair_occurances[old_left_pair] = [idx for idx in pair_occurances[old_left_pair] if idx != left_element]
                pair_occurances[new_left_pair] = pair_occurances.get(new_left_pair, []) + [left_element]
            else:
                pair_occurances[new_left_pair] = pair_occurances.get(new_left_pair, []) + [left_element]
        if right_element is not None:
            right_is_occurrence = right_element in occurrences_set
            right_token = new_id if right_is_occurrence else token_ids[right_element]
            old_right_pair = (best_pair[1], token_ids[right_element])
            new_right_pair = (new_id, right_token)
            if old_right_pair in pair_occurances:
                pair_occurances[old_right_pair] = [idx for idx in pair_occurances[old_right_pair] if idx != i+1]
                pair_occurances[new_right_pair] = pair_occurances.get(new_right_pair, []) + [i]
            else:
                pair_occurances[new_right_pair] = pair_occurances.get(new_right_pair, []) + [i]
    return pair_occurances

def find_nonNone_right(token_ids, i):
    for j in range(i, len(token_ids)):
        if token_ids[j] is not None:
            return j
    return None
    
def find_nonNone_left(token_ids, i):
    for j in range(i, -1, -1):
        if token_ids[j] is not None:
            return j
    return None


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):

    # Let's load the file. After each story there is an      token, so we use that to split 
    # and also introduce that as a special token in the vocab.
    # counts = rust_bpe.count_byte_pairs(input_path, special_tokens)

    # Now we will write the merge operation and build up the vocab
    
    # Base vocab setup
    vocab = []
    for i in range(256):
        vocab.append(bytes([i]))

    # vocab.append('<|endoftext|>'.encode('utf-8'))

    for token in special_tokens:
        vocab.append(token.encode('utf-8'))

    merges = []
    # Now we look at the count from the rust code and make changes to the vocab
    token_ids, pair_counts, special_token_bytes, boundary = rust_bpe.pre_tokenize(input_path, special_tokens)
    pair_occurances = None

    if pair_occurances is None:
        # Make a dict with keys as pairs and values as list of occurances
        pair_occurances = {}
        for i in range(len(token_ids)-1):
            if boundary[i]:    # Don't merge across pre-token boundary
                continue
            pair = (token_ids[i], token_ids[i+1])
            if pair not in pair_occurances:
                pair_occurances[pair] = []
            pair_occurances[pair].append(i)

    while(len(vocab) < vocab_size):

        best_pair = find_best_pair(pair_counts, vocab)

        # Add this pair to the merge and the vocab
        vocab.append(vocab[best_pair[0]] + vocab[best_pair[1]])
        merges.append(best_pair)

        # Update pair_occurances for the best pair
        occurrences = pair_occurances.get(best_pair, [])
        pair_occurances = update_pair_occurances(pair_occurances, best_pair, token_ids, vocab, occurrences)
        # pair_occurances = update_pair_occurances(pair_occurances, best_pair, token_ids, vocab)

        boundary, token_ids, pair_occurances = find_occurances_optimal(token_ids, boundary, best_pair, vocab, pair_occurances)


        # token_ids_new, boundary_new = compact_token_ids_optimal(token_ids, boundary)
        # Clean up the None entries from token_ids
        # token_ids = [t for t in token_ids if t is not None]
        # token_ids = token_ids_new
        # boundary = boundary_new

        pair_counts = recompute_pair_counts(token_ids, boundary, special_tokens)


    vocab_dict = {i: vocab[i] for i in range(len(vocab))}
    merges_bytes = [(vocab[mrg[0]], vocab[mrg[1]]) for mrg in merges]

    return vocab_dict, merges_bytes

if __name__ == "__main__":
    train_bpe('tests/fixtures/tiny_debug.txt', 500, ["<|endoftext|>"])