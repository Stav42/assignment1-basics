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


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):

    # Let's load the file. After each story there is an <|endoftext|> token, so we use that to split 
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
    token_ids, pair_counts, special_token_bytes = rust_bpe.pre_tokenize(input_path, special_tokens)
    while(len(vocab) < vocab_size):

        # Find pair with max count:
        best_pair = max(
            pair_counts,
            key=lambda p: (pair_counts[p], vocab[p[0]] + vocab[p[1]])
        )

        # Add this pair to the merge and the vocab
        vocab.append(vocab[best_pair[0]] + vocab[best_pair[1]])
        merges.append(best_pair)

        occurances = []
        # Find all occurances of best pair in the token_ids pre-tokenized list
        for i in range(len(token_ids)-1):

            if token_ids[i] == best_pair[0] and token_ids[i+1] == best_pair[1]:
                occurances.append(i)

        for i in occurances:
            token_ids[i] = len(vocab)-1
            token_ids[i+1] = None
        
        # Clean up the None entries from token_ids
        token_ids = [t for t in token_ids if t is not None]

        # Go through the token_ids and recompute the pair_count array
        pair_counts = {}
        for i in range(len(token_ids)-1):
            pair = (token_ids[i], token_ids[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1


        

        # Step 1: get the pair with max count
        # Step 2: add that as a merged element in the vocab
        # Step 3: remove that pair_count from the pair_count variable
        # Step 4: Decrease the count of the other pair elements by this amount. The other
        # pairs to remove it from must have the first of the pair as its end element, 
        # and the second of hte pair as teh first element. 
        # Step 5: Introduce a new element for each pair that we decrement at step 4. Each of 
        # these new elements must have the merged element as either prefix or suffix
        # and count of the merged element must be added.
        # Step 6: Go back to Step 1, and repeat till len(vocab)>vocab_size


    # vocab.append(bytes(i).decode('utf-8') for i in range(256))
    print(vocab)

    vocab_dict = {i: vocab[i] for i in range(len(vocab))}
    merges_bytes = [(vocab[mrg[0]], vocab[mrg[1]]) for mrg in merges]

    return vocab_dict, merges_bytes

if __name__ == "__main__":
    train_bpe('hello', 123, ['asdas'])