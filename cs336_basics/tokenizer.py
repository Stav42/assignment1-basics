import pickle
from typing import Iterable, Iterator

import rust_bpe

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[bytes]|None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.special_tokens = special_tokens or []
        # ensure special tokens are in the vocab
        for s in self.special_tokens:
            b = s.encode("utf-8") if isinstance(s, str) else s
            if b not in self.vocab.values():
                self.vocab[len(self.vocab)] = b
        
        self.special_tokens = special_tokens or []
        self.special_sorted = sorted(self.special_tokens, key=len, reverse=True)

        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.byte_to_id = {token: token_id for token_id, token in self.vocab.items()}

        # Precompute once instead of on every encode() call.
        self._special_set = {s.encode("utf-8") if isinstance(s, str) else s
                             for s in self.special_tokens}
        # pretoken bytes -> list[int]; pretokens repeat heavily in real text.
        self._cache: dict[bytes, list[int]] = {}

        # Native Rust tokenizer: runs the whole encode (pre-tokenize + merges)
        # in compiled code and returns token IDs directly. Special tokens are
        # passed as str (Rust splits on them during pre-tokenization).
        special_str = [s.decode("utf-8") if isinstance(s, bytes) else s
                       for s in self.special_tokens]
        self._rust = rust_bpe.RustTokenizer(self.vocab, self.merges, special_str)

        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        with open(vocab_filepath, 'rb') as f:
            vocab_dict = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges_bytes = pickle.load(f)
        return cls(vocab_dict, merges_bytes, special_tokens)


    def encode(self, text: str) -> list[int]:

#   As per the PDF,  
#   1. First we split the text into pretokens - DONE
#   2. Then we look at each pretoken and apply the merges
#   3. For each token: 
#   4.         We find the first applicable merge - What is applicable merge and how to find it
#                   We iterate through the merges in order, and check if the merge's byte pair is present in the current token.
#                   If we find a merge whose byte pair is present in the token, we apply replace all the two bytes occurances in the token with the merged token
#                   Then we break out of the merge loop and start again from the first merge, checking if it can be applied to the new token. 
#   5.              We repeat this process until no more merges can be applied to the token
#   6. Finally, we convert the resulting tokens into their corresponding IDs using the vocab dictionary
#   7. Return the list of token IDs to the adapter



        # Full encode runs natively in Rust (pre-tokenize + merges + IDs).
        return self._rust.encode(text)

    def _encode_pretoken(self, pre_token: bytes) -> list[int]:
        cached = self._cache.get(pre_token)
        if cached is not None:
            return cached

        merge_rank = self.merge_rank
        parts = [pre_token[i:i+1] for i in range(len(pre_token))]

        while len(parts) > 1:
            # Single pass to find the lowest-rank adjacent pair (no list/lambda alloc).
            best_rank = len(merge_rank)  # any real rank is < len(merge_rank)
            best_i = -1
            for i in range(len(parts) - 1):
                r = merge_rank.get((parts[i], parts[i + 1]))
                if r is not None and r < best_rank:
                    best_rank = r
                    best_i = i
            if best_i == -1:
                break

            best = (parts[best_i], parts[best_i + 1])
            new_parts, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == best:
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        byte_to_id = self.byte_to_id
        result = [byte_to_id[chunk] for chunk in parts]
        self._cache[pre_token] = result
        return result
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    
    # def decode(self, ids: list[int]) -> str:

    #     text = ""
    #     for token_id in ids:
    #         token_bytes = self.vocab[token_id]
    #         text += token_bytes.decode("utf-8", errors="replace")
    #     return text

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
    