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



        pre_token_list: list[bytes] = rust_bpe.pre_tokenize_string(text, self.special_sorted)
        ids: list[int] = []
        special_set = {s.encode("utf-8") if isinstance(s, str) else s
                         for s in self.special_tokens}

        for pre_token in pre_token_list:
            if pre_token in special_set:
                ids.append(self.byte_to_id[pre_token])
                continue
            parts = [pre_token[i:i+1] for i in range(len(pre_token))]
            

            while len(parts) > 1:
                pairs = [(parts[i], parts[i+1]) for i in range(len(parts)-1)]
                best = min(pairs, key = lambda pair: self.merge_rank.get(pair, float('inf')))
                if best not in self.merge_rank:
                    break

                new_parts, i = [], 0
                while i < len(parts):
                    if i+1<len(parts) and (parts[i], parts[i+1]) == best:
                        new_parts.append(parts[i] + parts[i+1])
                        i += 2
                    else:
                        new_parts.append(parts[i])
                        i += 1
                parts = new_parts
                
            ids.extend(self.byte_to_id[chunk] for chunk in parts)

        return ids
    

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
    