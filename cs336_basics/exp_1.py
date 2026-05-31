

MERGE_PATH_OWT = "/Users/stav.42/courses/assignment1-basics/cs336_basics/owt_train/merges_bytes.pkl"
VOCAB_PATH_OWT = "/Users/stav.42/courses/assignment1-basics/cs336_basics/owt_train/vocab_dict.pkl"

MERGE_PATH_TINY = "/Users/stav.42/courses/assignment1-basics/cs336_basics/tinystories_train/merges_bytes.pkl"
VOCAB_PATH_TINY = "/Users/stav.42/courses/assignment1-basics/cs336_basics/tinystories_train/vocab_dict.pkl"

OWT_DATA_PATH = "/Users/stav.42/courses/assignment1-basics/TinyStories-train.txt"
TINY_DATA_PATH = "/Users/stav.42/courses/assignment1-basics/TinyStories-train.txt"

if __name__ == "__main__":

    vocab_choice = "Tiny"   # OWT || Tiny
    data_choice = "Tiny"  # OWT || Tiny

    if vocab_choice == "OWT":
        vocab_path = VOCAB_PATH_OWT
        merge_path = MERGE_PATH_OWT
    else:
        vocab_path = VOCAB_PATH_TINY
        merge_path = MERGE_PATH_TINY

    if data_choice == "OWT":
        data_path = OWT_DATA_PATH
    else:
        data_path = TINY_DATA_PATH

    from cs336_basics.tokenizer import Tokenizer
    from cs336_basics.train_bpe import train_bpe

    tokenizer = Tokenizer.from_files(vocab_path, merge_path, special_tokens=["<|endoftext|>"])

    count = 0
    target = "<|endoftext|>"
    chunk_size = 4096
    leftover = ""
    print(f"Loaded data from {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        text = ""
        while count < 10:
            chunk = f.read(chunk_size)
            if not chunk:  # end of file
                break
            
            # Combine leftover from previous chunk with new chunk
            text = leftover + chunk
            
            # Count occurrences
            count += text.count(target)
            
            # Keep the tail that might contain a partial match
            # (the target is 13 chars, so keep last 12 chars as leftover)
            leftover = text[-(len(target) - 1):] if len(text) >= len(target) - 1 else text
        
        # text = f.read()
        print(f"Data length: {len(text)} characters, {len(text.encode('utf-8'))} bytes")
        # Sample 10 documents from here
        documents = text.split("<|endoftext|>")[:10]
        print(f"Sampled {len(documents)} documents for analysis")
        # Get tokenizer compression ratio for all documents 
        token_ids_total = 0
        byte_length_total = 0
        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1}:")
            token_ids = tokenizer.encode(doc)
            token_ids_total += len(token_ids)
            byte_length_total += len(doc.encode('utf-8'))
            print(f" Encoding result: {len(token_ids)} tokens")
            compression_ratio = len(token_ids) / len(doc.encode('utf-8'))
            print(f"Document {i+1}: Original bytes={len(doc.encode('utf-8'))}, Tokens={len(token_ids)}, Compression ratio={1/compression_ratio:.3f}")

        print(f"\nTotal tokens: {token_ids_total}")
        print(f"Total bytes: {byte_length_total}")
        print(f"Overall compression ratio: {1/(token_ids_total/byte_length_total):.3f}")
