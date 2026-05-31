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
    # from train_bpe import train_bpe

    tokenizer = Tokenizer.from_files(vocab_path, merge_path, special_tokens=["<|endoftext|>"])

    target = "<|endoftext|>"
    chunk_size = 4096
    print(f"Loaded data from {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        text = ""
        count = 0
        leftover = ""
        while count < 10:
            chunk = f.read(chunk_size)
            if not chunk:  # end of file
                break
            
            # Combine leftover from previous chunk with new chunk
            combined = leftover + chunk
            text += combined
            
            # Count occurrences
            count += combined.count(target)
            
            # Keep the tail that might contain a partial match
            # (the target is 13 chars, so keep last 12 chars as leftover)
            leftover = combined[-(len(target) - 1):] if len(combined) >= len(target) - 1 else combined
        
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

    # === TOKENIZER THROUGHPUT BENCHMARK ===
    print("\n" + "="*60)
    print("TOKENIZER THROUGHPUT BENCHMARK")
    print("="*60)
    
    import time
    
    # Read a larger sample for benchmarking (100 documents)
    print("\nLoading benchmark data...")
    with open(data_path, 'r', encoding='utf-8') as f:
        bench_text = ""
        bench_count = 0
        bench_leftover = ""
        while bench_count < 8000:
            chunk = f.read(chunk_size)
            if not chunk:  # end of file
                break
            
            # Combine leftover from previous chunk with new chunk
            combined = bench_leftover + chunk
            bench_text += combined
            
            # Count occurrences
            bench_count += combined.count(target)
            
            # Keep the tail that might contain a partial match
            bench_leftover = combined[-(len(target) - 1):] if len(combined) >= len(target) - 1 else combined
        
    # Split into a list of documents (not characters)
    benchmark_docs = bench_text.split(target)
    benchmark_text_joined = target.join(benchmark_docs)
    benchmark_bytes = len(benchmark_text_joined.encode('utf-8'))
    
    print(f"Benchmark dataset: {len(benchmark_docs)} documents")
    print(f"Total size: {benchmark_bytes:,} bytes ({benchmark_bytes / (1024*1024):.2f} MB)")
    
    # Warm-up run
    print("\nWarming up...")
    _ = tokenizer.encode(benchmark_docs[0])
    
    # Benchmark run
    print("Running benchmark...")
    start_time = time.perf_counter()
    
    total_tokens = 0
    for doc in benchmark_docs:
        tokens = tokenizer.encode(doc)  # doc is a full story string, not a single char
        total_tokens += len(tokens)
    
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    
    # Calculate throughput
    bytes_per_second = benchmark_bytes / elapsed_seconds
    tokens_per_second = total_tokens / elapsed_seconds
    mb_per_second = bytes_per_second / (1024 * 1024)
    
    print(f"\nResults:")
    print(f"  Total tokens generated: {total_tokens:,}")
    print(f"  Elapsed time: {elapsed_seconds:.3f} seconds")
    print(f"  Throughput: {bytes_per_second:,.0f} bytes/sec")
    print(f"  Throughput: {mb_per_second:.2f} MB/sec")
    print(f"  Throughput: {tokens_per_second:,.0f} tokens/sec")
    
    # Estimate time for Pile dataset (825 GB)
    pile_size_gb = 825
    pile_size_bytes = pile_size_gb * 1024 * 1024 * 1024
    
    estimated_seconds = pile_size_bytes / bytes_per_second
    estimated_minutes = estimated_seconds / 60
    estimated_hours = estimated_minutes / 60
    estimated_days = estimated_hours / 24
    
    print(f"\n{'='*60}")
    print(f"ESTIMATION: Tokenizing the Pile dataset ({pile_size_gb} GB)")
    print(f"{'='*60}")
    print(f"  Estimated time: {estimated_seconds:,.0f} seconds")
    print(f"  Estimated time: {estimated_minutes:,.1f} minutes")
    print(f"  Estimated time: {estimated_hours:.2f} hours")
    print(f"  Estimated time: {estimated_days:.2f} days")
    print(f"{'='*60}\n")
