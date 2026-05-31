MERGE_PATH_OWT = "/Users/stav.42/courses/assignment1-basics/cs336_basics/owt_train/merges_bytes.pkl"
VOCAB_PATH_OWT = "/Users/stav.42/courses/assignment1-basics/cs336_basics/owt_train/vocab_dict.pkl"

MERGE_PATH_TINY = "/Users/stav.42/courses/assignment1-basics/cs336_basics/tinystories_train/merges_bytes.pkl"
VOCAB_PATH_TINY = "/Users/stav.42/courses/assignment1-basics/cs336_basics/tinystories_train/vocab_dict.pkl"

OWT_DATA_PATH = "/Users/stav.42/courses/assignment1-basics/TinyStories-train.txt"
TINY_DATA_PATH = "/Users/stav.42/courses/assignment1-basics/TinyStories-train.txt"

import os
import time

import numpy as np


def file_chunk_generator(file_path, chunk_size=4096, max_bytes=None):
    """Yield large text blobs, each ending on a <|endoftext|> boundary.

    chunk_size controls how much raw text is read per step; using a LARGE
    chunk_size (multiple MB) yields blobs that contain many documents, which is
    what gives the parallel Rust `encode_batch` enough work to spread across
    cores. max_bytes (chars) optionally caps total output so before/after
    profiling runs stay fast and identical.
    """
    produced = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        leftover = ""
        while True:
            if max_bytes is not None and produced >= max_bytes:
                if leftover:
                    yield leftover
                break

            chunk = f.read(chunk_size)
            if not chunk:  # end of file
                if leftover:
                    yield leftover  # yield any remaining text as the last document
                break
            combined = leftover + chunk

            ## See where the last occurrence of the target token is in the combined text.
            # If it exists, split there: yield the part up to and including it as a
            # blob, and keep the remainder as leftover for the next chunk.
            target = "<|endoftext|>"
            last_occurrence = combined.rfind(target)
            if last_occurrence != -1:
                text = combined[:last_occurrence + len(target)]
                leftover = combined[last_occurrence + len(target):]
                produced += len(text)
                yield text
            else:
                # Target not found yet — keep accumulating into leftover.
                leftover = combined


def encode_to_array(tokenizer, blobs, use_batch=True):
    """Encode an iterable of large text blobs into one uint16 token-id array.

    Uses the parallel Rust `encode_batch` when available (and use_batch=True):
    each blob is split into documents and encoded across cores. Otherwise falls
    back to the serial per-blob `encode`. Builds the array per blob and
    concatenates at the end — this avoids the per-token Python/C boundary
    crossing that `np.fromiter` over a token-at-a-time generator incurs.

    use_batch=False forces the serial path even when `encode_batch` exists, so a
    serial baseline can be measured without reverting the build (EXP1_FORCE_SERIAL).
    """
    rust = tokenizer._rust
    batch = getattr(rust, "encode_batch", None) if use_batch else None
    parts = []
    for blob in blobs:
        if not blob:
            continue
        ids = batch(blob) if batch is not None else rust.encode(blob)
        if ids:
            parts.append(np.asarray(ids, dtype=np.uint16))
    if not parts:
        return np.empty(0, dtype=np.uint16)
    return np.concatenate(parts)


def main():
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

    # Benchmark run (serial, per-document — a stable control across before/after)
    print("Running benchmark (serial per-document encode)...")
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

    # --- Parallel batch throughput (only meaningful once encode_batch exists) ---
    if hasattr(tokenizer._rust, "encode_batch"):
        print("\nRunning benchmark (parallel encode_batch on the whole blob)...")
        blob = target.join(benchmark_docs)
        _ = tokenizer._rust.encode_batch(blob[:len(blob)//100] or blob)  # warm rayon pool
        start_b = time.perf_counter()
        batch_ids = tokenizer._rust.encode_batch(blob)
        batch_elapsed = time.perf_counter() - start_b
        batch_mb_s = benchmark_bytes / batch_elapsed / (1024 * 1024)
        print(f"  Tokens: {len(batch_ids):,}  (serial produced {total_tokens:,})")
        print(f"  Elapsed time: {batch_elapsed:.3f} seconds")
        print(f"  Throughput: {batch_mb_s:.2f} MB/sec")
        if elapsed_seconds > 0:
            print(f"  Speedup vs serial: {elapsed_seconds / batch_elapsed:.2f}x")

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

    # === FULL-DATASET ENCODE + SAVE ===
    # This is the section we profile before/after the parallelization change.
    # EXP1_MAX_BYTES caps how much of the file we process so before/after runs
    # are quick and identical; unset it to encode the whole file.
    print("Time to encode the entire dataset and save token IDs to a .npy file...")
    max_bytes_env = os.environ.get("EXP1_MAX_BYTES")
    max_bytes = int(max_bytes_env) if max_bytes_env else None
    blob_chunk_size = 8 * 1024 * 1024  # 8 MB blobs → many documents per Rust call
    force_serial = bool(os.environ.get("EXP1_FORCE_SERIAL"))
    parallel = hasattr(tokenizer._rust, "encode_batch") and not force_serial
    if max_bytes is not None:
        print(f"  [capped] Encoding only the first {max_bytes:,} chars (EXP1_MAX_BYTES)")
    if force_serial:
        print("  [EXP1_FORCE_SERIAL] forcing serial encode path")
    print(f"  Encode path: {'parallel encode_batch' if parallel else 'serial encode'}")
    print(f"  Blob size: {blob_chunk_size:,} bytes/read")

    t0 = time.perf_counter()
    generator = file_chunk_generator(data_path, chunk_size=blob_chunk_size, max_bytes=max_bytes)
    arr = encode_to_array(tokenizer, generator, use_batch=not force_serial)
    encode_seconds = time.perf_counter() - t0
    print(f"Encoding completed in {encode_seconds:.3f} seconds ({len(arr):,} tokens)")
    if encode_seconds > 0:
        print(f"  Encode throughput: {len(arr) / encode_seconds:,.0f} tokens/sec")

    arr_head = arr[:100]
    print(f"First 100 token IDs: {arr_head}")
    decoded_head = tokenizer.decode(arr_head.tolist())
    print(f"Decoded first 100 token IDs: {decoded_head}")

    t1 = time.perf_counter()
    np.save("token_ids.npy", arr)
    save_seconds = time.perf_counter() - t1
    print(f"Saved {arr.nbytes:,} bytes ({len(arr):,} tokens) in {save_seconds:.3f} seconds")

    print(
        f"\n[SUMMARY] path={'parallel' if parallel else 'serial'}  "
        f"encode={encode_seconds:.3f}s  save={save_seconds:.3f}s  "
        f"tokens={len(arr):,}"
    )


if __name__ == "__main__":
    # Optional function-level profiling of the WHOLE run. Enable with
    # EXP1_CPROFILE=1; label the output file with PROFILE_TAG (e.g. before/after).
    #   EXP1_CPROFILE=1 PROFILE_TAG=before EXP1_MAX_BYTES=200000000 .venv/bin/python -m cs336_basics.exp_1
    # Then compare: python -m pstats profile_exp1_before.prof
    if os.environ.get("EXP1_CPROFILE"):
        import cProfile
        import io
        import pstats

        tag = os.environ.get("PROFILE_TAG", "run")
        prof = cProfile.Profile()
        prof.enable()
        try:
            main()
        finally:
            prof.disable()
            out_path = f"profile_exp1_{tag}.prof"
            prof.dump_stats(out_path)
            stream = io.StringIO()
            stats = pstats.Stats(prof, stream=stream).sort_stats("cumulative")
            stats.print_stats(25)
            print("\n" + "=" * 60)
            print(f"cPROFILE TOP 25 (cumulative) — tag={tag}")
            print("=" * 60)
            print(stream.getvalue())
            print(f"[cProfile] wrote {out_path}  (inspect: python -m pstats {out_path})")
    else:
        main()
