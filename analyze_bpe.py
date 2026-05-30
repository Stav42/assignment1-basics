import pickle
from collections import Counter

def load_data():
    with open('cs336_basics/tinystories_train/vocab_dict.pkl', 'rb') as f:
        vocab_dict = pickle.load(f)
    with open('cs336_basics/tinystories_train/merges_bytes.pkl', 'rb') as f:
        merges_bytes = pickle.load(f)
    return vocab_dict, merges_bytes

def analyze_vocab_size(vocab_dict):
    print(f"Total vocabulary size: {len(vocab_dict)}")
    print(f"Base bytes: 256")
    print(f"Merged tokens: {len(vocab_dict) - 256}")

def analyze_token_lengths(vocab_dict):
    lengths = [len(token) for token in vocab_dict.values()]
    length_counts = Counter(lengths)
    
    print("\nToken length distribution:")
    for length in sorted(length_counts.keys()):
        count = length_counts[length]
        pct = count / len(vocab_dict) * 100
        print(f"  {length} bytes: {count} tokens ({pct:.1f}%)")
    
    print(f"\nAverage token length: {sum(lengths) / len(lengths):.2f} bytes")
    print(f"Max token length: {max(lengths)} bytes")

def analyze_longest_tokens(vocab_dict, top_n=20):
    tokens_with_lengths = [(token, len(token)) for token in vocab_dict.values()]
    tokens_with_lengths.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} longest tokens:")
    for token, length in tokens_with_lengths[:top_n]:
        try:
            decoded = token.decode('utf-8')
            print(f"  {length:3d} bytes: {repr(decoded)}")
        except:
            print(f"  {length:3d} bytes: {token} (invalid UTF-8)")

def analyze_early_merges(merges_bytes, top_n=30):
    print(f"\nFirst {top_n} merges (most frequent byte pairs):")
    for i, (a, b) in enumerate(merges_bytes[:top_n]):
        try:
            a_str = a.decode('utf-8')
            b_str = b.decode('utf-8')
            merged = (a + b).decode('utf-8')
            print(f"  {i+1:3d}. '{a_str}' + '{b_str}' -> '{merged}'")
        except:
            print(f"  {i+1:3d}. {a} + {b} (invalid UTF-8)")

def analyze_merge_patterns(merges_bytes):
    merge_lengths = []
    for a, b in merges_bytes:
        merge_lengths.append(len(a) + len(b))
    
    length_counts = Counter(merge_lengths)
    print("\nMerge result length distribution:")
    for length in sorted(length_counts.keys()):
        count = length_counts[length]
        pct = count / len(merges_bytes) * 100
        print(f"  {length} bytes: {count} merges ({pct:.1f}%)")

def analyze_common_subwords(vocab_dict, min_length=3, top_n=30):
    multi_byte_tokens = []
    for token_id, token in vocab_dict.items():
        if len(token) >= min_length:
            try:
                decoded = token.decode('utf-8')
                if decoded.isprintable() and not decoded.startswith('\\'):
                    multi_byte_tokens.append((decoded, len(token)))
            except:
                pass
    
    multi_byte_tokens.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} longest readable subwords (>= {min_length} bytes):")
    for text, length in multi_byte_tokens[:top_n]:
        print(f"  {length:3d} bytes: '{text}'")

def main():
    print("Loading BPE data...")
    vocab_dict, merges_bytes = load_data()
    
    print("=" * 60)
    print("VOCABULARY SIZE ANALYSIS")
    print("=" * 60)
    analyze_vocab_size(vocab_dict)
    
    print("\n" + "=" * 60)
    print("TOKEN LENGTH DISTRIBUTION")
    print("=" * 60)
    analyze_token_lengths(vocab_dict)
    
    print("\n" + "=" * 60)
    print("LONGEST TOKENS")
    print("=" * 60)
    analyze_longest_tokens(vocab_dict)
    
    print("\n" + "=" * 60)
    print("EARLY MERGES (MOST FREQUENT PAIRS)")
    print("=" * 60)
    analyze_early_merges(merges_bytes)
    
    print("\n" + "=" * 60)
    print("MERGE PATTERNS")
    print("=" * 60)
    analyze_merge_patterns(merges_bytes)
    
    print("\n" + "=" * 60)
    print("COMMON SUBWORDS")
    print("=" * 60)
    analyze_common_subwords(vocab_dict)

if __name__ == "__main__":
    main()
