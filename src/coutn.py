import pandas as pd
import pickle

# === CONFIG ===
train_file = "./data/train_orig.csv"
test_file = "./data/test_orig.csv"
train_dedup_file = "./data/train_dedup.csv"
test_dedup_file = "./data/test_dedup.csv"

tokenizer_original_pickle = "./models/sequential_orig.pickle"
tokenizer_dedup_pickle = "./models/sequential_dedup.pickle"

def analyze_file(file_path):
    df = pd.read_csv(file_path)
    word_counts = df['text'].apply(lambda x: len(str(x).split()))

    stats = {
        "file": file_path,
        "utterances": len(df),
        "duplicates": df.duplicated(subset=['text']).sum(),
        "avg_words": word_counts.mean(),
        "label_counts": df['label'].value_counts().to_dict()
    }
    return df, stats


def summarize_combined(train_df, test_df, name):
    combined = pd.concat([train_df, test_df], ignore_index=True)
    word_counts = combined['text'].apply(lambda x: len(str(x).split()))

    stats = {
        "set": name,
        "utterances": len(combined),
        "duplicates": combined.duplicated(subset=['text']).sum(),
        "avg_words": word_counts.mean(),
        "label_counts": combined['label'].value_counts().to_dict()
    }
    return stats


def count_oov_words(test_df, tokenizer_path, name):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    vocab = set(tokenizer.word_index.keys())
    test_words = [word.lower() for text in test_df['text'] for word in text.split()]

    oov_words = [w for w in test_words if w not in vocab]
    oov_count = len(oov_words)
    unique_oov = set(oov_words)

    print(f"\n--- OOV Analysis for {name} ---")
    print(f"Total OOV words in test set: {oov_count}")
    print(f"Unique OOV words: {len(unique_oov)}")
    if unique_oov:
        print(f"Some OOV examples: {list(unique_oov)[:10]}")


if __name__ == "__main__":
    # Load and analyze
    train_df, train_stats = analyze_file(train_file)
    test_df, test_stats = analyze_file(test_file)
    train_dedup_df, train_dedup_stats = analyze_file(train_dedup_file)
    test_dedup_df, test_dedup_stats = analyze_file(test_dedup_file)

    # Print per-file stats
    for stats in [train_stats, test_stats, train_dedup_stats, test_dedup_stats]:
        print(f"\n--- {stats['file']} ---")
        print(f"Utterances: {stats['utterances']}")
        print(f"Duplicates: {stats['duplicates']}")
        print(f"Avg words/utterance: {stats['avg_words']:.2f}")
        print(f"Label distribution: {stats['label_counts']}")

    # Combined stats
    combined_original = summarize_combined(train_df, test_df, "Original Train+Test")
    combined_dedup = summarize_combined(train_dedup_df, test_dedup_df, "Dedup Train+Test")

    print(f"\n=== Combined Statistics ===")
    for stats in [combined_original, combined_dedup]:
        print(f"\n--- {stats['set']} ---")
        print(f"Total utterances: {stats['utterances']}")
        print(f"Total duplicates: {stats['duplicates']}")
        print(f"Average words/utterance: {stats['avg_words']:.2f}")
        print(f"Label distribution: {stats['label_counts']}")

    # OOV analysis
    count_oov_words(test_df, tokenizer_original_pickle, "Original Test")
    count_oov_words(test_dedup_df, tokenizer_dedup_pickle, "Dedup Test")