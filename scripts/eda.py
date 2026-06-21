import os
import random
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def read_text_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError()
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def normalize_for_eda(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-ZÀ-ÿ.!?]+", r" ", s)
    return s


def check_integrity(src_lines, tgt_lines, src_lang="EN", tgt_lang="FR"):
    len_src = len(src_lines)
    len_tgt = len(tgt_lines)
    print(f"Total {src_lang} lines: {len_src}")
    print(f"Total {tgt_lang} lines: {len_tgt}")

    if len_src != len_tgt:
        print("Source and Target datasets have different lengths.")
        return False
    else:
        print("Parallel corpus aligned.")

    empty_src = sum(1 for line in src_lines if not line.strip())
    empty_tgt = sum(1 for line in tgt_lines if not line.strip())

    if empty_src > 0 or empty_tgt > 0:
        print(f"Found empty lines. ({src_lang}: {empty_src}, {tgt_lang}: {empty_tgt})")
    else:
        print("No empty lines detected.")

    return True


def run_eda(src_lines, tgt_lines, src_lang="English", tgt_lang="French"):
    print("\n" + "=" * 50)
    print("Exploraoory Data Analysis (EDA)")
    print("=" * 50)

    src_lengths = [len(normalize_for_eda(line).split()) for line in src_lines]
    tgt_lengths = [len(normalize_for_eda(line).split()) for line in tgt_lines]

    def print_stats(name, lengths):
        print(f"\n--- {name} Sequence Lengths ---")
        print(f"Min length:  {np.min(lengths)} words")
        print(f"Max length:  {np.max(lengths)} words")
        print(f"Mean length: {np.mean(lengths):.2f} words")
        print(f"95th %ile:   {np.percentile(lengths, 95):.0f} words")
        print(f"99th %ile:   {np.percentile(lengths, 99):.0f} words")

    print_stats(src_lang, src_lengths)
    print_stats(tgt_lang, tgt_lengths)

    src_coverage = sum(1 for l in src_lengths if l <= 50) / len(src_lengths) * 100
    tgt_coverage = sum(1 for l in tgt_lengths if l <= 50) / len(tgt_lengths) * 100

    print("\n--- Max Length (50) Coverage ---")
    print(f"Sentences fitting in 50 words ({src_lang}): {src_coverage:.2f}%")
    print(f"Sentences fitting in 50 words ({tgt_lang}): {tgt_coverage:.2f}%")

    src_vocab = Counter()
    tgt_vocab = Counter()
    for s, t in zip(src_lines, tgt_lines):
        src_vocab.update(normalize_for_eda(s).split())
        tgt_vocab.update(normalize_for_eda(t).split())

    print(f"\n--- Estimated Unique Vocabulary ---")
    print(f"{src_lang} unique words: {len(src_vocab)}")
    print(f"{tgt_lang} unique words: {len(tgt_vocab)}")

    plt.figure(figsize=(12, 5))

    plt.hist(
        src_lengths,
        bins=range(0, max(src_lengths) + 5, 2),
        alpha=0.6,
        color="tab:blue",
        label=f"{src_lang} Lengths",
        edgecolor="black",
    )

    plt.hist(
        tgt_lengths,
        bins=range(0, max(tgt_lengths) + 5, 2),
        alpha=0.6,
        color="tab:orange",
        label=f"{tgt_lang} Lengths",
        edgecolor="black",
    )

    plt.axvline(50, color="red", linestyle="dashed", linewidth=2, label="Current max_len (50)")

    plt.title("Distribution of Sequence Lengths in Multi30k")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("docs/figs/multi30k_eda.png", dpi=300)
    print("\n Plot saved successfully as 'multi30k_eda.png'")


def print_random_samples(src_lines, tgt_lines, num_samples=5):
    print("\n" + "=" * 50)
    print("Random Samples")
    print("=" * 50)

    indices = random.sample(range(len(src_lines)), num_samples)
    for i, idx in enumerate(indices, 1):
        print(f"Sample {i} (Index {idx}):")
        print(f"  EN: {src_lines[idx]}")
        print(f"  FR: {tgt_lines[idx]}")
        print("-" * 50)


if __name__ == "__main__":
    src_file = "dataset/multi30k/train.en"
    tgt_file = "dataset/multi30k/train.fr"

    try:
        src_lines = read_text_file(src_file)
        tgt_lines = read_text_file(tgt_file)

        is_valid = check_integrity(src_lines, tgt_lines)

        if is_valid:
            run_eda(src_lines, tgt_lines)
            print_random_samples(src_lines, tgt_lines)

    except Exception as e:
        print(e)
