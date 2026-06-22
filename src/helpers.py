import json
import os
import random
import urllib.request
import zipfile

import matplotlib.pyplot as plt


def download_dataset(dataset_dir):
    url = "http://www.manythings.org/anki/fra-eng.zip"
    zip_path = "fra-eng.zip"

    os.makedirs(dataset_dir, exist_ok=True)
    print(f"[INFO] Downloading French-English dataset to {dataset_dir}...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
        while chunk := response.read(8192):
            out_file.write(chunk)

    print("[INFO] Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    txt_file = os.path.join(dataset_dir, "fra.txt")
    pairs = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))

    random.seed(42)
    random.shuffle(pairs)

    val_size = 10000
    train_pairs, val_pairs = pairs[val_size:], pairs[:val_size]

    def save_split(split_pairs, prefix):
        with open(os.path.join(dataset_dir, f"{prefix}.en"), "w", encoding="utf-8") as f_en, open(
            os.path.join(dataset_dir, f"{prefix}.fr"), "w", encoding="utf-8"
        ) as f_fr:
            for en, fr in split_pairs:
                f_en.write(en + "\n")
                f_fr.write(fr + "\n")

    save_split(train_pairs, "train")
    save_split(val_pairs, "val")
    os.remove(zip_path)
    print(f"[SUCCESS] Total pairs: {len(pairs)} | Train: {len(train_pairs)} | Val: {len(val_pairs)}")


def plot_metrics(model_type, artifacts_dir="artifacts"):
    models = ["rnn", "lstm", "gru", "transformer"] if model_type == "all" else [model_type]

    if model_type != "all":
        # Single Model Plotting
        json_path = os.path.join(artifacts_dir, f"{model_type}_history.json")
        if not os.path.exists(json_path):
            print(f"[ERROR] Could not find {json_path}. Run training first.")
            return

        with open(json_path, "r") as f:
            history = json.load(f)

        epochs = range(1, len(history["val"]["loss"]) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Training Metrics ({model_type.upper()})", fontsize=16, fontweight="bold")

        axs[0].plot(epochs, history["train"]["loss"], label="Train", color="tab:blue")
        axs[0].plot(epochs, history["val"]["loss"], label="Val", color="tab:orange", linestyle="--")
        axs[0].set_title("Cross Entropy Loss")

        axs[1].plot(epochs, history["train"]["perplexity"], label="Train", color="tab:blue")
        axs[1].plot(epochs, history["val"]["perplexity"], label="Val", color="tab:orange", linestyle="--")
        axs[1].set_title("Perplexity")

        axs[2].plot(epochs, history["val"]["bleu"], label="Val BLEU", color="tab:green")
        axs[2].set_title("Validation BLEU Score")

        for ax in axs:
            ax.set_xlabel("Epochs")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.7)

        save_path = os.path.join(artifacts_dir, f"{model_type}_metrics.png")

    else:
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Model Architecture Comparison", fontsize=16, fontweight="bold")
        colors = {"rnn": "tab:red", "lstm": "tab:blue", "gru": "tab:orange", "transformer": "tab:purple"}

        for m in models:
            json_path = os.path.join(artifacts_dir, f"{m}_history.json")
            if not os.path.exists(json_path):
                continue

            with open(json_path, "r") as f:
                history = json.load(f)
            epochs = range(1, len(history["val"]["loss"]) + 1)

            axs[0].plot(epochs, history["val"]["loss"], label=m.upper(), color=colors[m], linewidth=2)
            axs[1].plot(epochs, history["val"]["bleu"], label=m.upper(), color=colors[m], linewidth=2)

        axs[0].set_title("Validation Loss")
        axs[1].set_title("Validation BLEU Score")

        for ax in axs:
            ax.set_xlabel("Epochs")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.7)

        save_path = os.path.join(artifacts_dir, "comparative_metrics.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor="white")
    print(f"[SUCCESS] Plot saved to '{save_path}'")
