import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.factory import get_model
from src.tokenizer import BasicTokenizer
from src.trainer import translate_sentence


def get_tokenizers():
    src_tok = BasicTokenizer(min_freq=2)
    tgt_tok = BasicTokenizer(min_freq=2)

    if os.path.exists("../models/src_vocab.json") and os.path.exists("../models/tgt_vocab.json"):
        print("[INFO] Loading tokenizers from JSON...")
        src_tok.load_vocab("src_vocab.json")
        tgt_tok.load_vocab("tgt_vocab.json")
    else:
        print("[ERROR] Tokenizer JSONs not found. Run train.py first.")
        sys.exit(1)

    return src_tok, tgt_tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="transformer", choices=["rnn", "lstm", "gru", "transformer"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device.type.upper()}")

    src_tokenizer, tgt_tokenizer = get_tokenizers()

    model = get_model(args.model, len(src_tokenizer), len(tgt_tokenizer), tgt_tokenizer.PAD, device)

    weights_path = f"../models/best_model_{args.model}.pth"
    if not os.path.exists(weights_path):
        print(f"[ERROR] Could not find '{weights_path}'.")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print("\n" + "=" * 50)
    print(f"Translation CLI ({args.model.upper()})")
    print("Type 'q' to exit.")
    print("=" * 50 + "\n")

    while True:
        sentence = input("English > ")
        if sentence.lower().strip() in ["q", "quit", "exit"]:
            break
        if not sentence.strip():
            continue

        start = time.perf_counter()
        try:
            translation = translate_sentence(sentence, model, src_tokenizer, tgt_tokenizer, device, args.model)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"French  > {translation}")
            print(f"[Latency: {elapsed:.2f} ms]\n")
        except Exception as e:
            print(f"[ERROR] {e}\n")


if __name__ == "__main__":
    main()
