import json
import os

import torch

from src.factory import build_model
from src.tokenizer import BasicTokenizer, BPETokenizer, SPMTokenizer


@torch.no_grad()
def run_inference(args, _dummy_src_tok, _dummy_tgt_tok, device, run_dir):
    hp_path = os.path.join(run_dir, "hyperparameters.json")
    if not os.path.exists(hp_path):
        print(f"[ERROR] {hp_path} missing. Cannot infer automatically.")
        return

    with open(hp_path, "r") as f:
        hp = json.load(f)

    tokenizer_type = hp.get("tokenizer", "basic")
    vocab_size = hp.get("vocab_size", 8000)
    model_type = hp.get("model", args.model)

    # 2. Reconstruct the correct tokenizers
    if tokenizer_type == "basic":
        src_tok, tgt_tok = BasicTokenizer(), BasicTokenizer()
    elif tokenizer_type == "bpe":
        src_tok, tgt_tok = BPETokenizer(vocab_size=vocab_size), BPETokenizer(vocab_size=vocab_size)
    elif tokenizer_type == "spm":
        src_tok, tgt_tok = SPMTokenizer(vocab_size=vocab_size), SPMTokenizer(vocab_size=vocab_size)

    try:
        src_tok.load_vocab(os.path.join(run_dir, f"src_vocab_{tokenizer_type}.json"))
        tgt_tok.load_vocab(os.path.join(run_dir, f"tgt_vocab_{tokenizer_type}.json"))
    except FileNotFoundError:
        print(f"[ERROR] Vocab files missing in {run_dir}. Please run training first.")
        return

    model = build_model(model_type, len(src_tok), len(tgt_tok), tgt_tok.PAD, device, **hp)
    weights_path = os.path.join(run_dir, f"best_{model_type}.pth")

    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights not found at {weights_path}")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print(f"\n[INFO] Loaded Run: {run_dir}")
    print(f"[INFO] Settings: Model={model_type.upper()} | Tokenizer={tokenizer_type.upper()} | Vocab={vocab_size}")
    print(f"[INFO] Interactive Mode | Type 'q' to quit.")
    print("=" * 60)

    while True:
        sentence = input("English > ")
        if sentence.lower().strip() in ["q", "quit", "exit"]:
            break
        if not sentence.strip():
            continue

        tokens = [src_tok.SOS] + src_tok.encode(sentence) + [src_tok.EOS]
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
        generated = torch.tensor([[tgt_tok.SOS]], dtype=torch.long, device=device)

        for _ in range(50):
            if model_type == "transformer":
                out = model(src_tensor, generated)
            else:
                dummy_tgt = torch.zeros(1, 50, dtype=torch.long, device=device)
                dummy_tgt[0, : generated.size(1)] = generated
                out = model(src_tensor, dummy_tgt, teacher_forcing_ratio=0.0)[:, : generated.size(1)]

            pred_token = out.argmax(2)[:, -1].item()
            generated = torch.cat([generated, torch.tensor([[pred_token]], device=device)], dim=1)

            if pred_token == tgt_tok.EOS:
                break

        translation = tgt_tok.decode(generated[0])
        print(f"French  > {translation}\n")
