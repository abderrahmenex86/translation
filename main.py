import argparse
import datetime
import os
import warnings

import torch

warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage.*")

from src.dataset import build_dataloaders, load_data_lines
from src.infer import run_inference
from src.optimize import run_optimization
from src.tester import run_test
from src.tokenizer import BasicTokenizer
from src.trainer import run_training


def get_run_dir(args, is_training=False):
    if not is_training and args.run_dir:
        return args.run_dir

    if is_training:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.model == "transformer":
            suffix = f"transformer_d{args.d_model}_h{args.nhead}_e{args.num_enc}_d{args.num_dec}"
        else:
            suffix = f"{args.model}_emb{args.embed_size}_hid{args.hidden_size}_lyr{args.num_layers}"

        run_dir = os.path.join("artifacts", f"{now}_{suffix}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    else:
        if not os.path.exists("artifacts"):
            raise ValueError("Artifacts directory does not exist. Train a model first.")
        dirs = [
            os.path.join("artifacts", d) for d in os.listdir("artifacts") if os.path.isdir(os.path.join("artifacts", d))
        ]
        if not dirs:
            raise ValueError("No previous runs found. Please specify --run_dir or train a model.")
        return max(dirs, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Machine Translation Master Entrypoint")

    parser.add_argument("--mode", type=str, required=True, choices=["test", "train", "optimize", "infer"])
    parser.add_argument("--model", type=str, required=True, choices=["rnn", "lstm", "gru", "transformer"])
    parser.add_argument("--dataset", type=str, default="dataset/tatoeba")
    parser.add_argument("--tokenizer", type=str, default="basic", choices=["basic"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint")
    parser.add_argument("--run_dir", type=str, default=None, help="Specific artifacts subfolder to load/resume")

    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_enc", type=int, default=3)
    parser.add_argument("--num_dec", type=int, default=3)
    parser.add_argument("--dim_ff", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    if args.tokenizer == "basic":
        src_tok, tgt_tok = BasicTokenizer(), BasicTokenizer()
    else:
        raise NotImplementedError("Other tokenizers are not yet integrated.")

    is_training = args.mode in ["train", "optimize"]
    if args.resume:
        if not args.run_dir:
            dirs = [
                os.path.join("artifacts", d)
                for d in os.listdir("artifacts")
                if os.path.isdir(os.path.join("artifacts", d))
            ]
            if not dirs:
                raise ValueError("No prior runs found to resume from.")
            run_dir = max(dirs, key=os.path.getmtime)
        else:
            run_dir = args.run_dir
    else:
        run_dir = get_run_dir(args, is_training)

    if args.mode == "infer":
        run_inference(args, src_tok, tgt_tok, device, run_dir)
        return

    print(f"[INFO] Preparing {args.dataset}...")
    (train_src, train_tgt), (val_src, val_tgt) = load_data_lines(args.dataset)

    if args.mode == "optimize":
        train_src, train_tgt = train_src[:10000], train_tgt[:10000]
        val_src, val_tgt = val_src[:1000], val_tgt[:1000]
    elif args.mode == "test":
        train_src, train_tgt = train_src[:200], train_tgt[:200]
        val_src, val_tgt = val_src[:50], val_tgt[:50]
        args.batch_size = min(args.batch_size, 32)

    if args.resume:
        print(f"[INFO] Resuming training. Re-loading vocab from {run_dir}...")
        try:
            src_tok.load_vocab(os.path.join(run_dir, f"src_vocab_{args.tokenizer}.json"))
            tgt_tok.load_vocab(os.path.join(run_dir, f"tgt_vocab_{args.tokenizer}.json"))
        except FileNotFoundError:
            print(f"[ERROR] Cannot resume. Vocabulary files are missing in {run_dir}")
            return
    else:
        src_tok.fit(train_src, max_vocab=15000)
        tgt_tok.fit(train_tgt, max_vocab=15000)
        if is_training:
            src_tok.save_vocab(os.path.join(run_dir, f"src_vocab_{args.tokenizer}.json"))
            tgt_tok.save_vocab(os.path.join(run_dir, f"tgt_vocab_{args.tokenizer}.json"))

    train_loader, val_loader = build_dataloaders(args, (train_src, train_tgt), (val_src, val_tgt), src_tok, tgt_tok)

    if args.mode == "train":
        run_training(args, train_loader, val_loader, src_tok, tgt_tok, device, run_dir)
    elif args.mode == "optimize":
        run_optimization(args, train_loader, val_loader, src_tok, tgt_tok, device, run_dir)
    elif args.mode == "test":
        run_test(args, train_loader, src_tok, tgt_tok, device)


if __name__ == "__main__":
    main()
