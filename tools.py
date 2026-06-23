import argparse
import os

from src.helpers import download_dataset, plot_metrics
from src.verify import run_verification


def main():
    parser = argparse.ArgumentParser(description="Machine Translation Utility Tools")

    parser.add_argument("--mode", type=str, required=True, choices=["download", "verify", "plot"])
    parser.add_argument("--model", type=str, default="all", choices=["all", "rnn", "lstm", "gru", "transformer"])
    parser.add_argument("--dataset", type=str, default="dataset/tatoeba")
    parser.add_argument("--tokenizer", type=str, default="basic", choices=["basic", "bpe"])
    parser.add_argument("--run_dir", type=str, default=None, help="Target run folder for plotting")
    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)

    if args.mode == "download":
        download_dataset(args.dataset)

    elif args.mode == "verify":
        args.batch_size = 4
        run_verification(args)

    elif args.mode == "plot":
        if not args.run_dir:
            dirs = [
                os.path.join("artifacts", d)
                for d in os.listdir("artifacts")
                if os.path.isdir(os.path.join("artifacts", d))
            ]
            if not dirs:
                print("[ERROR] No run directories found.")
                return
            args.run_dir = max(dirs, key=os.path.getmtime)

        plot_metrics(args.model, args.run_dir)


if __name__ == "__main__":
    main()
