import argparse
import os
import random
import sys

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import TranslationDataset, collate_fn, filter_pairs, read_text_file
from src.models.factory import get_model
from src.tokenizer import BasicTokenizer
from src.trainer import evaluate, train_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["all", "rnn", "lstm", "gru", "transformer"])
    args = parser.parse_args()

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.backends.cudnn.benchmark = True

    n_epochs = 50
    batch_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading datasets...")
    raw_src_train = read_text_file("dataset/tatoeba/train.en")
    raw_tgt_train = read_text_file("dataset/tatoeba/train.fr")
    raw_src_val = read_text_file("dataset/tatoeba/val.en")
    raw_tgt_val = read_text_file("dataset/tatoeba/val.fr")

    raw_src_train, raw_tgt_train = filter_pairs(raw_src_train, raw_tgt_train, max_len=50)
    raw_src_val, raw_tgt_val = filter_pairs(raw_src_val, raw_tgt_val, max_len=50)

    print("[INFO] Fitting tokenizers...")
    src_tokenizer = BasicTokenizer(min_freq=2)
    tgt_tokenizer = BasicTokenizer(min_freq=2)
    src_tokenizer.fit(raw_src_train, max_vocab=15000)
    tgt_tokenizer.fit(raw_tgt_train, max_vocab=15000)

    src_tokenizer.save_vocab("src_vocab.json")
    tgt_tokenizer.save_vocab("tgt_vocab.json")

    train_dataset = TranslationDataset(raw_src_train, raw_tgt_train, src_tokenizer, tgt_tokenizer)
    val_dataset = TranslationDataset(raw_src_val, raw_tgt_val, src_tokenizer, tgt_tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    models_to_train = ["transformer", "gru", "lstm", "rnn"] if args.model == "all" else [args.model]

    mlflow.set_experiment("Translation_Comparative_Study")

    for current_model in models_to_train:
        print(f"\n[INFO] Training Model: {current_model.upper()}")

        model = get_model(current_model, len(src_tokenizer), len(tgt_tokenizer), tgt_tokenizer.PAD, device)
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.PAD, label_smoothing=0.1)

        lr = 5e-4 if current_model == "transformer" else 5e-3
        weight_decay = 1e-4 if current_model == "transformer" else 1e-2

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        best_val_loss = float("inf")
        model_save_path = f"best_model_{current_model}.pth"

        with mlflow.start_run(run_name=f"{current_model}_tatoeba"):
            mlflow.log_params(
                {
                    "architecture": current_model,
                    "dataset": "tatoeba",
                    "epochs": n_epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "label_smoothing": 0.1,
                    "src_vocab_size": len(src_tokenizer),
                    "tgt_vocab_size": len(tgt_tokenizer),
                }
            )

            mlflow.log_artifact("src_vocab.json", artifact_path="tokenizers")
            mlflow.log_artifact("tgt_vocab.json", artifact_path="tokenizers")

            for epoch in range(1, n_epochs + 1):
                train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, current_model)
                val_metrics = evaluate(model, val_loader, criterion, device, current_model, tgt_tokenizer)
                scheduler.step(val_metrics["loss"])

                current_lr = optimizer.param_groups[0]["lr"]

                print(
                    f"Epoch {epoch:02d}/{n_epochs} | LR: {current_lr:.6f} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | BLEU: {val_metrics['bleu']:.2f}"
                )

                mlflow.log_metrics(
                    {
                        "learning_rate": current_lr,
                        "train_loss": train_metrics["loss"],
                        "train_perplexity": train_metrics["perplexity"],
                        "val_loss": val_metrics["loss"],
                        "val_perplexity": val_metrics["perplexity"],
                        "val_bleu": val_metrics["bleu"],
                    },
                    step=epoch,
                )

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    torch.save(model.state_dict(), model_save_path)
                    mlflow.log_artifact(model_save_path, artifact_path="weights")

        print(f"[INFO] Finished {current_model.upper()}.")


if __name__ == "__main__":
    main()
