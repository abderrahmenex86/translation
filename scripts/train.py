import argparse
import json
import os
import random
import sys

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data_utils import PadCollate, TranslationDataset, filter_pairs, read_text_file
from src.models.factory import get_model
from src.tokenizer import BasicTokenizer
from src.trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["all", "rnn", "lstm", "gru", "transformer"])
    args = parser.parse_args()

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    n_epochs = 15
    batch_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(4, os.cpu_count() or 1)

    os.makedirs(os.path.join(BASE_DIR, "artifacts"), exist_ok=True)

    print("[INFO] Loading datasets...")
    raw_src_train = read_text_file(os.path.join(BASE_DIR, "dataset/tatoeba/train.en"))
    raw_tgt_train = read_text_file(os.path.join(BASE_DIR, "dataset/tatoeba/train.fr"))
    raw_src_val = read_text_file(os.path.join(BASE_DIR, "dataset/tatoeba/val.en"))
    raw_tgt_val = read_text_file(os.path.join(BASE_DIR, "dataset/tatoeba/val.fr"))

    raw_src_train, raw_tgt_train = filter_pairs(raw_src_train, raw_tgt_train, max_len=50)
    raw_src_val, raw_tgt_val = filter_pairs(raw_src_val, raw_tgt_val, max_len=50)

    print("[INFO] Fitting tokenizers...")
    src_tokenizer = BasicTokenizer(min_freq=2)
    tgt_tokenizer = BasicTokenizer(min_freq=2)
    src_tokenizer.fit(raw_src_train, max_vocab=15000)
    tgt_tokenizer.fit(raw_tgt_train, max_vocab=15000)

    src_vocab_path = os.path.join(BASE_DIR, "artifacts/src_vocab.json")
    tgt_vocab_path = os.path.join(BASE_DIR, "artifacts/tgt_vocab.json")
    src_tokenizer.save_vocab(src_vocab_path)
    tgt_tokenizer.save_vocab(tgt_vocab_path)

    train_dataset = TranslationDataset(raw_src_train, raw_tgt_train, src_tokenizer, tgt_tokenizer)
    val_dataset = TranslationDataset(raw_src_val, raw_tgt_val, src_tokenizer, tgt_tokenizer)

    collate_fn = PadCollate(pad_idx=tgt_tokenizer.PAD)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
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

            mlflow.log_artifact(src_vocab_path, artifact_path="tokenizers")
            mlflow.log_artifact(tgt_vocab_path, artifact_path="tokenizers")

            history = train(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                n_epochs,
                current_model,
                tgt_tokenizer,
            )

            history_path = os.path.join(BASE_DIR, f"artifacts/{current_model}_history.json")
            with open(history_path, "w") as f:
                json.dump(history, f)
            mlflow.log_artifact(history_path, artifact_path="metrics")

            model_save_path = os.path.join(BASE_DIR, f"artifacts/best_model_{current_model}.pth")
            if os.path.exists(model_save_path):
                mlflow.log_artifact(model_save_path, artifact_path="weights")

        print(f"[INFO] Finished {current_model.upper()}.")


if __name__ == "__main__":
    main()
