import argparse
import os
import sys

import mlflow
import optuna
import torch
import torch.nn as nn
from optuna.integration.mlflow import MLflowCallback
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data_utils import PadCollate, TranslationDataset, filter_pairs, read_text_file
from src.models.factory import get_model
from src.tokenizer import BasicTokenizer
from src.trainer import evaluate, train_epoch


class OptimizationObjective:
    def __init__(self, model_type, train_loader, val_loader, src_tok, tgt_tok, device):
        self.model_type = model_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.device = device

    def __call__(self, trial):
        if self.model_type == "transformer":
            lr = 5e-4
            weight_decay = 1e-4
            dropout = 0.1
        else:
            lr = 5e-3
            weight_decay = 1e-2
            dropout = 0.2

        model_kwargs = {"dropout": dropout}

        if self.model_type in ["rnn", "lstm", "gru"]:
            model_kwargs["embed_size"] = trial.suggest_categorical("embed_size", [128, 256, 512])
            model_kwargs["hidden_size"] = trial.suggest_categorical("hidden_size", [256, 512, 1024])
            model_kwargs["num_layers"] = trial.suggest_int("num_layers", 1, 4)

        elif self.model_type == "transformer":
            model_kwargs["d_model"] = trial.suggest_categorical("d_model", [128, 256, 512])
            model_kwargs["nhead"] = trial.suggest_categorical("nhead", [4, 8])
            model_kwargs["num_encoder_layers"] = trial.suggest_int("num_encoder_layers", 2, 6)
            model_kwargs["num_decoder_layers"] = trial.suggest_int("num_decoder_layers", 2, 6)
            model_kwargs["dim_feedforward"] = trial.suggest_categorical("dim_feedforward", [512, 1024, 2048])

        print(f"[INFO] Trial {trial.number} starting. Arch Params: {trial.params}")

        model = get_model(
            self.model_type, len(self.src_tok), len(self.tgt_tok), self.tgt_tok.PAD, self.device, **model_kwargs
        )

        criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_tok.PAD, label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(5):
            train_epoch(model, self.train_loader, criterion, optimizer, self.device, self.model_type)
            val_metrics = evaluate(model, self.val_loader, criterion, self.device, self.model_type, self.tgt_tok)

            trial.report(val_metrics["loss"], epoch)
            if trial.should_prune():
                print(f"[INFO] Trial {trial.number} pruned at epoch {epoch}.")
                raise optuna.exceptions.TrialPruned()

        return val_metrics["loss"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["rnn", "lstm", "gru", "transformer"])
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing optimization for {args.model.upper()} on {device.type.upper()}")

    print("[INFO] Loading data subset for optimization...")
    raw_src = read_text_file(os.path.join(BASE_DIR, "dataset/multi30k/train.en"))[:10000]
    raw_tgt = read_text_file(os.path.join(BASE_DIR, "dataset/multi30k/train.fr"))[:10000]
    raw_src, raw_tgt = filter_pairs(raw_src, raw_tgt, max_len=50)

    src_tok = BasicTokenizer(min_freq=2)
    tgt_tok = BasicTokenizer(min_freq=2)
    src_tok.fit(raw_src)
    tgt_tok.fit(raw_tgt)

    dataset = TranslationDataset(raw_src, raw_tgt, src_tok, tgt_tok)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    collate_fn = PadCollate(pad_idx=tgt_tok.PAD)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=2)

    mlflow.set_experiment(f"Optimization_{args.model.upper()}")

    mlflc = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="val_loss", create_experiment=False)

    objective = OptimizationObjective(args.model, train_loader, val_loader, src_tok, tgt_tok, device)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

    print("[INFO] Starting optimization loop...")
    study.optimize(objective, n_trials=args.trials, callbacks=[mlflc])

    print("\n[INFO] Optimization Complete.")
    print("[INFO] Best Trial Details:")
    trial = study.best_trial
    print(f"  Validation Loss: {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
