import json
import math
import os

import torch
import torch.nn as nn
from torchinfo import summary
from torchmetrics.text import BLEUScore
from tqdm import tqdm

from src.factory import build_model


def train_epoch(model, loader, criterion, optimizer, device, model_type, epoch_num=None):
    model.train()
    total_loss, total_samples = 0.0, 0

    desc = f"Epoch {epoch_num} - Training" if epoch_num else "Training"
    pbar = tqdm(loader, desc=desc, leave=False)
    for src, tgt in pbar:
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        optimizer.zero_grad()

        if model_type == "transformer":
            predictions = model(src, tgt[:, :-1])
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = tgt[:, 1:].reshape(-1)
        else:
            predictions = model(src, tgt, teacher_forcing_ratio=0.5)
            predictions = predictions[:, 1:].reshape(-1, predictions.shape[-1])
            targets = tgt[:, 1:].reshape(-1)

        loss = criterion(predictions, targets)
        loss.backward()

        if model_type != "transformer":
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        batch_size = src.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, model_type, tgt_tok, epoch_num=None):
    model.eval()
    total_loss, total_samples = 0.0, 0
    bleu_scorer = BLEUScore().to(device)
    all_preds, all_refs = [], []

    desc = f"Epoch {epoch_num} - Evaluating" if epoch_num else "Evaluating"
    pbar = tqdm(loader, desc=desc, leave=False)
    for src, tgt in pbar:
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        batch_size = src.size(0)

        if model_type == "transformer":
            loss_preds = model(src, tgt[:, :-1]).reshape(-1, len(tgt_tok))
        else:
            loss_preds = model(src, tgt, teacher_forcing_ratio=0.0)[:, 1:].reshape(-1, len(tgt_tok))

        targets = tgt[:, 1:].reshape(-1)
        loss = criterion(loss_preds, targets)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        generated = torch.full((batch_size, 1), tgt_tok.SOS, dtype=torch.long, device=device)
        for _ in range(50):
            if model_type == "transformer":
                out = model(src, generated)
            else:
                dummy_tgt = torch.cat(
                    [generated, torch.zeros(batch_size, 50 - generated.size(1), dtype=torch.long, device=device)], dim=1
                )
                out = model(src, dummy_tgt, teacher_forcing_ratio=0.0)[:, : generated.size(1)]

            next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (generated == tgt_tok.EOS).any(dim=1).all():
                break

        for i in range(batch_size):
            all_preds.append(tgt_tok.decode(generated[i]))
            all_refs.append([tgt_tok.decode(tgt[i])])

    return total_loss / total_samples, bleu_scorer(all_preds, all_refs).item() * 100


def run_training(args, train_loader, val_loader, src_tok, tgt_tok, device, run_dir):
    model = build_model(args.model, len(src_tok), len(tgt_tok), tgt_tok.PAD, device, **vars(args))

    lr, weight_decay = (1e-4, 1e-4) if args.model == "transformer" else (5e-3, 1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.PAD, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
    history = {"train": {"loss": [], "perplexity": []}, "val": {"loss": [], "perplexity": [], "bleu": []}}
    best_loss = float("inf")
    start_epoch = 1
    early_stop_counter = 0

    # Load from checkpoint if --resume flag is called [2]
    if args.resume and os.path.exists(checkpoint_path):
        tqdm.write(f"[INFO] Found checkpoint. Resuming training on {device}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        history = checkpoint["history"]
        early_stop_counter = checkpoint["early_stop_counter"]
        tqdm.write(f"[INFO] Resumed successfully. Starting from Epoch {start_epoch} (Prior Best Loss: {best_loss:.4f})")
    else:
        with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        if summary is not None:
            try:
                dummy_src = torch.randint(0, len(src_tok), (2, 50)).to(device)
                dummy_tgt = torch.randint(0, len(tgt_tok), (2, 50)).to(device)
                arch_str = repr(summary(model, input_data=(dummy_src, dummy_tgt), verbose=0))
                with open(os.path.join(run_dir, "architecture.txt"), "w") as f:
                    f.write(arch_str)
            except Exception as e:
                tqdm.write(f"[WARNING] Could not generate architecture.txt: {e}")

    tqdm.write(f"\n[INFO] Run Directory: {run_dir}")
    tqdm.write(f"[INFO] Training {args.model.upper()} | LR: {lr} | Epochs: {args.epochs} | Patience: {args.patience}\n")

    epoch_bar = tqdm(range(start_epoch, args.epochs + 1), desc="Epoch Progress", unit="epoch")
    for epoch in epoch_bar:
        t_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.model, epoch_num=epoch)
        v_loss, v_bleu = evaluate(model, val_loader, criterion, device, args.model, tgt_tok, epoch_num=epoch)

        history["train"]["loss"].append(t_loss)
        history["train"]["perplexity"].append(math.exp(t_loss))
        history["val"]["loss"].append(v_loss)
        history["val"]["perplexity"].append(math.exp(v_loss))
        history["val"]["bleu"].append(v_bleu)

        tqdm.write(
            f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | BLEU: {v_bleu:.2f}"
        )

        is_best = v_loss < best_loss
        if is_best:
            best_loss = v_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, f"best_{args.model}.pth"))
        else:
            early_stop_counter += 1

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "history": history,
            "early_stop_counter": early_stop_counter,
        }
        torch.save(checkpoint_state, checkpoint_path)

        epoch_bar.set_postfix({"Train Loss": f"{t_loss:.4f}", "Val Loss": f"{v_loss:.4f}", "BLEU": f"{v_bleu:.2f}%"})

        if early_stop_counter >= args.patience:
            tqdm.write(
                f"\n[INFO] Early Stopping triggered at Epoch {epoch:02d}. No improvement for {args.patience} epochs."
            )
            break

    with open(os.path.join(run_dir, f"{args.model}_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    tqdm.write(f"\n[SUCCESS] Run Complete. All artifacts saved to {run_dir}")
