import json
import math

import torch
import torch.nn as nn
from torchmetrics.text import BLEUScore

from src.factory import build_model


def train_epoch(model, loader, criterion, optimizer, device, model_type):
    model.train()
    total_loss, total_samples = 0.0, 0

    for src, tgt in loader:
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

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, model_type, tgt_tok):
    model.eval()
    total_loss, total_samples = 0.0, 0
    bleu_scorer = BLEUScore().to(device)
    all_preds, all_refs = [], []

    for src, tgt in loader:
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


def run_training(args, train_loader, val_loader, src_tok, tgt_tok, device):
    model = build_model(args.model, len(src_tok), len(tgt_tok), tgt_tok.PAD, device, **vars(args))

    lr, weight_decay = (1e-4, 1e-4) if args.model == "transformer" else (5e-3, 1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.PAD, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train": {"loss": [], "perplexity": []}, "val": {"loss": [], "perplexity": [], "bleu": []}}
    best_loss = float("inf")

    print(f"\n[INFO] Training {args.model.upper()} | LR: {lr} | Epochs: {args.epochs}")
    for epoch in range(1, args.epochs + 1):
        t_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.model)
        v_loss, v_bleu = evaluate(model, val_loader, criterion, device, args.model, tgt_tok)

        history["train"]["loss"].append(t_loss)
        history["train"]["perplexity"].append(math.exp(t_loss))
        history["val"]["loss"].append(v_loss)
        history["val"]["perplexity"].append(math.exp(v_loss))
        history["val"]["bleu"].append(v_bleu)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | BLEU: {v_bleu:.2f}"
        )

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), f"artifacts/best_{args.model}.pth")

    with open(f"artifacts/{args.model}_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print(f"[SUCCESS] Training Complete. History saved to artifacts/{args.model}_history.json")
