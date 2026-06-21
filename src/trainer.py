import math

import torch
from torchmetrics.text import BLEUScore
from tqdm.auto import tqdm


def train_epoch(model, loader, criterion, optimizer, device, model_type):
    model.train()
    total_loss, total_samples = 0.0, 0

    for src, tgt in tqdm(loader, desc="Training", leave=False, unit="batch", disable=True):
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        optimizer.zero_grad()

        if model_type == "transformer":
            tgt_input, tgt_expected = tgt[:, :-1], tgt[:, 1:]
            predictions = model(src, tgt_input)
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = tgt_expected.reshape(-1)
        else:
            predictions = model(src, tgt, teacher_forcing_ratio=0.5)
            predictions = predictions[:, 1:].reshape(-1, predictions.shape[-1])
            targets = tgt[:, 1:].reshape(-1)

        loss = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = src.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

    avg_loss = total_loss / total_samples
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}


def evaluate(model, loader, criterion, device, model_type, tgt_tokenizer):
    model.eval()
    total_loss, total_samples = 0.0, 0
    bleu = BLEUScore().to(device)
    all_preds, all_refs = [], []

    with torch.no_grad():
        for src, tgt in tqdm(loader, desc="Evaluating", leave=False, unit="batch", disable=True):
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)

            if model_type == "transformer":
                tgt_input, tgt_expected = tgt[:, :-1], tgt[:, 1:]
                loss_predictions = model(src, tgt_input)
                loss_preds = loss_predictions.reshape(-1, loss_predictions.shape[-1])
                targets = tgt_expected.reshape(-1)

                batch_size = src.shape[0]
                max_len = tgt.shape[1]
                generated = torch.full((batch_size, 1), tgt_tokenizer.SOS, dtype=torch.long, device=device)

                for _ in range(max_len - 1):
                    out = model(src, generated)
                    next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                pred_tokens = generated
            else:
                predictions = model(src, tgt, teacher_forcing_ratio=0.0)
                loss_preds = predictions[:, 1:].reshape(-1, predictions.shape[-1])
                targets = tgt[:, 1:].reshape(-1)
                pred_tokens = predictions.argmax(dim=-1)

            loss = criterion(loss_preds, targets)
            batch_size = src.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            for i in range(batch_size):
                pred_str = tgt_tokenizer.decode(pred_tokens[i].tolist())
                ref_str = tgt_tokenizer.decode(tgt[i].tolist())
                all_preds.append(pred_str)
                all_refs.append([ref_str])

    avg_loss = total_loss / total_samples
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss), "bleu": bleu(all_preds, all_refs).item() * 100}


def train(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, n_epochs, model_type, tgt_tokenizer
):
    best_val_loss = float("inf")
    history = {"train": {"loss": [], "perplexity": []}, "val": {"loss": [], "perplexity": [], "bleu": []}}

    for epoch in range(1, n_epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, model_type)
        val_metrics = evaluate(model, val_loader, criterion, device, model_type, tgt_tokenizer)
        scheduler.step(val_metrics["loss"])

        for k, v in train_metrics.items():
            history["train"][k].append(v)
        for k, v in val_metrics.items():
            history["val"][k].append(v)

        print(
            f"Epoch {epoch:02d}/{n_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | BLEU: {val_metrics['bleu']:.2f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), f"best_model_{model_type}.pth")

    return history


def translate_sentence(sentence, model, src_tokenizer, tgt_tokenizer, device, model_type, max_length=50):
    model.eval()
    tokens = [src_tokenizer.SOS] + src_tokenizer.encode(sentence) + [src_tokenizer.EOS]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == "transformer":
            tgt_indices = [tgt_tokenizer.SOS]
            for _ in range(max_length):
                tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
                output = model(src_tensor, tgt_tensor)
                pred_token = output.argmax(2)[:, -1].item()
                tgt_indices.append(pred_token)
                if pred_token == tgt_tokenizer.EOS:
                    break
        else:
            dummy_tgt = torch.zeros(1, max_length, dtype=torch.long).to(device)
            dummy_tgt[0, 0] = tgt_tokenizer.SOS
            output = model(src_tensor, dummy_tgt, teacher_forcing_ratio=0.0)
            tgt_indices = output.argmax(2).squeeze(0).tolist()
            if tgt_tokenizer.EOS in tgt_indices:
                tgt_indices = tgt_indices[: tgt_indices.index(tgt_tokenizer.EOS)]

    return tgt_tokenizer.decode(tgt_indices)
