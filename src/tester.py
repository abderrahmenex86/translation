import torch
import torch.nn as nn

from src.factory import build_model


def run_test(args, train_loader, src_tok, tgt_tok, device):
    print(f"\n[INFO] Running Sanity Check for {args.model.upper()} Architecture...")

    print("\n[TEST 1] Instantiating model and optimizers...")
    model = build_model(args.model, len(src_tok), len(tgt_tok), tgt_tok.PAD, device, **vars(args))
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.PAD)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print("  -> Model instantiated successfully.")

    print("\n[TEST 2] Checking Training Forward and Backward Pass...")
    model.train()
    src, tgt = next(iter(train_loader))
    src, tgt = src.to(device), tgt.to(device)

    print(f"  -> Source Batch Shape: {src.shape}")
    print(f"  -> Target Batch Shape: {tgt.shape}")

    optimizer.zero_grad()
    if args.model == "transformer":
        predictions = model(src, tgt[:, :-1])
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = tgt[:, 1:].reshape(-1)
    else:
        predictions = model(src, tgt, teacher_forcing_ratio=0.5)
        predictions = predictions[:, 1:].reshape(-1, predictions.shape[-1])
        targets = tgt[:, 1:].reshape(-1)

    print(f"  -> Predictions Shape (Flattened): {predictions.shape}")
    print(f"  -> Targets Shape (Flattened): {targets.shape}")

    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()

    print(f"  -> Loss calculated successfully: {loss.item():.4f}")
    print("  -> Backward pass successful. Gradients updated.")

    print("\n[TEST 3] Checking Evaluation (Auto-Regressive Generation)...")
    model.eval()
    with torch.no_grad():
        batch_size = src.size(0)
        generated = torch.full((batch_size, 1), tgt_tok.SOS, dtype=torch.long, device=device)

        for _ in range(3):
            if args.model == "transformer":
                out = model(src, generated)
            else:
                dummy_tgt = torch.cat(
                    [generated, torch.zeros(batch_size, 3 - generated.size(1), dtype=torch.long, device=device)], dim=1
                )
                out = model(src, dummy_tgt, teacher_forcing_ratio=0.0)[:, : generated.size(1)]

            next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    print(f"  -> Generation logic executed correctly. Output shape: {generated.shape}")

    print("\n" + "=" * 60)
    print(f"[SUCCESS] All sanity checks passed! You are ready to run '--mode train'.")
    print("=" * 60 + "\n")
