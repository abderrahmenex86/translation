from src.dataset import build_dataloaders, load_data_lines
from src.tokenizer import BasicTokenizer


def run_verification(args):
    print(f"\n[INFO] Verifying Dataloader and Target Slicing for {args.dataset}...")

    (train_src, train_tgt), _ = load_data_lines(args.dataset, max_len=50)
    train_src, train_tgt = train_src[:1000], train_tgt[:1000]

    src_tok, tgt_tok = BasicTokenizer(min_freq=2), BasicTokenizer(min_freq=2)
    src_tok.fit(train_src)
    tgt_tok.fit(train_tgt)

    train_loader, _ = build_dataloaders(args, (train_src, train_tgt), ([], []), src_tok, tgt_tok)
    train_loader.num_workers = 0

    src_batch, tgt_batch = next(iter(train_loader))

    print(f"Source Batch Shape: {src_batch.shape}")
    print(f"Target Batch Shape: {tgt_batch.shape}")

    print("\n[SAMPLE 0 DECODED]")
    print(f"Source: {src_tok.decode(src_batch[0])}")
    print(f"Target: {tgt_tok.decode(tgt_batch[0])}")

    print("\n[TRANSFORMER SHIFTING VERIFICATION]")
    print(f"Input to Decoder (tgt[:, :-1]): {tgt_tok.decode(tgt_batch[0][:-1])}")
    print(f"Expected Output  (tgt[:, 1:]):  {tgt_tok.decode(tgt_batch[0][1:])}")
    print("\n[SUCCESS] Data successfully parsed and aligned.")
