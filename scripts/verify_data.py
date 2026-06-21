import os
import sys

from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import TranslationDataset, collate_fn, filter_pairs, read_text_file
from src.tokenizer import BasicTokenizer


def main():
    print("[INFO] Verifying DataLoader and Alignments...")

    raw_src = read_text_file("dataset/tatoeba/train.en")[:1000]
    raw_tgt = read_text_file("dataset/tatoeba/train.fr")[:1000]
    raw_src, raw_tgt = filter_pairs(raw_src, raw_tgt, max_len=50)

    src_tok = BasicTokenizer(min_freq=2)
    tgt_tok = BasicTokenizer(min_freq=2)
    src_tok.fit(raw_src)
    tgt_tok.fit(raw_tgt)

    dataset = TranslationDataset(raw_src, raw_tgt, src_tok, tgt_tok)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    src_batch, tgt_batch = next(iter(loader))

    print(f"Source Batch Shape: {src_batch.shape}")
    print(f"Target Batch Shape: {tgt_batch.shape}")

    def decode(tensor, tok):
        return " ".join([tok.idx2word.get(idx.item(), f"<UNK:{idx.item()}>") for idx in tensor])

    print("\n[SAMPLE 0 DECODED]")
    print(f"Source: {decode(src_batch[0], src_tok)}")
    print(f"Target: {decode(tgt_batch[0], tgt_tok)}")

    print("\n[TRANSFORMER SLICING VERIFICATION]")
    print(f"Input   (tgt[:, :-1]): {decode(tgt_batch[0][:-1], tgt_tok)}")
    print(f"Predict (tgt[:, 1:]):  {decode(tgt_batch[0][1:], tgt_tok)}")


if __name__ == "__main__":
    main()
