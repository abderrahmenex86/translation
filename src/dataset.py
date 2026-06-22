import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class TranslationDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, src_tok, tgt_tok):
        self.src_tensors, self.tgt_tensors = [], []
        for src, tgt in zip(src_lines, tgt_lines):
            src_encoded = [src_tok.SOS] + src_tok.encode(src) + [src_tok.EOS]
            tgt_encoded = [tgt_tok.SOS] + tgt_tok.encode(tgt) + [tgt_tok.EOS]
            self.src_tensors.append(torch.tensor(src_encoded))
            self.tgt_tensors.append(torch.tensor(tgt_encoded))

    def __len__(self):
        return len(self.src_tensors)

    def __getitem__(self, idx):
        return self.src_tensors[idx], self.tgt_tensors[idx]


class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, padding_value=self.pad_idx, batch_first=True)
        tgt_padded = pad_sequence(tgt_batch, padding_value=self.pad_idx, batch_first=True)
        return src_padded, tgt_padded


def load_data_lines(dataset_dir, max_len=50):
    def read_and_filter(src_path, tgt_path):
        with open(src_path, "r", encoding="utf-8") as fs, open(tgt_path, "r", encoding="utf-8") as ft:
            src_lines, tgt_lines = [l.strip() for l in fs], [l.strip() for l in ft]

        filtered_src, filtered_tgt = [], []
        for s, t in zip(src_lines, tgt_lines):
            if len(s.split()) <= max_len and len(t.split()) <= max_len:
                filtered_src.append(s)
                filtered_tgt.append(t)
        return filtered_src, filtered_tgt

    train_src, train_tgt = read_and_filter(f"{dataset_dir}/train.en", f"{dataset_dir}/train.fr")
    val_src, val_tgt = read_and_filter(f"{dataset_dir}/val.en", f"{dataset_dir}/val.fr")
    return (train_src, train_tgt), (val_src, val_tgt)


def build_dataloaders(args, train_data, val_data, src_tok, tgt_tok):
    collate_fn = PadCollate(pad_idx=tgt_tok.PAD)

    train_loader = DataLoader(
        TranslationDataset(*train_data, src_tok, tgt_tok),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        TranslationDataset(*val_data, src_tok, tgt_tok),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    return train_loader, val_loader
