import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def read_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def filter_pairs(src_lines, tgt_lines, max_len=50):
    filtered_src, filtered_tgt = [], []
    for src, tgt in zip(src_lines, tgt_lines):
        if len(src.split()) <= max_len and len(tgt.split()) <= max_len:
            filtered_src.append(src)
            filtered_tgt.append(tgt)
    return filtered_src, filtered_tgt


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_tokenizer, tgt_tokenizer):
        self.src_tensors = []
        self.tgt_tensors = []
        for src, tgt in zip(src_sentences, tgt_sentences):
            src_encoded = [src_tokenizer.SOS] + src_tokenizer.encode(src) + [src_tokenizer.EOS]
            tgt_encoded = [tgt_tokenizer.SOS] + tgt_tokenizer.encode(tgt) + [tgt_tokenizer.EOS]
            self.src_tensors.append(torch.tensor(src_encoded))
            self.tgt_tensors.append(torch.tensor(tgt_encoded))

    def __len__(self):
        return len(self.src_tensors)

    def __getitem__(self, idx):
        return self.src_tensors[idx], self.tgt_tensors[idx]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_padded = pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    return src_padded, tgt_padded
