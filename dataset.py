import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class OldTranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_tokenizer, tgt_tokenizer):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = [self.src_tok.SOS] + self.src_tok.encode(self.src_sentences[idx]) + [self.src_tok.EOS]
        tgt = [self.tgt_tok.SOS] + self.tgt_tok.encode(self.tgt_sentences[idx]) + [self.tgt_tok.EOS]
        return torch.tensor(src), torch.tensor(tgt)


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
