import json
import re
from collections import Counter


class BasicTokenizer:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.PAD, self.SOS, self.EOS, self.UNK = 0, 1, 2, 3
        self.word2idx = {"<PAD>": self.PAD, "<SOS>": self.SOS, "<EOS>": self.EOS, "<UNK>": self.UNK}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def normalize(self, s):
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-ZÀ-ÿ.!?]+", r" ", s)  # Preserves French accents correctly
        return s

    def fit(self, sentences, max_vocab=15000):
        counter = Counter()
        for sentence in sentences:
            counter.update(self.normalize(sentence).split())

        most_common = counter.most_common(max_vocab - 4)
        idx = 4
        for word, count in most_common:
            if count >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode(self, sentence):
        return [self.word2idx.get(w, self.UNK) for w in self.normalize(sentence).split()]

    def decode(self, indices):
        indices = indices.tolist() if hasattr(indices, "tolist") else indices
        if self.EOS in indices:
            indices = indices[: indices.index(self.EOS)]
        return " ".join([self.idx2word.get(idx, "<UNK>") for idx in indices if idx not in (self.PAD, self.SOS)])

    def __len__(self):
        return len(self.word2idx)

    def save_vocab(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=4)

    def load_vocab(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
