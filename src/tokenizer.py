import json
import re
from collections import Counter


class BasicTokenizer:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2
        self.UNK = 3

    def normalize(self, s):
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-ZÀ-ÿ.!?]+", r" ", s)
        return s

    def fit(self, sentences, max_vocab=15000):
        counter = Counter()
        for sentence in sentences:
            words = self.normalize(sentence).split()
            counter.update(words)

        self.word2idx = {"<PAD>": self.PAD, "<SOS>": self.SOS, "<EOS>": self.EOS, "<UNK>": self.UNK}
        most_common_words = counter.most_common(max_vocab - 4)

        idx = 4
        for word, count in most_common_words:
            if count >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1

        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode(self, sentence):
        words = self.normalize(sentence).split()
        return [self.word2idx.get(w, self.UNK) for w in words]

    def decode(self, indices):
        return " ".join(
            [self.idx2word.get(idx, "<UNK>") for idx in indices if idx not in (self.PAD, self.SOS, self.EOS)]
        )

    def __len__(self):
        return len(self.word2idx)

    def save_vocab(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=4)

    def load_vocab(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
