import json
import re
from collections import Counter

from tqdm import tqdm


class BasicTokenizer:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.PAD, self.SOS, self.EOS, self.UNK = 0, 1, 2, 3
        self.word2idx = {"<PAD>": self.PAD, "<SOS>": self.SOS, "<EOS>": self.EOS, "<UNK>": self.UNK}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def normalize(self, s):
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-ZÀ-ÿ.!?]+", r" ", s)
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


class BPETokenizer:
    def __init__(self, vocab_size=8000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.PAD, self.SOS, self.EOS, self.UNK = 0, 1, 2, 3
        self.word2idx = {"<PAD>": self.PAD, "<SOS>": self.SOS, "<EOS>": self.EOS, "<UNK>": self.UNK}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.merges = []
        self.cache = {}

    def normalize(self, s):
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-ZÀ-ÿ.!?]+", r" ", s)
        return s

    def fit(self, sentences, max_vocab=8000):
        self.vocab_size = max_vocab

        word_freqs = Counter()
        for sentence in sentences:
            words = self.normalize(sentence).split()
            for w in words:
                word_freqs[w + "</w>"] += 1

        chars = set()
        for word in word_freqs:
            chars.update(word)

        idx = 4
        for char in sorted(chars):
            if char not in self.word2idx:
                self.word2idx[char] = idx
                idx += 1

        splits = {tuple(word): freq for word, freq in word_freqs.items()}

        # Added TQDM Progress bar and Fast-Path skipping
        target_merges = self.vocab_size - len(self.word2idx)
        with tqdm(total=target_merges, desc="Training BPE", leave=False) as pbar:
            while len(self.word2idx) < self.vocab_size:
                pairs = self._get_stats(splits)
                if not pairs:
                    break

                best_pair = max(pairs, key=pairs.get)
                if pairs[best_pair] < self.min_freq:
                    break

                splits = self._merge_vocab(best_pair, splits)
                self.merges.append(best_pair)

                merged_token = "".join(best_pair)
                self.word2idx[merged_token] = len(self.word2idx)
                pbar.update(1)

        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def _get_stats(self, splits):
        pairs = Counter()
        for word, freq in splits.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, splits):
        merged_splits = {}
        for word, freq in splits.items():
            # FAST-PATH FILTER: Skip the loop if characters aren't in the word
            if pair[0] in word and pair[1] in word:
                merged_splits[self._merge_word(word, pair)] = freq
            else:
                merged_splits[word] = freq
        return merged_splits

    def _merge_word(self, word_tuple, pair):
        new_word = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and word_tuple[i] == pair[0] and word_tuple[i + 1] == pair[1]:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        return tuple(new_word)

    def encode(self, sentence):
        words = self.normalize(sentence).split()
        encoded_tokens = []

        for word in words:
            if word in self.cache:
                encoded_tokens.extend(self.cache[word])
                continue

            word_chars = tuple(word + "</w>")
            for pair in self.merges:
                if pair[0] in word_chars and pair[1] in word_chars:
                    word_chars = self._merge_word(word_chars, pair)

            tokens = [self.word2idx.get(token, self.UNK) for token in word_chars]
            self.cache[word] = tokens
            encoded_tokens.extend(tokens)

        return encoded_tokens

    def decode(self, indices):
        indices = indices.tolist() if hasattr(indices, "tolist") else indices
        if self.EOS in indices:
            indices = indices[: indices.index(self.EOS)]

        decoded_tokens = [self.idx2word.get(idx, "<UNK>") for idx in indices if idx not in (self.PAD, self.SOS)]

        out = []
        for token in decoded_tokens:
            if token.endswith("</w>"):
                out.append(token[:-5] + " ")
            else:
                out.append(token)
        return "".join(out).strip()

    def __len__(self):
        return len(self.word2idx)

    def save_vocab(self, filepath):
        state = {"word2idx": self.word2idx, "merges": [list(pair) for pair in self.merges]}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

    def load_vocab(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.word2idx = state["word2idx"]
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.merges = [tuple(pair) for pair in state["merges"]]
