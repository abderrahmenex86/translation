import random

import torch
import torch.nn as nn


class Seq2SeqRNN(nn.Module):
    def __init__(self, cell_type, src_vocab, tgt_vocab, embed_size=256, hidden_size=512, num_layers=2, dropout=0.2):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell_type]
        drop_rate = dropout if num_layers > 1 else 0

        self.encoder_emb = nn.Embedding(src_vocab, embed_size)
        self.encoder_rnn = rnn_cls(embed_size, hidden_size, num_layers, batch_first=True, dropout=drop_rate)

        self.decoder_emb = nn.Embedding(tgt_vocab, embed_size)
        self.decoder_rnn = rnn_cls(embed_size, hidden_size, num_layers, batch_first=True, dropout=drop_rate)
        self.fc_out = nn.Linear(hidden_size, tgt_vocab)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        outputs = torch.zeros(batch_size, tgt_len, self.fc_out.out_features, device=src.device)

        _, state = self.encoder_rnn(self.encoder_emb(src))
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            emb = self.decoder_emb(input_token.unsqueeze(1))
            out, state = self.decoder_rnn(emb, state)
            pred = self.fc_out(out.squeeze(1))

            outputs[:, t] = pred
            input_token = tgt[:, t] if random.random() < teacher_forcing_ratio else pred.argmax(1)

        return outputs
