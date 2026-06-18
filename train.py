import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TranslationDataset, collate_fn
from helpers import train, translate_sentence
from tokenizer import BasicTokenizer


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


def get_model(model_type, src_vocab_size, tgt_vocab_size, pad_idx, device):
    from models.gru import DecoderGRU, EncoderGRU, Seq2SeqGRU
    from models.lstm import DecoderLSTM, EncoderLSTM, Seq2SeqLSTM
    from models.rnn import DecoderRNN, EncoderRNN, Seq2SeqRNN
    from models.transformer import Transformer

    embed_size = 256
    hidden_size = 512
    num_layers = 1
    dropout = 0.7

    if model_type == "rnn":
        encoder = EncoderRNN(src_vocab_size, embed_size, hidden_size, num_layers, dropout)
        decoder = DecoderRNN(tgt_vocab_size, embed_size, hidden_size, num_layers, dropout)
        return Seq2SeqRNN(encoder, decoder, device).to(device)
    elif model_type == "lstm":
        encoder = EncoderLSTM(src_vocab_size, embed_size, hidden_size, num_layers, dropout)
        decoder = DecoderLSTM(tgt_vocab_size, embed_size, hidden_size, num_layers, dropout)
        return Seq2SeqLSTM(encoder, decoder, device).to(device)
    elif model_type == "gru":
        encoder = EncoderGRU(src_vocab_size, embed_size, hidden_size, num_layers, dropout)
        decoder = DecoderGRU(tgt_vocab_size, embed_size, hidden_size, num_layers, dropout)
        return Seq2SeqGRU(encoder, decoder, device).to(device)
    elif model_type == "transformer":
        return Transformer(
            src_vocab_size,
            tgt_vocab_size,
            pad_idx,
            d_model=embed_size,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
        ).to(device)
    else:
        raise ValueError("Unknown model type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Translation Comparative Study Training")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "rnn", "lstm", "gru", "transformer"],
        help="Specify the model to train. Defaults to 'all' to run the full comparative study sequentially.",
    )
    args = parser.parse_args()

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.backends.cudnn.benchmark = True

    n_epochs = 50
    batch_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("-> Loading datasets")
    raw_src_train = read_text_file("dataset/multi30k/train.en")
    raw_tgt_train = read_text_file("dataset/multi30k/train.fr")

    raw_src_val = read_text_file("dataset/multi30k/val.en")
    raw_tgt_val = read_text_file("dataset/multi30k/val.fr")

    raw_src_train, raw_tgt_train = filter_pairs(raw_src_train, raw_tgt_train, max_len=50)
    raw_src_val, raw_tgt_val = filter_pairs(raw_src_val, raw_tgt_val, max_len=50)
    print("-> Datasets loaded and filterd.")

    src_tokenizer = BasicTokenizer(min_freq=2)
    tgt_tokenizer = BasicTokenizer(min_freq=2)

    print("-> Fitting tokenizers")
    src_tokenizer.fit(raw_src_train)
    tgt_tokenizer.fit(raw_tgt_train)

    print(f"English Vocab Size: {len(src_tokenizer)}")
    print(f"French Vocab Size: {len(tgt_tokenizer)}")

    train_dataset = TranslationDataset(raw_src_train, raw_tgt_train, src_tokenizer, tgt_tokenizer)
    val_dataset = TranslationDataset(raw_src_val, raw_tgt_val, src_tokenizer, tgt_tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    if args.model == "all":
        models_to_train = ["rnn", "lstm", "gru", "transformer"]
    else:
        models_to_train = [args.model]

    for current_model in models_to_train:
        print("\n" + "=" * 50)
        print(f" Initializing and Training: {current_model.upper()}")
        print("=" * 50)

        model = get_model(
            current_model,
            src_vocab_size=len(src_tokenizer),
            tgt_vocab_size=len(tgt_tokenizer),
            pad_idx=tgt_tokenizer.PAD,
            device=device,
        )

        criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.PAD, label_smoothing=0.1)

        if current_model == "transformer":
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-2)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        history = train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            n_epochs,
            current_model,
            tgt_tokenizer,
        )

        with open(f"{current_model}_history.json", "w") as f:
            json.dump(history, f)

        print(f"-> Finished training {current_model.upper()}. History saved to {current_model}_history.json\n")
