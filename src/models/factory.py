def get_model(model_type, src_vocab_size, tgt_vocab_size, pad_idx, device, **kwargs):
    from .gru import DecoderGRU, EncoderGRU, Seq2SeqGRU
    from .lstm import DecoderLSTM, EncoderLSTM, Seq2SeqLSTM
    from .rnn import DecoderRNN, EncoderRNN, Seq2SeqRNN
    from .transformer import Transformer

    if model_type in ["rnn", "lstm", "gru"]:
        embed_size = kwargs.get("embed_size", 256)
        hidden_size = kwargs.get("hidden_size", 512)
        num_layers = kwargs.get("num_layers", 2)
        dropout = kwargs.get("dropout", 0.2)

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
        d_model = kwargs.get("d_model", 256)
        nhead = kwargs.get("nhead", 8)
        num_encoder_layers = kwargs.get("num_encoder_layers", 3)
        num_decoder_layers = kwargs.get("num_decoder_layers", 3)
        dim_feedforward = kwargs.get("dim_feedforward", 1024)
        dropout = kwargs.get("dropout", 0.1)

        return Transformer(
            src_vocab_size,
            tgt_vocab_size,
            pad_idx,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
