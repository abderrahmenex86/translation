from src.rnns import Seq2SeqRNN
from src.transformer import Transformer


def build_model(model_name, len_src, len_tgt, pad_idx, device, trial=None, **kwargs):
    if model_name == "transformer":
        d_model = trial.suggest_categorical("d_model", [128, 256, 512]) if trial else kwargs.get("d_model", 256)
        nhead = trial.suggest_categorical("nhead", [4, 8]) if trial else kwargs.get("nhead", 8)
        num_enc = trial.suggest_int("num_enc", 2, 6) if trial else kwargs.get("num_enc", 3)
        num_dec = trial.suggest_int("num_dec", 2, 6) if trial else kwargs.get("num_dec", 3)
        dim_ff = trial.suggest_categorical("dim_ff", [256, 512, 1024, 2048]) if trial else kwargs.get("dim_ff", 1024)

        return Transformer(
            len_src,
            len_tgt,
            pad_idx,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            dim_feedforward=dim_ff,
        ).to(device)
    else:
        embed_size = (
            trial.suggest_categorical("embed_size", [128, 256, 512]) if trial else kwargs.get("embed_size", 256)
        )
        hidden_size = (
            trial.suggest_categorical("hidden_size", [256, 512, 1024]) if trial else kwargs.get("hidden_size", 512)
        )
        num_layers = trial.suggest_int("num_layers", 1, 4) if trial else kwargs.get("num_layers", 2)

        return Seq2SeqRNN(
            model_name, len_src, len_tgt, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers
        ).to(device)
