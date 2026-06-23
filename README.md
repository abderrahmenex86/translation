# PyTorch NMT Sandbox

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/Optuna-20232A?style=for-the-badge&logo=python&logoColor=white" alt="Optuna" />
    <img src="https://img.shields.io/badge/TQDM-FFD700?style=for-the-badge&logo=python&logoColor=black" alt="TQDM" />
    <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib" />
  </p>
</div>

PyTorch NMT Sandbox is an end-to-end Neural Machine Translation (NMT) system built completely from scratch using pure PyTorch. This project performs a comparative architectural study comparing Vanilla RNN, LSTM, GRU, and Transformer architectures on the Tatoeba English-French dataset (~230k parallel sentence pairs).

<div align="center">
  <img src="artifacts/transformer_metrics.png" width="45%" />
  <img src="artifacts/comparative_metrics.png" width="45%" />
</div>

## Features

- **No HuggingFace or High-Level Abstractions**: Implements a custom word-level tokenizer preserving French accent configurations (`r"[^a-zA-ZÀ-ÿ.!?]+"`) and a dynamic padding collation pipeline from scratch.
- **Unified Recurrent & Self-Attention Pipelines**:
  - **Seq2Seq RNN Family**: Dynamically maps Vanilla RNN, LSTM, and GRU cells into a single unified encoder-decoder architecture.
  - **Transformer Architecture**: Implements a full Transformer from scratch featuring the critical attention scaling adjustment ($\sqrt{d_{model}}$) before applying positional encodings.
- 🚀 **Hardware-Optimized Dataloaders**: Specifically tuned for high-performance parallel computing (e.g., 24-core CPUs and RTX 4060/5060 Ti GPUs) leveraging multi-process prefetching (`num_workers=20`, `prefetch_factor=4`).
- 💾 **Fault-Tolerant Checkpointing & Resuming**: Saves the entire state of the model, optimizer, epoch history, and early stopping state at the end of every epoch. Resuming via `--resume` automatically recovers the matching vocabulary files to avoid representation-shift weight corruption.
- 🧠 **Architectural Early Stopping & Tuning**: Exposes independent architectural parameters to automated Optuna searches (while keeping training parameters static) and dumps an `optuna_summary.json` file.
- 💬 **Interactive Evaluation Mode**: Generates translations dynamically on the terminal using fully auto-regressive decoding stopping at the `<EOS>` token.

---

## Tech Stack

- **Machine Learning & Frameworks:** PyTorch, Torchmetrics (BLEU Score), Torchinfo
- **Hyperparameter Optimization:** Optuna
- **Visualizations & Tracking:** Matplotlib, JSON-based Metrics Logging

---

## Folder Structure

```text
mt_project/
├── dataset/
│   └── tatoeba/             # Downloaded parallel train and val splits
├── artifacts/               # Run subfolders storing vocab, checkpoints, logs, and plots
├── src/
│   ├── tokenizer.py         # Custom BasicTokenizer
│   ├── dataset.py           # Dataset class and optimized DataLoaders
│   ├── rnn_models.py        # Unified Recurrent models
│   ├── transformer.py       # Transformer model & Positional Encoding
│   ├── factory.py           # Model assembly logic
│   ├── trainer.py           # Epoch loops and run configurations
│   ├── optimize.py          # Optuna integration
│   ├── infer.py             # Autoregressive interactive translation
│   ├── tester.py            # Gradient & Forward pass checks
│   └── verify.py            # Dataloader & target slicing checks
├── main.py                  # CLI Entrypoint for ML Lifecycle (train/test/optimize/infer)
├── tools.py                 # CLI Entrypoint for Utilities (download/verify/plot)
├── push.sh                  # Exclude-aware local-to-cloud upload script
└── fetch.sh                 # Latest-run-aware cloud-to-local download script
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- A CUDA-capable GPU (RTX 30/40/50 series recommended)
- `pip install torch torchmetrics torchinfo optuna matplotlib tqdm`

### Installation & Initialization

1. **Clone the repository:**
```bash
git clone https://github.com/abderrahmenex86/translation.git
cd translation
```

2. **Download and prepare the Tatoeba dataset:**
```bash
python tools.py --mode download
```

3. **Verify dataloaders and target shifts:**
```bash
python tools.py --mode verify
```

---

## Workflow Guide

### 1. Run a Sanity Test
Before running long training runs, check the memory layout, model output shapes, and backward gradient flow:
```bash
python main.py --mode test --model transformer
```

### 2. Run Architectural Hyperparameter Sweeps
Optimize model shapes (e.g., layers, embed dimension, hidden size) for a specific architecture using Optuna:
```bash
python main.py --mode optimize --model transformer
```
*The resulting configuration will write to `artifacts/<timestamp>_optimize/optuna_summary.json`.*

### 3. Start Model Training (With Early Stopping)
Kick off standard training runs using your optimized configurations. Training will exit early if the validation loss fails to improve for 7 epochs:
```bash
python main.py --mode train --model transformer --d_model 256 --dim_ff 1024 --num_enc 3 --num_dec 3 --epochs 50 --patience 7
```

### 4. Resume interrupted training runs
If your training gets interrupted or interrupted spot instance goes offline, push your code or log back in and run:
```bash
# This automatically finds the absolute latest folder in artifacts/ and continues training!
python main.py --mode train --model transformer --resume
```

Alternatively, if you want to explicitly resume from a *specific* folder, you can target it directly using `--run_dir`:
```bash
python main.py --mode train --model transformer --resume --run_dir artifacts/20260623_120000_transformer_d256
```

### 5. Generate Training Curves & Comparisons
Visualize the metrics (Loss, Perplexity, BLEU) for a single run or compare all architectures side-by-side:
```bash
# Plot metrics for the latest run
python tools.py --mode plot --model transformer

# Compare all runs side-by-side
python tools.py --mode plot --model all
```

### 6. Interactive Live Inference
Test your model’s conversational translations on custom prompts directly inside your terminal:
```bash
python main.py --mode infer --model transformer
```
