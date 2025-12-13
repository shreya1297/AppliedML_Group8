"""
Encoder-only; nn.TransformerEncoder. 
4 layers: set num_layers=4. d_model=256, nhead=8, head_dim=32: matches our EMBED_D_MODEL=256 and nhead=8. 
FFN=1024 (~4times): matches dim_ff=1024. Dropout=0.1: matches our embeddings.py dropout and encoder layer dropout. 
Max length=128: correct if EMBED_MAX_LEN=128 and we pass that to preprocessing_scratch. 
Pooling: we currently use CLS pooling in the run shown; we can evaluate masked mean pooling as an ablation because your model supports it. 
We use a dedicated [CLS] token prepended to each sequence and pool from its final hidden state.
We selected a small encoder-only Transformer appropriate for scratch training and evaluate targeted ablations over depth, heads, learning rate, 
sequence length, and pooling; final results are reported with validation accuracy and seed-averaged performance.

"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset

from preprocessing import load_data, split_dataset
from tokenizer_trainonly import TrainOnlyVocab
from preprocessing_scratch import preprocess_mc_batch_scratch
from transformer_model import TransformerMCQModel
from config import TRAIN_CSV, TEST_CSV, EMBED_MAX_LEN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def acc(preds, labels):
    return (preds == labels).float().mean().item()


def build_vocab_and_splits(seed_for_training: int):
    """
    Uses your existing stratified split in preprocessing.py.
    Note: split_dataset currently uses random_state=42 (fixed split).
    Seeds here control training randomness (init + dataloader shuffle).
    """
    df = load_data(TRAIN_CSV)
    train_set, val_set = split_dataset(df)

    train_df_for_vocab = train_set.to_pandas()
    vocab = TrainOnlyVocab.build(train_df_for_vocab, max_vocab=30000, min_freq=2)

    return train_set, val_set, vocab


def make_loaders(train_set, val_set, vocab, max_len: int, batch_size: int):
    train_set = train_set.map(
        lambda b: preprocess_mc_batch_scratch(b, vocab, max_length=max_len),
        batched=True
    )
    val_set = val_set.map(
        lambda b: preprocess_mc_batch_scratch(b, vocab, max_length=max_len),
        batched=True
    )

    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    return train_loader, val_loader


def train_one_run(config: dict, seed: int):
    """
    Trains one model and returns:
      - best_val_acc
      - path to best checkpoint
      - vocab object
      - device string
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("submission", exist_ok=True)

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== RUN seed={seed} config={config} device={device} ===")
    print("CWD:", os.getcwd())

    # Data + vocab
    train_set, val_set, vocab = build_vocab_and_splits(seed)
    print("Vocab size:", vocab.vocab_size)

    # Loaders
    train_loader, val_loader = make_loaders(
        train_set=train_set,
        val_set=val_set,
        vocab=vocab,
        max_len=config["max_len"],
        batch_size=config["batch_size"],
    )

    # Model
    model = TransformerMCQModel(
        vocab_size=vocab.vocab_size,
        pad_token_id=vocab.pad_id,
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        dropout=config["dropout"],
        pooling=config["pooling"],  # "cls" or "mean"
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

    best = -1.0
    best_path = f"models/scratch_transformer_{config['name']}_seed{seed}.pt"

    try:
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            total = 0.0

            for batch in train_loader:
                x = batch["input_ids"].to(device)
                m = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)

                logits = model(input_ids=x, attention_mask=m)
                loss = F.cross_entropy(logits, y)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item()

            # Validate
            model.eval()
            ps, ys = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["input_ids"].to(device)
                    m = batch["attention_mask"].to(device)
                    y = batch["labels"].to(device)
                    logits = model(input_ids=x, attention_mask=m)
                    ps.append(logits.argmax(-1).cpu())
                    ys.append(y.cpu())

            v = acc(torch.cat(ps), torch.cat(ys))
            print(f"[seed {seed}] epoch {epoch} loss={total/len(train_loader):.4f} val_acc={v:.4f}")

            if v > best:
                best = v
                torch.save(
                    {"model": model.state_dict(), "vocab": vocab.itos, "config": config, "seed": seed},
                    best_path
                )
                print("  saved ->", os.path.abspath(best_path))

    except KeyboardInterrupt:
        print("\n[Interrupted] Caught KeyboardInterrupt. Will proceed using best checkpoint so far if available.\n")

    # If no checkpoint was saved for some reason, save current model anyway
    if not os.path.exists(best_path):
        torch.save(
            {"model": model.state_dict(), "vocab": vocab.itos, "config": config, "seed": seed},
            best_path
        )
        print("  (fallback) saved ->", os.path.abspath(best_path))

    return best, best_path, vocab, device


def write_submission(best_path: str, vocab: TrainOnlyVocab, device: str, out_csv: str, max_len: int):
    """
    Loads best checkpoint and writes submission CSV.
    """
    os.makedirs("submission", exist_ok=True)

    # Load test safely (no eval)
    test_df = load_data(TEST_CSV)

    test_ds = Dataset.from_pandas(test_df)
    test_ds = test_ds.map(lambda b: preprocess_mc_batch_scratch(b, vocab, max_length=max_len), batched=True)
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Rebuild model from checkpoint config (safe)
    ckpt = torch.load(best_path, map_location=device)
    cfg = ckpt.get("config", None)
    if cfg is None:
        raise RuntimeError("Checkpoint missing config. Delete it and re-train.")

    model = TransformerMCQModel(
        vocab_size=vocab.vocab_size,
        pad_token_id=vocab.pad_id,
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"],
        dropout=cfg["dropout"],
        pooling=cfg["pooling"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    preds_all = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input_ids"].to(device)
            m = batch["attention_mask"].to(device)
            logits = model(input_ids=x, attention_mask=m)
            preds_all.extend(logits.argmax(-1).cpu().tolist())

    sub = pd.DataFrame({"id": test_df["id"], "label": preds_all})
    sub.to_csv(out_csv, index=False)
    print("Saved submission ->", os.path.abspath(out_csv))

def run_ablation(configs, seeds):
    all_results = []
    for cfg in configs:
        vals = []
        best_paths = []
        last_vocab = None
        last_device = None

        print(f"\n============================\nCONFIG: {cfg['name']}\n============================")
        for s in seeds:
            best_val, best_path, vocab, device = train_one_run(cfg, s)
            vals.append(best_val)
            best_paths.append(best_path)
            last_vocab = vocab
            last_device = device

        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        print(f"\nRESULT {cfg['name']}: mean={mean:.4f} std={std:.4f} vals={[round(v,4) for v in vals]}")

        all_results.append({
            "name": cfg["name"],
            "config": cfg,
            "mean": mean,
            "std": std,
            "vals": vals,
            "best_paths": best_paths,
            "vocab": last_vocab,
            "device": last_device,
        })

        # Choose best config by mean val
        best_cfg_res = max(all_results, key=lambda r: r["mean"])
        print("\n===============================")
        print("BEST CONFIG BY MEAN VAL:", best_cfg_res["name"])
        print("mean:", best_cfg_res["mean"], "std:", best_cfg_res["std"], "vals:", best_cfg_res["vals"])
        print("===============================\n")

        # Use the best checkpoint from the best-performing SEED run for that config
        # (pick seed index with max val)
        best_seed_idx = int(np.argmax(best_cfg_res["vals"]))
        best_path = best_cfg_res["best_paths"][best_seed_idx]


    return best_cfg_res, best_path

def run_single_config(config, seed=42):
    print(f"\n=== Running config {config['name']} with seed {seed} ===")
    best_val, best_path, vocab, device = train_one_run(config, seed)
    print(f"\nFinished {config['name']}: best_val={best_val:.4f}")
    return best_val, best_path, vocab, device

def main():

    # ------------ DEFINE CONFIGS -------------
    base_cls = {
        "name": "base_cls",
        "num_layers": 4,
        "nhead": 8,
        "dim_ff": 1024,
        "dropout": 0.1,
        "lr": 2e-4,
        "max_len": EMBED_MAX_LEN,
        "pooling": "cls",
        "batch_size": 16,
        "epochs": 1,
    }

    base_mean = {
        "name": "base_mean",
        "num_layers": 4,
        "nhead": 8,
        "dim_ff": 1024,
        "dropout": 0.1,
        "lr": 2e-4,
        "max_len": EMBED_MAX_LEN,
        "pooling": "mean",
        "batch_size": 16,
        "epochs": 1,
    }

    # ------------ CHOOSE MODE -------------
    # MODE = "ablation"
    MODE = "single"   # <-- change this for hyperparameter tuning

    if MODE == "ablation":
        configs = [base_cls, base_mean]
        seeds = [42, 43, 44]
        best_cfg, best_path = run_ablation(configs, seeds)

        # generate submission:
        write_submission(best_path, best_cfg["vocab"], best_cfg["device"],
                         out_csv=f"submission/scratch_transformer_{best_cfg['name']}.csv",
                         max_len=best_cfg["config"]["max_len"])

    elif MODE == "single":

        # hyperparameter tuning config
        tune_cfg = {
        "name": "tune_cfg",
        "num_layers": 4,
        "nhead": 8,
        "dim_ff": 1024,
        "dropout": 0.1,
        "lr": 1e-4,
        "max_len": EMBED_MAX_LEN,
        "pooling": "mean",
        "batch_size": 16,
        "epochs": 10,
        }

        best_val, best_path, vocab, device = run_single_config(tune_cfg, seed=43)

        print(f"\nFinished {tune_cfg['name']} | best_val={best_val:.4f}")

        out_csv = f"submission/{tune_cfg['name']}.csv"
        write_submission(
            best_path=best_path,
            vocab=vocab,
            device=device,
            out_csv=out_csv,
            max_len=tune_cfg["max_len"],
        )

        print(f"\nSaved submission -> {out_csv}\n")


if __name__ == "__main__":
    main()

