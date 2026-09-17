"""Real-data validation pass: MNL, DeepMNL, and the Set Transformer on the
Bakery basket dataset (Benson, Kumar & Tomkins, WSDM 2018).

Reuses the exact same shared harness as the synthetic experiment
(src/train.py) -- featurize, build_padded_tensors, grouped_split,
train_choice_model, and the fit_* functions -- unchanged. Only the data
loading differs (src/data/bakery.py). No Bayes-optimal reference (no known
true utility for real data) and no decoy-shift diagnostic (no injected
effect to check recovery of) -- just NLL and accuracy for the three
models, appended as a new section in results/comparison_report.md rather
than replacing the synthetic results there.

Run: python -m src.train_bakery
"""
from __future__ import annotations

import pathlib
import time

import pandas as pd
import torch

from src.data.bakery import build_bakery_dataset
from src.utils import set_seed, build_padded_tensors, grouped_split, select_by_ids, feature_dim
from src.models.mnl import fit_mnl_pytorch
from src.models.deep_mnl import fit_deep_mnl
from src.models.set_transformer import fit_set_transformer
from src.evaluate import (
    stratified_metrics_from_logits, write_bakery_section, compute_composition_leakage_rate,
)

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
CHECKPOINT_DIR = pathlib.Path(__file__).resolve().parents[1] / "checkpoints"


def main(seed: int = 0, n_transactions: int = 24000, negative_sampling: str = "substitute"):
    set_seed(seed)

    print(f"Loading Bakery dataset (negative_sampling={negative_sampling!r})...")
    t0 = time.time()
    df, n_items = build_bakery_dataset(n_transactions=n_transactions, seed=seed,
                                        negative_sampling=negative_sampling)
    print(f"  {len(df):,} rows / {df['choice_set_id'].nunique():,} choice sets / "
          f"{n_items} unique items ({time.time() - t0:.1f}s)")

    tensors, meta = build_padded_tensors(df, n_items)
    train_ids, val_ids, test_ids = grouped_split(meta, seed=seed)
    train_tensors, _ = select_by_ids(tensors, meta, train_ids)
    val_tensors, _ = select_by_ids(tensors, meta, val_ids)
    test_tensors, test_meta = select_by_ids(tensors, meta, test_ids)
    dim = feature_dim(n_items)
    X_te, mask_te, y_te = test_tensors
    print(f"  train/val/test sets: {len(train_ids):,}/{len(val_ids):,}/{len(test_ids):,}, "
          f"feature_dim={dim}")

    models = {}

    print("Training MNL...")
    t0 = time.time()
    set_seed(seed)
    models["MNL"], _ = fit_mnl_pytorch(train_tensors, val_tensors, dim, epochs=500, lr=0.05, patience=25)
    print(f"  done ({time.time() - t0:.1f}s)")

    print("Training DeepMNL...")
    t0 = time.time()
    set_seed(seed)
    models["DeepMNL"], _ = fit_deep_mnl(train_tensors, val_tensors, dim, epochs=300, patience=25)
    print(f"  done ({time.time() - t0:.1f}s)")

    print("Training Set Transformer (this takes several minutes)...")
    t0 = time.time()
    set_seed(seed)
    models["Set Transformer"], _ = fit_set_transformer(
        train_tensors, val_tensors, dim, epochs=1500, lr=0.001, patience=100,
    )
    print(f"  done ({time.time() - t0:.1f}s)")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    for name, model in models.items():
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"bakery_{name.lower().replace(' ', '_')}.pt")

    print("Evaluating...")
    # constant context_strength=0.0 for all rows (no injected effect concept
    # for real data) -- stratified_metrics_from_logits degrades gracefully
    # to a single stratum per model, which is exactly the right behavior here.
    context_strength = test_meta["context_strength"].to_numpy()
    frames = []
    for name, model in models.items():
        with torch.no_grad():
            logits = model(X_te, mask_te)
        frames.append(stratified_metrics_from_logits(name, logits, y_te, context_strength))
    metrics_df = pd.concat(frames, ignore_index=True).drop(columns=["context_strength"])

    print("Checking for choice-set composition leakage between train and test...")
    leakage_rate = compute_composition_leakage_rate(df, train_ids, test_ids)
    print(f"  {leakage_rate*100:.1f}% of test compositions also appear in training "
          f"(see decisions.md for why this is checked and whether it matters)")

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_df.to_csv(RESULTS_DIR / "bakery_metrics.csv", index=False)
    write_bakery_section(metrics_df, RESULTS_DIR / "comparison_report.md",
                          n_items=n_items, n_sets=df["choice_set_id"].nunique(),
                          negative_sampling=negative_sampling, leakage_rate=leakage_rate)

    print(f"\nWrote:\n  {RESULTS_DIR / 'bakery_metrics.csv'}\n"
          f"  {RESULTS_DIR / 'comparison_report.md'} (Bakery section)\n  {CHECKPOINT_DIR}/bakery_*.pt")
    print("\n", metrics_df.sort_values("model").to_string(index=False))


if __name__ == "__main__":
    main()
