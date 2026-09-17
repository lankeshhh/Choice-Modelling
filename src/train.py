"""Synthetic-benchmark training/comparison pipeline.

Generates the synthetic benchmark, trains MNL, DeepMNL, and the Set
Transformer on identical splits with the same loss and metrics, evaluates
all three (plus the Bayes-optimal reference), and writes
results/metrics.csv, results/decoy_shifts.csv, and
results/comparison_report.md.

Hyperparameters here match the settings validated and discussed in
decisions.md -- in particular the Set Transformer uses the "extended"
config (epochs=1500, patience=100) confirmed there to reach genuine
convergence (flat validation NLL), not the under-converged first-pass
config from earlier exploration.

Run: python -m src.train
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import pandas as pd
import torch

from src.data.synthetic import SyntheticConfig, generate_benchmark
from src.utils import (
    set_seed, build_padded_tensors, build_true_utility_tensor,
    grouped_split, select_by_ids, feature_dim,
)
from src.models.mnl import fit_mnl_pytorch
from src.models.deep_mnl import fit_deep_mnl
from src.models.set_transformer import fit_set_transformer
from src.evaluate import (
    stratified_metrics_from_logits, decoy_shift_by_strength,
    true_decoy_shift_by_strength, write_comparison_report,
)

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
CHECKPOINT_DIR = pathlib.Path(__file__).resolve().parents[1] / "checkpoints"


def main(seed: int = 0):
    set_seed(seed)
    cfg = SyntheticConfig()

    print("Generating synthetic benchmark...")
    t0 = time.time()
    df, items, triads = generate_benchmark(cfg=cfg, seed=seed)
    print(f"  {len(df):,} rows / {df['choice_set_id'].nunique():,} choice sets "
          f"({time.time() - t0:.1f}s)")

    tensors, meta = build_padded_tensors(df, cfg.n_categories)
    true_u = build_true_utility_tensor(df, tensors[0].shape[1])
    train_ids, val_ids, test_ids = grouped_split(meta, seed=seed)
    train_tensors, _ = select_by_ids(tensors, meta, train_ids)
    val_tensors, _ = select_by_ids(tensors, meta, val_ids)
    test_tensors, test_meta = select_by_ids(tensors, meta, test_ids)
    true_u_test, _ = select_by_ids((true_u, tensors[1], tensors[2]), meta, test_ids)
    dim = feature_dim(cfg.n_categories)
    max_set_size = tensors[0].shape[1]
    X_te, mask_te, y_te = test_tensors
    print(f"  train/val/test sets: {len(train_ids):,}/{len(val_ids):,}/{len(test_ids):,}")

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
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"{name.lower().replace(' ', '_')}.pt")

    print("Evaluating...")
    context_strength = test_meta["context_strength"].to_numpy()

    metrics_frames = [
        stratified_metrics_from_logits("Bayes-optimal", true_u_test[0], y_te, context_strength)
    ]
    for name, model in models.items():
        with torch.no_grad():
            logits = model(X_te, mask_te)
        metrics_frames.append(stratified_metrics_from_logits(name, logits, y_te, context_strength))
    metrics_df = pd.concat(metrics_frames, ignore_index=True)

    decoy_frames = [true_decoy_shift_by_strength(df)]
    for name, model in models.items():
        decoy_frames.append(decoy_shift_by_strength(name, model, df, cfg.n_categories, max_set_size))
    decoy_df = pd.concat(decoy_frames, ignore_index=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    decoy_df.to_csv(RESULTS_DIR / "decoy_shifts.csv", index=False)
    write_comparison_report(metrics_df, decoy_df, RESULTS_DIR / "comparison_report.md", cfg)

    print(f"\nWrote:\n  {RESULTS_DIR / 'metrics.csv'}\n  {RESULTS_DIR / 'decoy_shifts.csv'}\n"
          f"  {RESULTS_DIR / 'comparison_report.md'}\n  {CHECKPOINT_DIR}/*.pt")


if __name__ == "__main__":
    main()
