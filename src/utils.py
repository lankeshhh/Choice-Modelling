"""Shared data-prep, splitting, and metric utilities used by every model
(MNL, DeepMNL, Set Transformer) so the training/eval harness is identical
across all three.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_dim(n_categories: int) -> int:
    # price_z, quality_z, (n_categories - 1) category dummies. Category 0 is
    # the reference category (all-zero dummy row) -- see featurize() for why
    # a full K-of-K one-hot encoding is *not* used here.
    return 2 + (n_categories - 1)


def featurize(df: pd.DataFrame, n_categories: int) -> np.ndarray:
    """Per-row feature vector: [price_z, quality_z, one-hot(category)],
    using n_categories - 1 dummy columns (category 0 is the reference,
    encoded as all-zeros) rather than the full n_categories one-hot.

    A full K-of-K one-hot block sums to exactly 1 in every row, so shifting
    every category weight by the same constant shifts every item's score by
    that same constant -- and softmax within a choice set is invariant to
    adding a constant to every item in the set. That direction is therefore
    exactly unidentified from choice data (with or without a separate bias
    term), which is really just the standard discrete-choice fact that only
    utility *differences* are identifiable, never absolute levels. Dropping
    one category as a reference removes the degeneracy in the usual way.
    """
    cats = df["category"].to_numpy()
    onehot = np.zeros((len(df), n_categories - 1), dtype=np.float32)
    nonref = cats > 0
    onehot[nonref, cats[nonref] - 1] = 1.0
    base = df[["price_z", "quality_z"]].to_numpy(dtype=np.float32)
    return np.concatenate([base, onehot], axis=1)


def build_padded_tensors(df: pd.DataFrame, n_categories: int, max_set_size: int | None = None):
    """Group a long-format choice dataframe by choice_set_id into padded
    (X, mask, y) tensors.

    X: (n_sets, max_set_size, feature_dim) float32
    mask: (n_sets, max_set_size) bool, True where a real (non-pad) item sits
    y: (n_sets,) int64, index of the chosen item within its set

    Returns ((X, mask, y), meta) where meta is one row per set with
    choice_set_id / scenario / decoy_strength / context_strength, used for
    stratified evaluation.
    """
    groups = df.groupby("choice_set_id", sort=True)
    if max_set_size is None:
        max_set_size = int(groups.size().max())
    dim = feature_dim(n_categories)
    n_sets = groups.ngroups

    X = np.zeros((n_sets, max_set_size, dim), dtype=np.float32)
    mask = np.zeros((n_sets, max_set_size), dtype=bool)
    y = np.zeros(n_sets, dtype=np.int64)
    meta_rows = []

    for i, (set_id, g) in enumerate(groups):
        g = g.reset_index(drop=True)
        n = len(g)
        if n > max_set_size:
            raise ValueError(f"choice_set_id={set_id} has {n} items > max_set_size={max_set_size}")
        X[i, :n] = featurize(g, n_categories)
        mask[i, :n] = True
        y[i] = int(g.index[g["chosen"] == 1][0])
        meta_rows.append(dict(
            choice_set_id=set_id,
            scenario=g["scenario"].iloc[0],
            decoy_strength=float(g["decoy_strength"].iloc[0]),
            context_strength=float(g["context_strength"].iloc[0]),
        ))

    meta = pd.DataFrame(meta_rows)
    tensors = (torch.from_numpy(X), torch.from_numpy(mask), torch.from_numpy(y))
    return tensors, meta


def grouped_split(meta: pd.DataFrame, frac=(0.7, 0.15, 0.15), seed: int = 0):
    """Split choice_set_ids into train/val/test sets of ids, stratified
    within each (decoy_strength, scenario) stratum so every split sees
    every difficulty level and scenario type in roughly equal proportion."""
    assert abs(sum(frac) - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    train_ids, val_ids, test_ids = [], [], []
    for _, grp in meta.groupby(["decoy_strength", "scenario"]):
        ids = grp["choice_set_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(frac[0] * n))
        n_val = int(round(frac[1] * n))
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])
    return set(train_ids), set(val_ids), set(test_ids)


def train_choice_model(
    model: torch.nn.Module, train_tensors, val_tensors,
    epochs: int = 200, lr: float = 0.05, weight_decay: float = 0.0,
    patience: int | None = 15, verbose: bool = False,
):
    """Generic training loop shared by every model: full-batch Adam on
    cross-entropy over the masked choice set. Every model here (MNL,
    DeepMNL, Set Transformer) only differs in its forward(X, mask) ->
    logits -- this loop is identical for all three, which is the point:
    differences in results come from model structure, not from different
    training procedures.

    patience=int (default): early stopping against val_tensors, the
    correct behavior for real training against a genuine held-out
    validation set (it's what makes early stopping a regularizer) --
    checkpoints model state only on >1e-5 validation-NLL improvement, then
    reloads the best checkpoint at the end.

    patience=None: skip early stopping and checkpointing entirely; run all
    `epochs` and return the model's final state as-is. Use this for
    convergence checks / smoke tests where val_tensors isn't a genuine
    held-out set (e.g. the same tensors passed as both train and val, to
    check "does this model+optimizer actually reach the MLE / fit the
    training data"). With early stopping, once per-epoch improvement on
    whatever's passed as "val" drops below 1e-5 -- which can happen well
    before real convergence -- checkpointing freezes and the final reload
    silently discards every later epoch of real progress, regardless of
    how many total epochs were requested. This bit MNL's PyTorch/scipy MLE
    agreement check and DeepMNL's training smoke test before being fixed
    here; see decisions.md.
    """
    X_tr, mask_tr, y_tr = train_tensors
    X_val, mask_val, y_val = val_tensors

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_nll = float("inf")
    best_state = None
    epochs_since_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr, mask_tr)
        loss = torch.nn.functional.cross_entropy(logits, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val, mask_val)
            val_nll = mean_nll(val_logits, y_val)
            val_acc = accuracy(val_logits, y_val)
        history.append(dict(epoch=epoch, train_nll=loss.item(), val_nll=val_nll, val_acc=val_acc))

        if patience is None:
            continue

        if val_nll < best_val_nll - 1e-5:
            best_val_nll = val_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch}, best val_nll={best_val_nll:.4f}")
                break

    if patience is not None:
        model.load_state_dict(best_state)
    return model, history


def select_by_ids(tensors, meta: pd.DataFrame, ids: set):
    X, mask, y = tensors
    keep = meta["choice_set_id"].isin(ids).to_numpy()
    idx = np.nonzero(keep)[0]
    idx_t = torch.from_numpy(idx)
    return (X[idx_t], mask[idx_t], y[idx_t]), meta.iloc[idx].reset_index(drop=True)


def build_true_utility_tensor(df: pd.DataFrame, max_set_size: int) -> torch.Tensor:
    """Pad the DGP's true (pre-noise) effective utility per set. Used only
    for validation: softmax(true_utility) is the exact Bayes-optimal choice
    distribution, so cross-entropy against it is the best expected NLL any
    model could achieve on this data."""
    groups = df.groupby("choice_set_id", sort=True)
    n_sets = groups.ngroups
    U = np.full((n_sets, max_set_size), -np.inf, dtype=np.float32)
    for i, (_, g) in enumerate(groups):
        g = g.reset_index(drop=True)
        U[i, :len(g)] = g["true_utility"].to_numpy(dtype=np.float32)
    return torch.from_numpy(U)


def bayes_optimal_nll(true_utility: torch.Tensor, y: torch.Tensor) -> float:
    """Mean NLL of the true generative softmax distribution at the realized
    choices -- an unbiased estimate (for large N) of the entropy of the
    choice distribution, i.e. the best expected NLL any model could reach."""
    return torch.nn.functional.cross_entropy(true_utility, y, reduction="mean").item()


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def mean_nll(logits: torch.Tensor, y: torch.Tensor) -> float:
    return torch.nn.functional.cross_entropy(logits, y, reduction="mean").item()


def predicted_target_share(model: torch.nn.Module, sub_df: pd.DataFrame,
                            n_categories: int, max_set_size: int) -> float:
    """Mean predicted P(chosen = target) over a set of choice sets, where
    "target" is the role=="target" item (A in the decoy triads -- see
    synthetic.py). Used to measure a fitted model's learned P(A) shift
    from decoy presence (comparing this on decoy_treated vs decoy_control
    subsets), the diagnostic for "did the model actually learn to use
    context, not just fit NLL/accuracy overall."

    build_padded_tensors groups by choice_set_id with sort=True and
    preserves each group's original row order (0..n-1 after reset_index)
    -- target_pos below must (and does) use that same order to find each
    set's target-item position within the padded tensor.
    """
    tens, _ = build_padded_tensors(sub_df, n_categories, max_set_size=max_set_size)
    X, mask, y = tens
    with torch.no_grad():
        probs = torch.softmax(model(X, mask), dim=1)

    def target_pos(g):
        g = g.reset_index(drop=True)
        return g.index[g["role"] == "target"][0]

    role = sub_df.groupby("choice_set_id", sort=True).apply(target_pos).to_numpy().copy()
    return probs[np.arange(len(role)), role].mean().item()
