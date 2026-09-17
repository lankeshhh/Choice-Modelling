import numpy as np
import pytest
import torch

from src.utils import (
    set_seed, select_by_ids, feature_dim, mean_nll, bayes_optimal_nll,
)
from src.models.set_transformer import SetTransformer, fit_set_transformer


def test_padded_positions_dont_affect_valid_items(benchmark):
    """Masking correctness: src_key_padding_mask must actually exclude
    padded slots from attention. Perturb ONLY padded feature values and
    confirm scores at real items are unchanged. (If the mask polarity were
    inverted -- an easy mistake, since PyTorch's convention (True=ignore)
    is the opposite of ours (True=valid) -- this would fail immediately.)"""
    tensors = benchmark["tensors"]
    X, mask, y = tensors
    dim = feature_dim(benchmark["cfg"].n_categories)

    set_seed(1)
    model = SetTransformer(dim)
    model.eval()

    batch, batch_mask = X[:16].clone(), mask[:16].clone()
    with torch.no_grad():
        scores_before = model(batch, batch_mask)

    perturbed = batch.clone()
    rng = np.random.default_rng(0)
    noise = torch.from_numpy(rng.normal(size=perturbed.shape).astype("float32"))
    perturbed[~batch_mask] = noise[~batch_mask]
    with torch.no_grad():
        scores_after = model(perturbed, batch_mask)

    assert torch.allclose(scores_before[batch_mask], scores_after[batch_mask], atol=1e-6)


def test_score_depends_on_other_items_in_set(benchmark):
    """The contrast to DeepMNL's independence test: item 0's score SHOULD
    change when other real items in the set change -- this is the whole
    point of the architecture. Checked at initialization (no training
    needed): a nontrivial function of the whole set should already show
    this, since nothing in the architecture special-cases it away."""
    tensors = benchmark["tensors"]
    X, mask, y = tensors
    dim = feature_dim(benchmark["cfg"].n_categories)

    set_seed(1)
    model = SetTransformer(dim)
    model.eval()

    batch, batch_mask = X[:16].clone(), mask[:16].clone()
    with torch.no_grad():
        scores_before = model(batch, batch_mask)

    perturbed = batch.clone()
    rng = np.random.default_rng(0)
    noise = torch.from_numpy(rng.normal(size=perturbed[:, 1:, :].shape).astype("float32"))
    perturbed[:, 1:, :] = noise
    with torch.no_grad():
        scores_after = model(perturbed, batch_mask)

    changed = (scores_before[:, 0] - scores_after[:, 0]).abs() > 1e-4
    assert changed.any(), "item 0's score never changed when other items changed"


def test_permutation_equivariance(benchmark):
    """No positional encoding is used, so shuffling item order within a
    set should shuffle the per-item outputs the same way and nothing
    else -- a choice set is unordered, and the model should treat it that
    way rather than learning position-specific behavior."""
    tensors = benchmark["tensors"]
    X, mask, y = tensors
    dim = feature_dim(benchmark["cfg"].n_categories)

    set_seed(1)
    model = SetTransformer(dim)
    model.eval()

    batch, batch_mask = X[:16].clone(), mask[:16].clone()
    with torch.no_grad():
        scores_before = model(batch, batch_mask)

    perm = torch.randperm(batch.shape[1])
    inv_perm = torch.argsort(perm)
    with torch.no_grad():
        scores_perm = model(batch[:, perm, :], batch_mask[:, perm])
    scores_unpermuted = scores_perm[:, inv_perm]

    assert torch.allclose(scores_before[batch_mask], scores_unpermuted[batch_mask], atol=1e-5)
    assert torch.isinf(scores_unpermuted[~batch_mask]).all()


def test_forward_shapes_and_masking(benchmark):
    tensors = benchmark["tensors"]
    X, mask, y = tensors
    dim = feature_dim(benchmark["cfg"].n_categories)
    model = SetTransformer(dim)
    with torch.no_grad():
        logits = model(X[:32], mask[:32])
    assert logits.shape == (32, X.shape[1])
    assert torch.all(logits[~mask[:32]] == float("-inf"))
    assert torch.all(torch.isfinite(logits[mask[:32]]))


def test_set_transformer_fits_training_data(benchmark):
    """Smoke test: training should substantially reduce train NLL from
    random init (patience=None -- see decisions.md/utils.train_choice_model
    for why real early stopping isn't appropriate when the training set
    stands in as its own validation set)."""
    tensors, meta = benchmark["tensors"], benchmark["meta"]
    train_tensors, _ = select_by_ids(tensors, meta, benchmark["train_ids"])
    dim = feature_dim(benchmark["cfg"].n_categories)
    X_tr, mask_tr, y_tr = train_tensors

    set_seed(0)
    model = SetTransformer(dim)
    with torch.no_grad():
        initial_nll = mean_nll(model(X_tr, mask_tr), y_tr)

    set_seed(0)
    trained, history = fit_set_transformer(train_tensors, train_tensors, dim,
                                            epochs=200, patience=None)
    with torch.no_grad():
        final_nll = mean_nll(trained(X_tr, mask_tr), y_tr)

    # Observed drop with these fixed seeds is ~0.13 (deterministic, not a
    # lucky draw); 0.08 leaves real margin while still catching an actual
    # training regression.
    assert final_nll < initial_nll - 0.08, (
        f"Set Transformer barely learned: initial_nll={initial_nll:.4f}, final_nll={final_nll:.4f}"
    )


def test_set_transformer_near_bayes_optimal_at_zero_context_strength(benchmark):
    """Like DeepMNL, the Transformer's function class is a strict superset
    of MNL's (it can ignore the other items and behave linearly if that's
    optimal), so on the context_strength == 0 slice it should also land
    reasonably close to Bayes-optimal. Loosest tolerance of the three
    models here -- most parameters, least convexity, most to learn to
    "turn off" (ignoring genuinely irrelevant other items) when it isn't
    needed."""
    tensors, meta = benchmark["tensors"], benchmark["meta"]
    true_u = benchmark["true_u"]
    dim = feature_dim(benchmark["cfg"].n_categories)
    train_tensors, _ = select_by_ids(tensors, meta, benchmark["train_ids"])
    val_tensors, _ = select_by_ids(tensors, meta, benchmark["val_ids"])
    test_tensors, test_meta = select_by_ids(tensors, meta, benchmark["test_ids"])

    set_seed(0)
    model, _ = fit_set_transformer(train_tensors, val_tensors, dim, epochs=300, patience=25)

    zero_ctx = (test_meta["context_strength"] == 0.0).to_numpy()
    idx = torch.from_numpy(np.nonzero(zero_ctx)[0])
    X_te, mask_te, y_te = test_tensors
    X0, mask0, y0 = X_te[idx], mask_te[idx], y_te[idx]

    test_idx_full, _ = select_by_ids((true_u, tensors[1], tensors[2]), meta, benchmark["test_ids"])
    true_u0 = test_idx_full[0][idx]

    with torch.no_grad():
        fitted_nll = mean_nll(model(X0, mask0), y0)
    bayes_nll = bayes_optimal_nll(true_u0, y0)

    # Observed gap with these fixed seeds is ~0.005 -- the Transformer
    # matches MNL/DeepMNL's closeness to Bayes-optimal here, not just "not
    # terrible." 0.05 leaves a real margin (~10x) while still being a
    # meaningful bound, not the essentially-unfalsifiable 0.3 first guessed.
    assert fitted_nll < bayes_nll + 0.05, (
        f"Set Transformer NLL ({fitted_nll:.4f}) too far above Bayes-optimal ({bayes_nll:.4f}) "
        "at context_strength=0"
    )
