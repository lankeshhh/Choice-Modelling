import numpy as np
import pytest
import torch

from src.utils import (
    set_seed, select_by_ids, feature_dim, accuracy, mean_nll, bayes_optimal_nll,
    predicted_target_share,
)
from src.models.deep_mnl import DeepMNL, fit_deep_mnl


def test_score_is_independent_of_other_items_in_set(benchmark):
    """The literal claim DeepMNL is built to test: an item's score is a
    pure function of its own features, with no path for other items in
    the set to influence it. Checked directly on the forward pass (not
    inferred from training behavior): perturb every item *except* position
    0 and confirm position 0's score is bit-identical."""
    tensors = benchmark["tensors"]
    X, mask, y = tensors
    dim = feature_dim(benchmark["cfg"].n_categories)

    set_seed(0)
    model = DeepMNL(dim)
    model.eval()

    batch = X[:16].clone()
    batch_mask = mask[:16].clone()
    with torch.no_grad():
        scores_before = model(batch, batch_mask)

    perturbed = batch.clone()
    rng = np.random.default_rng(0)
    # overwrite every item's features except position 0 with fresh noise
    # (only where the position is a real, non-padded item)
    noise = torch.from_numpy(rng.normal(size=perturbed[:, 1:, :].shape).astype("float32"))
    perturbed[:, 1:, :] = noise

    with torch.no_grad():
        scores_after = model(perturbed, batch_mask)

    assert torch.equal(scores_before[:, 0], scores_after[:, 0]), (
        "DeepMNL's score for item 0 changed when other items' features changed -- "
        "this should be architecturally impossible"
    )


def test_forward_shapes_and_masking(benchmark):
    tensors = benchmark["tensors"]
    X, mask, y = tensors
    dim = feature_dim(benchmark["cfg"].n_categories)
    model = DeepMNL(dim)
    with torch.no_grad():
        logits = model(X[:32], mask[:32])
    assert logits.shape == (32, X.shape[1])
    assert torch.all(logits[~mask[:32]] == float("-inf"))
    assert torch.all(torch.isfinite(logits[mask[:32]]))


def test_deep_mnl_fits_training_data(benchmark):
    """Smoke test: training should substantially reduce train NLL from a
    random initialization, i.e. the model and training loop actually work
    together (not testing generalization here, just that it learns)."""
    tensors, meta = benchmark["tensors"], benchmark["meta"]
    train_tensors, _ = select_by_ids(tensors, meta, benchmark["train_ids"])
    dim = feature_dim(benchmark["cfg"].n_categories)

    set_seed(0)
    model = DeepMNL(dim)
    X_tr, mask_tr, y_tr = train_tensors
    with torch.no_grad():
        initial_nll = mean_nll(model(X_tr, mask_tr), y_tr)

    set_seed(0)
    trained, history = fit_deep_mnl(train_tensors, train_tensors, dim, epochs=150, patience=None)
    with torch.no_grad():
        final_nll = mean_nll(trained(X_tr, mask_tr), y_tr)

    # Converges quickly and stably to ~0.208 drop regardless of epochs/lr
    # in a reasonable range (checked interactively at 150-1000 epochs,
    # lr 0.01-0.02: final_nll always ~1.524); 0.15 leaves margin below that
    # while still being a real, meaningful bar.
    assert final_nll < initial_nll - 0.15, (
        f"DeepMNL barely learned: initial_nll={initial_nll:.4f}, final_nll={final_nll:.4f}"
    )


def test_deep_mnl_near_bayes_optimal_at_zero_context_strength(benchmark):
    """DeepMNL's utility function is a strict superset of MNL's (an MLP can
    represent a linear function), so on the context_strength == 0 slice
    (where the true utility genuinely is linear-in-features) it should
    also land close to the Bayes-optimal NLL -- extra flexibility isn't
    supposed to hurt when the truth doesn't need it. Looser tolerance than
    MNL's version of this test: more parameters, more variance, and no
    convexity guarantee, so some extra slack above Bayes-optimal is
    expected even from a well-trained model."""
    tensors, meta = benchmark["tensors"], benchmark["meta"]
    true_u = benchmark["true_u"]
    dim = feature_dim(benchmark["cfg"].n_categories)
    train_tensors, _ = select_by_ids(tensors, meta, benchmark["train_ids"])
    val_tensors, _ = select_by_ids(tensors, meta, benchmark["val_ids"])
    test_tensors, test_meta = select_by_ids(tensors, meta, benchmark["test_ids"])

    set_seed(0)
    model, _ = fit_deep_mnl(train_tensors, val_tensors, dim, epochs=300, patience=25)

    zero_ctx = (test_meta["context_strength"] == 0.0).to_numpy()
    idx = torch.from_numpy(np.nonzero(zero_ctx)[0])
    X_te, mask_te, y_te = test_tensors
    X0, mask0, y0 = X_te[idx], mask_te[idx], y_te[idx]

    test_idx_full, _ = select_by_ids((true_u, tensors[1], tensors[2]), meta, benchmark["test_ids"])
    true_u0 = test_idx_full[0][idx]

    with torch.no_grad():
        fitted_nll = mean_nll(model(X0, mask0), y0)
    bayes_nll = bayes_optimal_nll(true_u0, y0)

    assert fitted_nll < bayes_nll + 0.15, (
        f"DeepMNL NLL ({fitted_nll:.4f}) too far above Bayes-optimal ({bayes_nll:.4f}) "
        "at context_strength=0, where the truth is linear and DeepMNL should manage it"
    )


def test_deep_mnl_misses_decoy_effect_like_mnl(benchmark):
    """The point of building DeepMNL: on the decoy-treated slice, it
    should show essentially no learned P(A) shift from the decoy's
    presence, the same structural failure as MNL and for the same reason
    (per-item scoring, no visibility into the rest of the set) -- despite
    being strictly more flexible everywhere else. This checks the outcome
    on real held-out data, complementing the architectural proof in
    test_score_is_independent_of_other_items_in_set."""
    tensors, meta = benchmark["tensors"], benchmark["meta"]
    dim = feature_dim(benchmark["cfg"].n_categories)
    train_tensors, _ = select_by_ids(tensors, meta, benchmark["train_ids"])
    val_tensors, _ = select_by_ids(tensors, meta, benchmark["val_ids"])

    set_seed(0)
    model, _ = fit_deep_mnl(train_tensors, val_tensors, dim, epochs=300, patience=25)

    df = benchmark["df"]
    treated = df[(df["scenario"] == "decoy_treated") & (df["decoy_strength"] == 2.0)]
    control = df[(df["scenario"] == "decoy_control") & (df["decoy_strength"] == 2.0)]
    assert len(treated) > 0 and len(control) > 0

    max_set_size = tensors[0].shape[1]
    n_categories = benchmark["cfg"].n_categories
    p_a_treated = predicted_target_share(model, treated, n_categories, max_set_size)
    p_a_control = predicted_target_share(model, control, n_categories, max_set_size)
    shift = p_a_treated - p_a_control

    # True injected shift at decoy_strength=2.0 is large (empirically an
    # odds-ratio multiplier of several x, see decisions.md). DeepMNL's
    # learned shift should be small in comparison.
    assert abs(shift) < 0.05, (
        f"DeepMNL learned a P(A) shift of {shift:+.4f} from decoy presence -- "
        "expected ~0, since per-item scoring cannot see the decoy"
    )
