import numpy as np
import pytest
import torch

from src.utils import (
    set_seed, build_padded_tensors, select_by_ids, feature_dim,
    accuracy, mean_nll, bayes_optimal_nll,
)
from src.models.mnl import (
    fit_mnl_pytorch, fit_mnl_scipy, mnl_nll_and_grad, mnl_hessian,
    mnl_standard_errors, MNL,
)


def test_pytorch_and_scipy_mle_agree(benchmark):
    """Cross-entropy-over-softmax gradient descent and closed-form BFGS are
    solving the same convex MLE problem; they should land on essentially
    the same parameters and the same NLL."""
    tensors, meta = benchmark["tensors"], benchmark["meta"]
    train_ids = benchmark["train_ids"]
    train_tensors, _ = select_by_ids(tensors, meta, train_ids)
    X_tr, mask_tr, y_tr = train_tensors

    # scipy/BFGS closed-form fit
    w_scipy, nll_scipy, result = fit_mnl_scipy(
        X_tr.numpy(), mask_tr.numpy(), y_tr.numpy()
    )
    assert result.success

    # pytorch/Adam fit, run to convergence: patience=None disables early
    # stopping/checkpointing, since train_tensors is standing in for
    # val_tensors here and there's no genuine held-out set to regularize
    # against (see train_choice_model's docstring for why that matters).
    set_seed(0)
    model, _ = fit_mnl_pytorch(
        train_tensors, train_tensors, feature_dim(benchmark["cfg"].n_categories),
        epochs=3000, lr=0.05, patience=None,
    )
    with torch.no_grad():
        logits = model(X_tr, mask_tr)
        nll_pytorch = mean_nll(logits, y_tr)
    w_pytorch = model.linear.weight.detach().numpy().ravel()

    assert nll_pytorch == pytest.approx(nll_scipy, abs=1e-4)
    assert np.allclose(w_pytorch, w_scipy, atol=1e-3)


def test_analytic_gradient_matches_finite_differences(benchmark):
    """Directly check mnl_nll_and_grad's gradient against numerical
    differentiation, independent of any optimizer converging correctly."""
    tensors, meta = benchmark["tensors"], benchmark["meta"]
    train_tensors, _ = select_by_ids(tensors, meta, benchmark["train_ids"])
    X, mask, y = train_tensors
    X, mask, y = X.numpy()[:50], mask.numpy()[:50], y.numpy()[:50]
    dim = X.shape[2]
    rng = np.random.default_rng(0)
    theta = rng.normal(scale=0.1, size=dim)

    nll0, grad = mnl_nll_and_grad(theta, X, mask, y)
    eps = 1e-5
    numeric_grad = np.zeros_like(theta)
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        fp, _ = mnl_nll_and_grad(tp, X, mask, y)
        fm, _ = mnl_nll_and_grad(tm, X, mask, y)
        numeric_grad[i] = (fp - fm) / (2 * eps)

    assert np.allclose(grad, numeric_grad, atol=1e-4)


def test_analytic_hessian_matches_finite_differences(benchmark):
    """Check mnl_hessian (used for standard errors) against numerical
    differentiation of the analytic gradient, the same way the gradient
    itself is checked against finite differences of the NLL."""
    tensors, meta = benchmark["tensors"], benchmark["meta"]
    train_tensors, _ = select_by_ids(tensors, meta, benchmark["train_ids"])
    X, mask, y = train_tensors
    X, mask, y = X.numpy()[:80], mask.numpy()[:80], y.numpy()[:80]
    dim = X.shape[2]
    n_sets = X.shape[0]
    rng = np.random.default_rng(1)
    theta = rng.normal(scale=0.2, size=dim)

    H_analytic = mnl_hessian(theta, X, mask)

    eps = 1e-5
    H_numeric = np.zeros((dim, dim))
    for i in range(dim):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        _, gp = mnl_nll_and_grad(tp, X, mask, y)
        _, gm = mnl_nll_and_grad(tm, X, mask, y)
        # mnl_nll_and_grad's gradient is of the MEAN nll; mnl_hessian is of
        # the SUMMED nll (observed Fisher information), so scale by n_sets
        H_numeric[i] = n_sets * (gp - gm) / (2 * eps)

    assert np.allclose(H_analytic, H_numeric, atol=1e-4)


def test_mnl_near_bayes_optimal_at_zero_context_strength(benchmark):
    """At context_strength == 0 the data is exactly correctly-specified MNL
    (no boost applied anywhere in that slice), so a well-fit MNL's held-out
    NLL should sit close to the Bayes-optimal NLL (the entropy of the true
    generating softmax, estimated from true_utility)."""
    tensors, meta = benchmark["tensors"], benchmark["meta"]
    true_u = benchmark["true_u"]
    train_tensors, _ = select_by_ids(tensors, meta, benchmark["train_ids"])
    val_tensors, _ = select_by_ids(tensors, meta, benchmark["val_ids"])
    test_tensors, test_meta = select_by_ids(tensors, meta, benchmark["test_ids"])

    set_seed(0)
    model, _ = fit_mnl_pytorch(
        train_tensors, val_tensors, feature_dim(benchmark["cfg"].n_categories),
        epochs=500, lr=0.05, patience=25,
    )

    zero_ctx = (test_meta["context_strength"] == 0.0).to_numpy()
    idx = torch.from_numpy(np.nonzero(zero_ctx)[0])
    X_te, mask_te, y_te = test_tensors
    X0, mask0, y0 = X_te[idx], mask_te[idx], y_te[idx]

    # true_u was built over the *full* padded tensor (same row order as
    # `tensors`/`meta`); re-select the same test ids + zero-context rows.
    test_idx_full, _ = select_by_ids((true_u, tensors[1], tensors[2]), meta, benchmark["test_ids"])
    true_u_test = test_idx_full[0]
    true_u0 = true_u_test[idx]

    with torch.no_grad():
        fitted_nll = mean_nll(model(X0, mask0), y0)
    bayes_nll = bayes_optimal_nll(true_u0, y0)

    assert fitted_nll < bayes_nll + 0.03, (
        f"fitted MNL NLL ({fitted_nll:.4f}) too far above Bayes-optimal ({bayes_nll:.4f}) "
        "at context_strength=0, where MNL is correctly specified"
    )


def test_mnl_recovers_true_coefficients_at_zero_context_strength(benchmark):
    """Near-Bayes-optimal NLL doesn't by itself prove the individual
    coefficients are recovered -- a misspecified-but-flexible model could
    hit similar NLL with different weights. Fit MNL on the context_strength
    == 0 slice (the only slice where MNL is correctly specified) and check
    every true parameter (beta_price, beta_quality, and the K-1 relative
    category effects) falls within a wide multiple of its Hessian-based
    standard error -- a real coefficient-recovery check, not just a fit
    quality one. Threshold is |z| < 4 (not the usual 1.96) since this is a
    bug-catching regression test on a fairly small fixture, not a
    from-scratch significance test; see the interactive run in
    decisions.md for the tight, large-sample version (all |z| < 1)."""
    cfg, df, items = benchmark["cfg"], benchmark["df"], benchmark["items"]
    zero_ctx_df = df[df["context_strength"] == 0.0]

    tensors, _ = build_padded_tensors(zero_ctx_df, cfg.n_categories)
    X, mask, y = tensors
    X, mask, y = X.numpy(), mask.numpy(), y.numpy()

    w_hat, nll, result = fit_mnl_scipy(X, mask, y)
    assert result.success
    se, _ = mnl_standard_errors(w_hat, X, mask)

    alpha = items.drop_duplicates("category").sort_values("category")["category_effect"].to_numpy()
    true_rel_alpha = alpha[1:] - alpha[0]
    true_vals = np.concatenate([[cfg.beta_price, cfg.beta_quality], true_rel_alpha])

    z = (w_hat - true_vals) / se
    assert np.all(np.abs(z) < 4), f"coefficient recovery z-scores out of range: {z}"
