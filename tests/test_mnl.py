import numpy as np
import pytest
import torch

from src.data.synthetic import SyntheticConfig, generate_benchmark
from src.utils import (
    set_seed, build_padded_tensors, build_true_utility_tensor, grouped_split,
    select_by_ids, feature_dim, accuracy, mean_nll, bayes_optimal_nll,
)
from src.models.mnl import fit_mnl_pytorch, fit_mnl_scipy, mnl_nll_and_grad, MNL


@pytest.fixture(scope="module")
def benchmark():
    set_seed(0)
    cfg = SyntheticConfig(n_categories=4, items_per_category=4, seed=0)
    df, items, triads = generate_benchmark(
        cfg=cfg, strengths=(0.0, 0.5, 1.0, 2.0), n_sets_per_strength=1500, seed=0,
    )
    tensors, meta = build_padded_tensors(df, cfg.n_categories)
    true_u = build_true_utility_tensor(df, tensors[0].shape[1])
    train_ids, val_ids, test_ids = grouped_split(meta, seed=0)
    return dict(cfg=cfg, df=df, tensors=tensors, meta=meta, true_u=true_u,
                train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)


def _fit_mnl_to_convergence(train_tensors, dim, epochs=3000, lr=0.05):
    """Plain full-batch Adam with no early-stopping bookkeeping.

    fit_mnl_pytorch's "only checkpoint on >1e-5 validation improvement,
    then reload that checkpoint" logic is correct behavior for real
    training (that's what makes early stopping a regularizer), but wrong
    for an MLE-agreement check: it freezes progress the moment improvement
    dips below 1e-5 per epoch, discarding thousands of further epochs of
    real convergence even when there's no actual overfitting risk (no
    honest validation set here). This bug was caught by this exact test:
    the frozen model's category weights disagreed with scipy's BFGS fit by
    up to 0.12 no matter how many total epochs were requested, because the
    returned weights weren't the final ones. A plain convergence loop
    (below) fixes it for this test; fit_mnl_pytorch's early stopping is
    left as-is since it's appropriate for the real train/val harness.
    """
    X_tr, mask_tr, y_tr = train_tensors
    model = MNL(dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(X_tr, mask_tr), y_tr)
        loss.backward()
        optimizer.step()
    return model


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

    # pytorch/Adam fit, run to convergence (see _fit_mnl_to_convergence)
    set_seed(0)
    model = _fit_mnl_to_convergence(train_tensors, feature_dim(benchmark["cfg"].n_categories))
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
