"""Classical Multinomial Logit (MNL).

utility_i = w . x_i, shared weights across items and across the set (no
separate bias term -- see the MNL class docstring for why). Fit two ways
on purpose:

1. `MNL` (torch.nn.Module) + `fit_mnl_pytorch`: gradient descent (Adam) on
   cross-entropy over the choice set. This is what DeepMNL and the Set
   Transformer will also use, so all three models share one training loop.
2. `fit_mnl_scipy`: the same MLE problem solved from its closed form --
   multinomial-logit negative log-likelihood and its analytic gradient,
   optimized with scipy's BFGS. This is the "understand the mechanics"
   piece: conditional-logit MLE has a well-known score equation
   (sum_i p_i * x_i - x_chosen, averaged over sets), implemented directly
   here rather than only via autograd.

Cross-entropy over a masked softmax *is* MNL's log-likelihood, so both
fits are solving the same convex problem and should land on the same
parameters. See tests/test_mnl.py for the agreement check, and
decisions.md for the Bayes-optimal benchmark at decoy_strength=0.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

from src.utils import mean_nll, accuracy


class MNL(nn.Module):
    """No separate bias term. Category is encoded as n_categories - 1
    dummy columns (see utils.featurize), which already avoids the "full
    K-of-K one-hot sums to 1" degeneracy. A bias term would reintroduce an
    equivalent problem on its own: it adds the same constant to every
    item's score regardless of category, and softmax within a choice set
    is invariant to adding a constant to every item in the set -- so a raw
    bias is never identifiable from choice-only data (there is no free
    additive normalization without an outside option of fixed utility).
    Two optimizers minimizing the same NLL could converge to the same
    predictions but different individual bias/weight values if a bias were
    included; dropping it removes that remaining flat direction."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """X: (batch, set_size, dim), mask: (batch, set_size) -> masked logits."""
        scores = self.linear(X).squeeze(-1)
        return scores.masked_fill(~mask, float("-inf"))


def fit_mnl_pytorch(
    train_tensors, val_tensors, feature_dim: int,
    epochs: int = 200, lr: float = 0.05, weight_decay: float = 0.0,
    patience: int = 15, verbose: bool = False,
):
    """Train MNL with Adam + early stopping on validation NLL."""
    X_tr, mask_tr, y_tr = train_tensors
    X_val, mask_val, y_val = val_tensors

    model = MNL(feature_dim)
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

    model.load_state_dict(best_state)
    return model, history


def _masked_logsumexp(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    neg_inf = -np.inf
    masked = np.where(mask, logits, neg_inf)
    row_max = np.max(masked, axis=1, keepdims=True)
    row_max_safe = np.where(np.isfinite(row_max), row_max, 0.0)
    exp_shifted = np.where(mask, np.exp(masked - row_max_safe), 0.0)
    row_sum = exp_shifted.sum(axis=1, keepdims=True)
    return (np.log(row_sum) + row_max_safe).squeeze(1), exp_shifted, row_sum


def mnl_nll_and_grad(theta: np.ndarray, X: np.ndarray, mask: np.ndarray, y: np.ndarray):
    """Analytic negative log-likelihood and gradient for MNL.

    theta = w (dim,) -- no separate bias, see MNL docstring for why.
    X: (n, s, dim), mask: (n, s) bool, y: (n,) chosen index within each set.
    """
    n_sets, set_size, dim = X.shape
    w = theta

    logits = X @ w  # (n_sets, set_size)
    log_row_sum, exp_shifted, row_sum = _masked_logsumexp(logits, mask)
    chosen_logits = logits[np.arange(n_sets), y]
    nll = -(chosen_logits - log_row_sum).mean()

    probs = exp_shifted / row_sum  # softmax, 0 at masked positions
    onehot_y = np.zeros_like(probs)
    onehot_y[np.arange(n_sets), y] = 1.0
    diff = probs - onehot_y  # (n_sets, set_size), 0 at masked positions

    grad_w = np.einsum("ns,nsd->d", diff, X) / n_sets
    return nll, grad_w


def fit_mnl_scipy(X: np.ndarray, mask: np.ndarray, y: np.ndarray):
    """Fit MNL by directly minimizing the closed-form NLL with BFGS."""
    dim = X.shape[2]
    theta0 = np.zeros(dim)
    result = minimize(mnl_nll_and_grad, theta0, args=(X, mask, y), jac=True, method="BFGS")
    w = result.x
    return w, result.fun, result


def mnl_hessian(theta: np.ndarray, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Analytic Hessian of the *summed* (not mean) NLL, i.e. the observed
    Fisher information matrix at theta.

    Closed form for conditional/multinomial logit: for each choice set n
    with softmax probabilities p_n over its items,

        H_n = X_n^T (diag(p_n) - p_n p_n^T) X_n
        H   = sum_n H_n

    This is the same formula used for standard errors in any conditional
    logit / mlogit package. Notably it does not depend on y -- for the
    multinomial-logit (canonical exponential family) case, observed and
    expected Fisher information coincide, so the Hessian is a function of
    theta and the covariates alone.
    """
    logits = X @ theta
    _, exp_shifted, row_sum = _masked_logsumexp(logits, mask)
    p = exp_shifted / row_sum  # (n_sets, set_size), 0 at masked positions

    weighted_x = X * p[..., None]  # p_i * x_i, (n_sets, set_size, dim)
    term1 = np.einsum("nsd,nse->de", weighted_x, X)  # sum_{n,i} p_i x_i x_i^T
    s_n = weighted_x.sum(axis=1)  # (n_sets, dim), = X_n^T p_n per set
    term2 = np.einsum("nd,ne->de", s_n, s_n)  # sum_n outer(s_n, s_n)
    return term1 - term2


def mnl_standard_errors(theta: np.ndarray, X: np.ndarray, mask: np.ndarray):
    """Asymptotic covariance / standard errors of the MLE, from the inverse
    of the observed Fisher information (mnl_hessian, summed-NLL scale)."""
    H = mnl_hessian(theta, X, mask)
    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))
    return se, cov
