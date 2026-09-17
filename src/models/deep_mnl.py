"""DeepMNL: a small MLP utility function, still with no interaction
between items in a choice set.

utility_i = MLP(x_i), same MLP weights shared across items and across the
set, applied independently to each item. This isolates "does nonlinearity
in the utility function help" from "does seeing the rest of the assortment
help" (the Set Transformer's contribution) -- DeepMNL can bend the utility
surface arbitrarily, but it is structurally *just as blind* to set
composition as plain MNL: an item's score here is a pure function of its
own features, nothing else in the set can reach it. See
test_score_is_independent_of_other_items_in_set for a direct check of
that claim (not just an architectural assumption), and
test_deep_mnl_misses_decoy_effect_like_mnl for the point of building this
model at all: it should fail on the context-effect slice the same way MNL
does, for the same structural reason, despite being strictly more
flexible on everything else.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.utils import train_choice_model


class DeepMNL(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """X: (batch, set_size, dim), mask: (batch, set_size) -> masked logits.

        nn.Linear/ReLU apply over the last dimension and broadcast over any
        leading dimensions, so this applies the identical MLP to every
        (batch, item) position independently -- no reshaping needed, and
        no path exists in this computation for one item's features to
        reach another item's score.
        """
        scores = self.mlp(X).squeeze(-1)
        return scores.masked_fill(~mask, float("-inf"))


def fit_deep_mnl(
    train_tensors, val_tensors, feature_dim: int, hidden_dim: int = 32,
    epochs: int = 300, lr: float = 0.01, weight_decay: float = 1e-4,
    patience: int = 20, verbose: bool = False,
):
    """Train DeepMNL with Adam + early stopping on validation NLL (see
    utils.train_choice_model, shared by every model in this project).
    Small weight_decay by default -- DeepMNL has far more parameters than
    MNL and no reason not to regularize a bit given a fairly small
    benchmark."""
    model = DeepMNL(feature_dim, hidden_dim=hidden_dim)
    return train_choice_model(model, train_tensors, val_tensors, epochs=epochs, lr=lr,
                               weight_decay=weight_decay, patience=patience, verbose=verbose)
