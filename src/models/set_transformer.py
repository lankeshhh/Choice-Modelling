"""Set-aware Transformer: self-attention over the items in a choice set.

utility_i = Head(Encoder(embed(x_1), ..., embed(x_n))_i)

Each item is embedded independently, then a small stack of standard
Transformer encoder layers lets every item's representation attend to
every *other* item currently in the set before a linear head turns it
into a scalar logit. This is the one model in this project whose score
for item i can depend on which other items are in S -- exactly the
capability MNL and DeepMNL are structurally missing, and the reason this
benchmark exists at all.

No positional encoding, deliberately: a choice set is an unordered
collection, not a sequence, so the model should be permutation-equivariant
(shuffle the items, the per-item outputs shuffle the same way -- there is
no "item 3" for the model to develop position-specific behavior around).
This is the standard framing for set inputs (cf. Lee et al.'s "Set
Transformer"), and it's directly checked in
test_permutation_equivariance rather than just assumed.

Kept small on purpose (1-2 layers, small embedding dim) -- this is a
portfolio project on a modest synthetic benchmark, not a push for SOTA;
see decisions.md for the capacity choice.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.utils import train_choice_model


class SetTransformer(nn.Module):
    def __init__(self, feature_dim: int, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        # enable_nested_tensor=False: the nested-tensor fast path for
        # padded input is still prototype-stage in PyTorch and warns on
        # every forward call; disabled for reproducibility (this project
        # is small enough that the standard path's speed is a non-issue).
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                              enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """X: (batch, set_size, dim), mask: (batch, set_size), True = real item.

        nn.TransformerEncoder's src_key_padding_mask convention is the
        opposite of ours (True = ignore/padding), hence ~mask below. Every
        real choice set has at least min_set_size items, so no row is ever
        fully masked (which would make the padding-masked softmax inside
        attention undefined).
        """
        h = self.embed(X)
        h = self.encoder(h, src_key_padding_mask=~mask)
        scores = self.head(h).squeeze(-1)
        return scores.masked_fill(~mask, float("-inf"))


def fit_set_transformer(
    train_tensors, val_tensors, feature_dim: int,
    d_model: int = 32, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 64,
    dropout: float = 0.1, epochs: int = 300, lr: float = 0.001, weight_decay: float = 1e-4,
    patience: int = 25, verbose: bool = False,
):
    """Train the Set Transformer with Adam + early stopping on validation
    NLL (see utils.train_choice_model, shared by every model here)."""
    model = SetTransformer(feature_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                            dim_feedforward=dim_feedforward, dropout=dropout)
    return train_choice_model(model, train_tensors, val_tensors, epochs=epochs, lr=lr,
                               weight_decay=weight_decay, patience=patience, verbose=verbose)
