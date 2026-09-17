"""Bakery real-data loader.

Converts the raw subset-selection ("basket") data from Benson, Kumar &
Tomkins (WSDM 2018), "A Discrete Choice Model for Subset Selection" --
vendored at src/data/external/bakery.txt -- into the same long-format
schema src/data/synthetic.py produces (one row per item per choice set,
with a `chosen` 0/1 column), so the existing harness (featurize,
build_padded_tensors, grouped_split, train_choice_model) needs no
changes -- this loader is the only new code for the real-data pass.

Two real structural gaps from the synthetic setting, handled explicitly
rather than papered over (see decisions.md for the full reasoning):

1. bakery.txt is *subset-selection* data: each line is a whole basket of
   items purchased together, not "an assortment was shown, one item was
   chosen." There is no assortment in the raw data at all. This loader
   constructs one: for each basket, one item is picked (uniformly at
   random) as the "chosen" item, and the rest of the choice set is filled
   with items *not* in that basket. This is a real methodological
   approximation: we don't know what a shopper actually saw and rejected,
   only what they didn't buy that trip. Results on this data should be
   read as "does the model comparison hold up on a plausible real-world
   proxy for choice data," not as a rigorous recovery test the way the
   synthetic benchmark is.

2. Items have no attributes beyond an integer ID -- no price, quality, or
   category in the source data. To reuse featurize() unchanged, `category`
   is set to the item's own identity (each item is its own singleton
   category, i.e. a per-item fixed effect via the same K-1 dummy encoding
   used for synthetic categories), and `price_z`/`quality_z` are set to
   0.0 for every row (present so featurize()'s column access doesn't
   break, but genuinely uninformative -- there is no real price or quality
   signal in this data).

Negative sampling: two schemes, selected via `negative_sampling`.

- "popularity" (the original scheme): negatives drawn with probability
  proportional to each item's overall popularity^0.75 (standard
  implicit-feedback negative-sampling dampening). Simple, but negatives
  are independent of which item was chosen -- there is no substitution
  structure for a set-aware model to exploit, since a negative's identity
  carries no information beyond population-level popularity, which a
  per-item fixed effect already captures completely. See decisions.md for
  why the first real-data pass with this scheme found MNL, DeepMNL, and
  the Set Transformer statistically indistinguishable, and why that's an
  artifact of the sampling scheme, not evidence the Transformer's
  synthetic-data advantage doesn't transfer to real data.

- "substitute" (the default): negatives drawn from the chosen item's
  `substitute_top_k` nearest neighbors by co-occurrence-*profile*
  similarity -- cosine similarity between items' co-occurrence vectors
  (which OTHER items each tends to appear alongside), NOT their direct
  co-occurrence with each other. Direct co-occurrence measures
  complementarity (bought together, e.g. bread and butter); profile
  similarity measures substitutability (playing a similar role across
  different baskets, e.g. two bread varieties that are rarely bought
  together precisely because a shopper picks one or the other, but both
  tend to appear alongside the same other items). Verified this
  distinction empirically before relying on it (see decisions.md): profile
  similarity and direct co-occurrence are essentially uncorrelated
  (r=0.015) across all item pairs in this data, and a popular item's
  top-5 by each measure overlap in only 2 of 5 items.
"""
from __future__ import annotations

import pathlib
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

DEFAULT_PATH = pathlib.Path(__file__).resolve().parent / "external" / "bakery.txt"


def load_raw_baskets(path=DEFAULT_PATH) -> list[list[int]]:
    baskets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                baskets.append([int(tok) for tok in line.split()])
    return baskets


def compute_cooccurrence_and_substitute_similarity(baskets: list[list[int]], n_items: int):
    """Returns (cooc, sim), both (n_items, n_items) with row/col i
    representing item i+1 (0-indexed arrays, 1-indexed item ids).

    cooc[i, j] = number of baskets containing both item i+1 and item j+1
    (direct co-occurrence -- a complementarity signal).

    sim[i, j] = cosine similarity between item i+1's and item j+1's
    co-occurrence *profiles* (their full cooc row vectors, diagonal
    excluded) -- a substitutability proxy, deliberately NOT the same
    thing as cooc[i, j] itself. Two items can have high profile
    similarity (they tend to co-occur with the same other items) while
    rarely or never appearing in the same basket together.
    """
    cooc = np.zeros((n_items, n_items))
    for basket in baskets:
        unique_items = sorted(set(basket))
        for a, b in combinations(unique_items, 2):
            cooc[a - 1, b - 1] += 1
            cooc[b - 1, a - 1] += 1

    norms = np.linalg.norm(cooc, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = cooc / norms
    sim = normed @ normed.T
    np.fill_diagonal(sim, 0.0)
    return cooc, sim


def _sample_negatives(
    rng, chosen_item: int, basket_set: set, n_negatives: int, n_items: int,
    method: str, sampling_weights=None, sim_matrix=None, substitute_top_k: int = 10,
):
    not_in_basket = np.array([it not in basket_set for it in range(1, n_items + 1)])

    if method == "popularity":
        avail_items = np.arange(1, n_items + 1)[not_in_basket]
        avail_weights = sampling_weights[not_in_basket]
        avail_weights = avail_weights / avail_weights.sum()
        n_neg = min(n_negatives, len(avail_items))
        return rng.choice(avail_items, size=n_neg, replace=False, p=avail_weights)

    elif method == "substitute":
        row = sim_matrix[chosen_item - 1].copy()
        row[~not_in_basket] = -np.inf  # basket items can never be sampled as negatives
        order = np.argsort(-row)  # descending similarity; excluded items sort last
        pool = order[:substitute_top_k] + 1  # 0-indexed position -> 1-indexed item id
        n_neg = min(n_negatives, len(pool))
        return rng.choice(pool, size=n_neg, replace=False)

    raise ValueError(f"unknown negative_sampling method: {method!r}")


def build_bakery_dataset(
    path=DEFAULT_PATH,
    n_transactions: int | None = 24000,
    min_set_size: int = 4,
    max_set_size: int = 8,
    seed: int = 0,
    negative_sampling: str = "substitute",
    substitute_top_k: int = 10,
):
    """Returns (df, n_items). df matches synthetic.py's schema (minus the
    synthetic-only columns role/true_utility, which nothing here needs):
    choice_set_id, item_id, category, price_z, quality_z, scenario,
    decoy_strength, context_strength, chosen.

    n_transactions=24000 (default) subsamples to match the synthetic
    benchmark's scale -- deliberate, for computational parity and so the
    comparison isn't confounded by "more data helps everyone regardless of
    architecture." Pass None to use all 75,000 transactions.

    Co-occurrence/similarity statistics (needed for negative_sampling=
    "substitute") are always computed from the FULL 75,000-transaction
    dataset, not the (possibly subsampled) training set -- more data gives
    a more reliable similarity estimate, and there's no reason to throw
    that away just because the constructed *training examples* are
    subsampled for compute-budget reasons.
    """
    rng = np.random.default_rng(seed)
    all_baskets = load_raw_baskets(path)

    all_items = sorted({item for basket in all_baskets for item in basket})
    n_items = len(all_items)
    if all_items != list(range(1, n_items + 1)):
        raise ValueError("expected contiguous 1..n_items item ids in bakery.txt")

    sampling_weights = None
    sim_matrix = None
    if negative_sampling == "popularity":
        item_counts = Counter(item for basket in all_baskets for item in basket)
        freqs = np.array([item_counts[i] for i in range(1, n_items + 1)], dtype=float)
        sampling_weights = freqs ** 0.75
    elif negative_sampling == "substitute":
        _, sim_matrix = compute_cooccurrence_and_substitute_similarity(all_baskets, n_items)
    else:
        raise ValueError(f"unknown negative_sampling method: {negative_sampling!r}")

    baskets = all_baskets
    if n_transactions is not None and n_transactions < len(baskets):
        keep = np.sort(rng.choice(len(baskets), size=n_transactions, replace=False))
        baskets = [baskets[i] for i in keep]

    records = []
    for choice_set_id, basket in enumerate(baskets):
        chosen_item = int(basket[rng.integers(0, len(basket))])
        size = int(rng.integers(min_set_size, max_set_size + 1))
        n_negatives = max(0, size - 1)
        basket_set = set(basket)

        negatives = _sample_negatives(
            rng, chosen_item, basket_set, n_negatives, n_items,
            method=negative_sampling, sampling_weights=sampling_weights,
            sim_matrix=sim_matrix, substitute_top_k=substitute_top_k,
        )

        set_items = [chosen_item] + [int(x) for x in negatives]
        rng.shuffle(set_items)
        chosen_pos = set_items.index(chosen_item)

        for pos, item in enumerate(set_items):
            records.append(dict(
                choice_set_id=choice_set_id,
                item_id=item,
                category=item - 1,
                price_z=0.0,
                quality_z=0.0,
                scenario="bakery",
                decoy_strength=0.0,
                context_strength=0.0,
                chosen=int(pos == chosen_pos),
            ))

    df = pd.DataFrame.from_records(records)
    return df, n_items


if __name__ == "__main__":
    df, n_items = build_bakery_dataset()
    out_dir = pathlib.Path(__file__).resolve().parents[2] / "data"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "bakery_choices.csv", index=False)
    print(f"Wrote {len(df):,} rows across {df['choice_set_id'].nunique():,} choice sets "
          f"({n_items} unique items) to {out_dir}")
    print(df.head(10))
