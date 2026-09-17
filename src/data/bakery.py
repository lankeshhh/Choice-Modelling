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
   with items *not* in that basket, sampled with probability proportional
   to each item's overall popularity^0.75 (standard implicit-feedback
   negative-sampling practice, not uniform noise -- uniform negatives
   would make the discrimination task trivially easy and mostly measure
   "did the model memorize popularity," which isn't the point). This is a
   real methodological approximation: we don't know what a shopper
   actually saw and rejected, only what they didn't buy that trip. Results
   on this data should be read as "does the model comparison hold up on a
   plausible real-world proxy for choice data," not as a rigorous recovery
   test the way the synthetic benchmark is.

2. Items have no attributes beyond an integer ID -- no price, quality, or
   category in the source data. To reuse featurize() unchanged, `category`
   is set to the item's own identity (each item is its own singleton
   category, i.e. a per-item fixed effect via the same K-1 dummy encoding
   used for synthetic categories), and `price_z`/`quality_z` are set to
   0.0 for every row (present so featurize()'s column access doesn't
   break, but genuinely uninformative -- there is no real price or quality
   signal in this data).
"""
from __future__ import annotations

import pathlib
from collections import Counter

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


def build_bakery_dataset(
    path=DEFAULT_PATH,
    n_transactions: int | None = 24000,
    min_set_size: int = 4,
    max_set_size: int = 8,
    seed: int = 0,
):
    """Returns (df, n_items). df matches synthetic.py's schema (minus the
    synthetic-only columns role/true_utility, which nothing here needs):
    choice_set_id, item_id, category, price_z, quality_z, scenario,
    decoy_strength, context_strength, chosen.

    n_transactions=24000 (default) subsamples to match the synthetic
    benchmark's scale -- deliberate, for computational parity and so the
    comparison isn't confounded by "more data helps everyone regardless of
    architecture." Pass None to use all 75,000 transactions.
    """
    rng = np.random.default_rng(seed)
    baskets = load_raw_baskets(path)

    all_items = sorted({item for basket in baskets for item in basket})
    n_items = len(all_items)
    if all_items != list(range(1, n_items + 1)):
        raise ValueError("expected contiguous 1..n_items item ids in bakery.txt")

    item_counts = Counter(item for basket in baskets for item in basket)
    freqs = np.array([item_counts[i] for i in range(1, n_items + 1)], dtype=float)
    sampling_weights = freqs ** 0.75  # standard unigram^0.75 negative-sampling dampening

    if n_transactions is not None and n_transactions < len(baskets):
        keep = np.sort(rng.choice(len(baskets), size=n_transactions, replace=False))
        baskets = [baskets[i] for i in keep]

    records = []
    for choice_set_id, basket in enumerate(baskets):
        chosen_item = basket[rng.integers(0, len(basket))]
        size = int(rng.integers(min_set_size, max_set_size + 1))
        n_negatives = max(0, size - 1)

        basket_set = set(basket)
        not_in_basket = np.array([it not in basket_set for it in range(1, n_items + 1)])
        avail_items = np.arange(1, n_items + 1)[not_in_basket]
        avail_weights = sampling_weights[not_in_basket]
        avail_weights = avail_weights / avail_weights.sum()

        n_negatives = min(n_negatives, len(avail_items))
        negatives = rng.choice(avail_items, size=n_negatives, replace=False, p=avail_weights)

        set_items = [int(chosen_item)] + [int(x) for x in negatives]
        rng.shuffle(set_items)
        chosen_pos = set_items.index(int(chosen_item))

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
