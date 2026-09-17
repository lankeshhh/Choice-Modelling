"""Synthetic choice data generator.

Data-generating process
------------------------
Each observation is one customer facing one choice set S (a subset of a
fixed item universe). Item i has attributes (price, quality, category).

Structural utility (the part any item-feature-only model, e.g. MNL or
DeepMNL, could in principle learn):

    V_i = beta_price * price_z_i + beta_quality * quality_z_i + alpha[category_i]

Context effect (the part that requires seeing the whole set): for every
category we construct a triad (A, B, D_A) where A and B are a genuine
trade-off pair (neither dominates the other under V) and D_A is a decoy
that is strictly dominated by A on both price and quality. When D_A is
present in the same choice set as A, A receives a flat utility boost:

    boost_i(S) = decoy_strength   if i == A and D_A in S
                 0                otherwise

Realized utility and choice:

    U_i = V_i + boost_i(S) + eps_i,   eps_i ~ Gumbel(0, 1) iid
    chosen = argmax_{i in S} U_i

Conditional on the realized V + boost for a given set, this is exactly a
softmax/conditional-logit draw (the Gumbel-max trick), so an MNL fit on
the *true* effective utility would be exact. The point is that plain MNL
(and DeepMNL) only ever observe x_i = (price_i, quality_i, category_i) --
they cannot compute boost_i(S) because it depends on which *other* item is
in the set, not on i's own features. That is the structural IIA violation
this benchmark is built to expose. See decisions.md for why this design
was chosen over a nested-logit / correlated-error alternative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SyntheticConfig:
    n_categories: int = 5
    items_per_category: int = 4          # "filler" items per category (excludes the 2 anchors + 1 decoy)
    n_customers: int = 500

    # structural utility
    beta_price: float = -0.12
    beta_quality: float = 1.0
    category_effect_std: float = 0.5

    # item attribute distributions (raw units, before z-scoring)
    price_mean: float = 20.0
    price_std: float = 8.0
    quality_mean: float = 5.0
    quality_std: float = 1.5

    # trade-off pair (A, B) construction: how far apart on price/quality.
    # trade_off_quality_gap is auto-derived so that, ignoring the category
    # effect, A and B have ~equal structural utility (a genuine trade-off,
    # not one item quietly dominating the other).
    trade_off_price_gap: float = 10.0

    # decoy D_A construction: dominated by A on both attributes
    decoy_delta_price: float = 3.0        # D_A costs this much more than A
    decoy_delta_quality: float = 1.0      # D_A is this much worse in quality than A

    # choice set composition
    min_set_size: int = 4
    max_set_size: int = 8
    decoy_fraction: float = 0.3           # fraction of sets built around a decoy triad
    decoy_presence_prob: float = 0.5      # within those, P(D_A actually included)

    seed: int = 0


def _gumbel(size, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(size)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return -np.log(-np.log(u))


def build_item_universe(cfg: SyntheticConfig, rng: np.random.Generator):
    """Build the fixed item universe and the per-category decoy triads.

    Returns
    -------
    items : pd.DataFrame indexed by item_id with columns
        [category, price, quality, price_z, quality_z, role]
        role in {"filler", "target", "competitor", "decoy"}
    triads : dict[int, dict[str, int]]
        category -> {"A": item_id, "B": item_id, "D": item_id}
    """
    quality_gap = cfg.trade_off_price_gap * abs(cfg.beta_price / cfg.beta_quality)

    rows = []
    triads: dict[int, dict[str, int]] = {}
    next_id = 0

    for cat in range(cfg.n_categories):
        # filler items: ordinary random draws
        for _ in range(cfg.items_per_category):
            price = max(1.0, rng.normal(cfg.price_mean, cfg.price_std))
            quality = max(0.1, rng.normal(cfg.quality_mean, cfg.quality_std))
            rows.append(dict(item_id=next_id, category=cat, price=price,
                              quality=quality, role="filler"))
            next_id += 1

        # trade-off pair: A cheap/low-quality, B pricey/high-quality,
        # roughly equal structural utility under V (ignoring alpha[cat])
        price_a = cfg.price_mean - cfg.trade_off_price_gap / 2
        quality_a = cfg.quality_mean - quality_gap / 2
        price_b = cfg.price_mean + cfg.trade_off_price_gap / 2
        quality_b = cfg.quality_mean + quality_gap / 2

        id_a, id_b, id_d = next_id, next_id + 1, next_id + 2
        rows.append(dict(item_id=id_a, category=cat, price=price_a,
                          quality=quality_a, role="target"))
        rows.append(dict(item_id=id_b, category=cat, price=price_b,
                          quality=quality_b, role="competitor"))

        # decoy: dominated by A on both price (higher = worse) and quality (lower = worse)
        price_d = price_a + cfg.decoy_delta_price
        quality_d = max(0.05, quality_a - cfg.decoy_delta_quality)
        rows.append(dict(item_id=id_d, category=cat, price=price_d,
                          quality=quality_d, role="decoy"))

        triads[cat] = {"A": id_a, "B": id_b, "D": id_d}
        next_id += 3

    items = pd.DataFrame(rows).set_index("item_id")
    items["price_z"] = (items["price"] - items["price"].mean()) / items["price"].std()
    items["quality_z"] = (items["quality"] - items["quality"].mean()) / items["quality"].std()

    alpha = rng.normal(0.0, cfg.category_effect_std, size=cfg.n_categories)
    items["category_effect"] = items["category"].map(lambda c: alpha[c])
    items["V"] = (cfg.beta_price * items["price_z"]
                  + cfg.beta_quality * items["quality_z"]
                  + items["category_effect"])

    return items, triads


def generate_dataset(
    cfg: SyntheticConfig,
    decoy_strength: float,
    n_choice_sets: int,
    rng: np.random.Generator,
    items: pd.DataFrame,
    triads: dict,
    id_offset: int = 0,
) -> pd.DataFrame:
    """Simulate n_choice_sets choice sets at a fixed decoy_strength (gamma)."""
    filler_pool = items.index[items["role"] == "filler"].to_numpy()
    non_decoy_pool = items.index[items["role"] != "decoy"].to_numpy()
    categories = list(triads.keys())

    records = []
    for local_set_id in range(n_choice_sets):
        set_id = id_offset + local_set_id
        customer_id = int(rng.integers(0, cfg.n_customers))

        is_decoy_scenario = rng.random() < cfg.decoy_fraction
        if is_decoy_scenario:
            cat = categories[rng.integers(0, len(categories))]
            a_id, b_id, d_id = triads[cat]["A"], triads[cat]["B"], triads[cat]["D"]
            include_decoy = rng.random() < cfg.decoy_presence_prob
            # Filler count is drawn independently of whether the decoy is
            # included, so decoy_control and decoy_treated sets share the
            # same filler distribution and differ by exactly one item (D_A).
            # (Subtracting a fixed core size from a fixed total `size` would
            # instead give decoy_treated sets one fewer filler competitor on
            # average -- a confound that inflates both P(A) and P(B) in
            # decoy_treated for reasons unrelated to the decoy effect.)
            # Base size is capped at max_set_size - 1 so the treated set
            # (base + 1 for D_A) still respects max_set_size.
            base_size = int(rng.integers(cfg.min_set_size, cfg.max_set_size))
            n_filler = base_size - 2
            core = [a_id, b_id] + ([d_id] if include_decoy else [])
            filler = rng.choice(filler_pool, size=min(n_filler, len(filler_pool)),
                                 replace=False)
            set_items = np.array(core + list(filler))
            scenario = "decoy_treated" if include_decoy else "decoy_control"
            active_decoy_target = a_id if include_decoy else None
        else:
            size = int(rng.integers(cfg.min_set_size, cfg.max_set_size + 1))
            set_items = rng.choice(non_decoy_pool, size=min(size, len(non_decoy_pool)),
                                    replace=False)
            scenario = "plain"
            active_decoy_target = None

        sub = items.loc[set_items]
        boost = np.where(sub.index.to_numpy() == active_decoy_target, decoy_strength, 0.0) \
            if active_decoy_target is not None else np.zeros(len(sub))
        true_utility = sub["V"].to_numpy() + boost  # effective utility before noise
        eps = _gumbel(len(sub), rng)
        utility = true_utility + eps
        chosen_pos = int(np.argmax(utility))

        for pos, item_id in enumerate(sub.index):
            records.append(dict(
                customer_id=customer_id,
                choice_set_id=set_id,
                item_id=int(item_id),
                category=int(sub.loc[item_id, "category"]),
                price=float(sub.loc[item_id, "price"]),
                quality=float(sub.loc[item_id, "quality"]),
                price_z=float(sub.loc[item_id, "price_z"]),
                quality_z=float(sub.loc[item_id, "quality_z"]),
                role=sub.loc[item_id, "role"],
                scenario=scenario,
                decoy_strength=decoy_strength,
                context_strength=decoy_strength if scenario == "decoy_treated" else 0.0,
                # true effective utility (V + boost, pre-noise) -- the softmax
                # of this within a set is the exact Bayes-optimal choice
                # distribution, used later to benchmark model NLL against the
                # best any model could theoretically achieve.
                true_utility=float(true_utility[pos]),
                chosen=int(pos == chosen_pos),
            ))

    return pd.DataFrame.from_records(records)


def generate_benchmark(
    cfg: Optional[SyntheticConfig] = None,
    strengths=(0.0, 0.5, 1.0, 2.0),
    n_sets_per_strength: int = 6000,
    seed: Optional[int] = None,
):
    """Generate the full synthetic benchmark: one block per decoy_strength value.

    Returns (df, items, triads).
    """
    cfg = cfg or SyntheticConfig()
    seed = cfg.seed if seed is None else seed
    rng = np.random.default_rng(seed)

    items, triads = build_item_universe(cfg, rng)

    blocks = []
    offset = 0
    for gamma in strengths:
        block = generate_dataset(cfg, gamma, n_sets_per_strength, rng, items, triads,
                                  id_offset=offset)
        blocks.append(block)
        offset += n_sets_per_strength

    df = pd.concat(blocks, ignore_index=True)
    return df, items, triads


def softmax_probs(utilities: np.ndarray) -> np.ndarray:
    u = utilities - utilities.max()
    e = np.exp(u)
    return e / e.sum()


if __name__ == "__main__":
    import pathlib

    df, items, triads = generate_benchmark()
    out_dir = pathlib.Path(__file__).resolve().parents[2] / "data"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "synthetic_benchmark.csv", index=False)
    items.to_csv(out_dir / "item_universe.csv")
    print(f"Wrote {len(df):,} rows across {df['choice_set_id'].nunique():,} choice sets "
          f"to {out_dir}")
    print(df.head(10))
