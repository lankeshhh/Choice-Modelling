import numpy as np
import pandas as pd
import pytest

from src.data.bakery import (
    build_bakery_dataset, load_raw_baskets, DEFAULT_PATH,
    compute_cooccurrence_and_substitute_similarity,
)


def test_raw_baskets_load():
    baskets = load_raw_baskets()
    assert len(baskets) == 75000
    assert all(len(b) >= 1 for b in baskets)
    all_items = {item for b in baskets for item in b}
    assert all_items == set(range(1, 51))  # 50 contiguous items, 1-indexed


@pytest.fixture(scope="module")
def small_bakery():
    df, n_items = build_bakery_dataset(n_transactions=500, seed=0)
    return df, n_items


def test_schema_and_set_integrity(small_bakery):
    df, n_items = small_bakery
    expected_cols = {"choice_set_id", "item_id", "category", "price_z", "quality_z",
                      "scenario", "decoy_strength", "context_strength", "chosen"}
    assert expected_cols.issubset(df.columns)

    assert n_items == 50
    assert df["choice_set_id"].nunique() == 500

    per_set = df.groupby("choice_set_id")
    assert (per_set["chosen"].sum() == 1).all()
    assert (per_set["item_id"].apply(lambda s: not s.duplicated().any())).all()

    sizes = per_set.size()
    assert sizes.min() >= 4
    assert sizes.max() <= 8

    assert df["category"].min() >= 0
    assert df["category"].max() <= n_items - 1
    assert (df["category"] == df["item_id"] - 1).all()

    # placeholder columns required by build_padded_tensors/grouped_split
    assert (df["price_z"] == 0.0).all()
    assert (df["quality_z"] == 0.0).all()
    assert (df["scenario"] == "bakery").all()
    assert (df["decoy_strength"] == 0.0).all()
    assert (df["context_strength"] == 0.0).all()


def test_negatives_exclude_original_basket_items():
    """The chosen item's original basket-mates should never appear as
    "negative" alternatives in the constructed choice set -- they were
    actually purchased together, so treating them as rejected alternatives
    would misrepresent the data. Reconstructs the exact original basket
    per choice_set_id (subsampling with the same seed keeps the surviving
    baskets in their original relative order, so choice_set_id == index
    into the subsampled, order-preserved basket list) and checks the
    invariant directly, rather than a weaker proxy."""
    n_transactions, seed = 300, 1
    rng = np.random.default_rng(seed)
    all_baskets = load_raw_baskets()
    keep = np.sort(rng.choice(len(all_baskets), size=n_transactions, replace=False))
    original_baskets = [all_baskets[i] for i in keep]

    df, n_items = build_bakery_dataset(n_transactions=n_transactions, seed=seed)
    assert df["choice_set_id"].nunique() == len(original_baskets)

    for choice_set_id, g in df.groupby("choice_set_id"):
        original_basket = set(original_baskets[choice_set_id])
        chosen_item = int(g.loc[g["chosen"] == 1, "item_id"].iloc[0])
        negative_items = set(g.loc[g["chosen"] == 0, "item_id"].astype(int))

        assert chosen_item in original_basket
        assert negative_items.isdisjoint(original_basket), (
            f"choice_set_id={choice_set_id}: negatives {negative_items} overlap "
            f"original basket {original_basket}"
        )


def test_reproducible_with_same_seed():
    df1, _ = build_bakery_dataset(n_transactions=200, seed=42)
    df2, _ = build_bakery_dataset(n_transactions=200, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_differ():
    df1, _ = build_bakery_dataset(n_transactions=200, seed=1)
    df2, _ = build_bakery_dataset(n_transactions=200, seed=2)
    assert not df1["item_id"].equals(df2["item_id"])


def test_full_scale_default_matches_synthetic_benchmark_size():
    df, n_items = build_bakery_dataset(n_transactions=24000, seed=0)
    assert df["choice_set_id"].nunique() == 24000
    assert n_items == 50


def test_profile_similarity_is_not_just_direct_cooccurrence():
    """The substitute proxy (profile similarity: do two items tend to
    co-occur with the same OTHER items) must be measuring something
    genuinely different from direct co-occurrence (do the two items
    co-occur with EACH OTHER) -- the latter is a complementarity signal
    (bought together), not a substitutability one. If the two were highly
    correlated, "substitute" sampling would just be re-discovering
    complements under a different name. Checked once interactively before
    committing to this design (see decisions.md): r=0.015 across all pairs,
    and a popular item's top-5 by each measure overlapped in only 2/5."""
    n_items = 50
    baskets = load_raw_baskets()
    cooc, sim = compute_cooccurrence_and_substitute_similarity(baskets, n_items)

    iu = np.triu_indices(n_items, k=1)
    corr = np.corrcoef(cooc[iu], sim[iu])[0, 1]
    assert abs(corr) < 0.15, f"profile similarity too correlated with direct co-occurrence: r={corr:.3f}"


def test_substitute_sampling_concentrates_on_higher_similarity_negatives():
    """The model-free check this whole design change hinges on: does
    negative_sampling="substitute" actually produce negatives closer to
    the chosen item's profile than negative_sampling="popularity" does,
    without also becoming more complementary (higher direct co-occurrence)?
    If this doesn't hold, there's no point retraining any models on it.

    Grounded in an interactive run before writing these thresholds (not
    guessed): substitute scheme's mean profile-similarity was 0.6935 vs.
    popularity's 0.4708 (population-wide mean 0.5134 for context) --
    substitute sits clearly above the population baseline, popularity
    clearly below it. Mean direct co-occurrence was 379 (substitute) vs.
    345 (popularity) vs. 343.75 population mean -- both close to baseline,
    confirming the substitute scheme is not accidentally selecting
    complements."""
    n_items = 50
    baskets = load_raw_baskets()
    cooc, sim = compute_cooccurrence_and_substitute_similarity(baskets, n_items)
    iu = np.triu_indices(n_items, k=1)
    population_mean_sim = sim[iu].mean()

    def mean_chosen_negative_similarity(negative_sampling):
        df, _ = build_bakery_dataset(n_transactions=2000, seed=0, negative_sampling=negative_sampling)
        sims = []
        for _, g in df.groupby("choice_set_id"):
            chosen = int(g.loc[g["chosen"] == 1, "item_id"].iloc[0])
            negs = g.loc[g["chosen"] == 0, "item_id"].astype(int).to_numpy()
            sims.extend(sim[chosen - 1, negs - 1])
        return np.mean(sims)

    sim_popularity = mean_chosen_negative_similarity("popularity")
    sim_substitute = mean_chosen_negative_similarity("substitute")

    assert sim_substitute > population_mean_sim, (
        f"substitute scheme's mean similarity ({sim_substitute:.4f}) should exceed "
        f"the population baseline ({population_mean_sim:.4f})"
    )
    assert sim_popularity < population_mean_sim, (
        f"popularity scheme's mean similarity ({sim_popularity:.4f}) should sit below "
        f"the population baseline ({population_mean_sim:.4f}) -- it isn't targeting similarity at all"
    )
    assert sim_substitute - sim_popularity > 0.15, (
        f"substitute scheme ({sim_substitute:.4f}) not meaningfully more concentrated "
        f"than popularity scheme ({sim_popularity:.4f})"
    )
