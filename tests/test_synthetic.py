import numpy as np
import pytest

from src.data.synthetic import (
    SyntheticConfig,
    build_item_universe,
    generate_dataset,
    generate_benchmark,
    softmax_probs,
)


@pytest.fixture
def cfg():
    return SyntheticConfig(n_categories=3, items_per_category=3, n_customers=50, seed=1)


def test_item_universe_shape_and_roles(cfg):
    rng = np.random.default_rng(cfg.seed)
    items, triads = build_item_universe(cfg, rng)

    expected_n_items = cfg.n_categories * (cfg.items_per_category + 3)  # +A +B +D
    assert len(items) == expected_n_items
    assert set(items["role"]) == {"filler", "target", "competitor", "decoy"}
    assert len(triads) == cfg.n_categories

    for cat, triad in triads.items():
        a, b, d = triad["A"], triad["B"], triad["D"]
        # decoy is strictly dominated by A: costlier AND lower quality
        assert items.loc[d, "price"] > items.loc[a, "price"]
        assert items.loc[d, "quality"] < items.loc[a, "quality"]
        # A and B are a genuine trade-off, not one dominating the other
        cheaper = items.loc[a, "price"] < items.loc[b, "price"]
        lower_quality = items.loc[a, "quality"] < items.loc[b, "quality"]
        assert cheaper and lower_quality


def test_dataset_schema_and_set_integrity(cfg):
    rng = np.random.default_rng(cfg.seed)
    items, triads = build_item_universe(cfg, rng)
    df = generate_dataset(cfg, decoy_strength=1.0, n_choice_sets=200, rng=rng,
                           items=items, triads=triads)

    expected_cols = {"customer_id", "choice_set_id", "item_id", "category", "price",
                      "quality", "price_z", "quality_z", "role", "scenario",
                      "decoy_strength", "context_strength", "chosen"}
    assert expected_cols.issubset(df.columns)

    # exactly one chosen item per set, and set sizes within bounds
    per_set = df.groupby("choice_set_id")
    assert (per_set["chosen"].sum() == 1).all()
    sizes = per_set.size()
    assert sizes.min() >= cfg.min_set_size
    assert sizes.max() <= cfg.max_set_size
    # no duplicate items within a set
    assert (per_set["item_id"].apply(lambda s: s.duplicated().any()) == False).all()

    # decoy item only ever appears in decoy_treated sets
    decoy_rows = df[df["role"] == "decoy"]
    assert (decoy_rows["scenario"] == "decoy_treated").all()


def test_choice_probabilities_match_closed_form_softmax(cfg):
    """Monte Carlo check: with the DGP's boost/eps mechanics fixed, repeated
    draws of the *same* structural choice set should recover softmax(V+boost)
    within sampling error, for both gamma=0 and gamma>0. This validates the
    Gumbel-max mechanics independent of any model fit."""
    rng = np.random.default_rng(cfg.seed)
    items, triads = build_item_universe(cfg, rng)
    cat = 0
    a, b, d = triads[cat]["A"], triads[cat]["B"], triads[cat]["D"]

    for gamma, include_decoy in [(0.0, False), (1.5, True)]:
        set_items = [a, b, d] if include_decoy else [a, b]
        sub = items.loc[set_items]
        boost = np.array([gamma if i == a and include_decoy else 0.0 for i in set_items])
        true_probs = softmax_probs(sub["V"].to_numpy() + boost)

        n_draws = 20000
        counts = np.zeros(len(set_items))
        for _ in range(n_draws):
            eps = -np.log(-np.log(np.clip(rng.random(len(set_items)), 1e-12, 1 - 1e-12)))
            u = sub["V"].to_numpy() + boost + eps
            counts[int(np.argmax(u))] += 1
        empirical_probs = counts / n_draws

        assert np.allclose(empirical_probs, true_probs, atol=0.02), (
            f"gamma={gamma}: empirical {empirical_probs} vs true {true_probs}"
        )


def test_decoy_breaks_iia_in_predicted_direction(cfg):
    """Model-free sanity check on the generator itself (no model involved).

    IIA says the *ratio* P(A)/P(B) should be unaffected by adding an
    "irrelevant" third alternative. Adding a third item always shifts the
    raw softmax denominator (expected, not a violation) -- so the right
    test is on the odds ratio, not the raw probabilities:
      - gamma=0: odds ratio P(A)/P(B) is unchanged by adding D_A (IIA holds,
        D_A really is irrelevant to the A-vs-B comparison here).
      - gamma>0: odds ratio increases, and increases further as gamma grows
        (IIA is violated in exactly the injected direction).
    """
    rng = np.random.default_rng(cfg.seed)
    items, triads = build_item_universe(cfg, rng)
    cat = 0
    a, b, d = triads[cat]["A"], triads[cat]["B"], triads[cat]["D"]
    v_a, v_b, v_d = items.loc[a, "V"], items.loc[b, "V"], items.loc[d, "V"]

    p_control = softmax_probs(np.array([v_a, v_b]))
    odds_control = p_control[0] / p_control[1]

    prev_odds_treated = odds_control
    for gamma in [0.0, 1.0, 3.0]:
        p_treated = softmax_probs(np.array([v_a + gamma, v_b, v_d]))
        odds_treated = p_treated[0] / p_treated[1]

        if gamma == 0.0:
            assert odds_treated == pytest.approx(odds_control, rel=1e-9)
        else:
            assert odds_treated > odds_control, "decoy should raise the A:B odds ratio"
            assert odds_treated > prev_odds_treated, "the shift should grow with decoy_strength"
        prev_odds_treated = odds_treated


def test_generate_benchmark_covers_all_strengths():
    strengths = (0.0, 0.5, 1.0)
    df, items, triads = generate_benchmark(
        cfg=SyntheticConfig(n_categories=2, items_per_category=2, seed=3),
        strengths=strengths, n_sets_per_strength=100, seed=3,
    )
    assert set(df["decoy_strength"].unique()) == set(strengths)
    assert df["choice_set_id"].nunique() == 100 * len(strengths)
    # context_strength stratification: only decoy_treated rows carry nonzero context
    non_treated = df[df["scenario"] != "decoy_treated"]
    assert (non_treated["context_strength"] == 0.0).all()
