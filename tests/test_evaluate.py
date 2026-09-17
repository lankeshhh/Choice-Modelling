import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.data.synthetic import SyntheticConfig, generate_benchmark
from src.evaluate import (
    stratified_metrics_from_logits, decoy_shift_by_strength,
    true_decoy_shift_by_strength, write_comparison_report, write_bakery_section,
    BAKERY_SECTION_HEADER,
)


def test_stratified_metrics_from_logits_matches_manual_computation():
    # 4 sets, set_size=2: y picks the chosen index; context_strength splits
    # them into two strata of 2 sets each.
    logits = torch.tensor([
        [2.0, 0.0],
        [2.0, 0.0],
        [0.0, 5.0],
        [3.0, 0.0],
    ])
    y = torch.tensor([0, 0, 0, 1])
    context_strength = np.array([0.0, 0.0, 1.0, 1.0])

    df = stratified_metrics_from_logits("test_model", logits, y, context_strength)

    assert set(df["context_strength"]) == {0.0, 1.0}
    assert (df["n"] == 2).all()

    stratum0 = df[df["context_strength"] == 0.0].iloc[0]
    manual_nll0 = torch.nn.functional.cross_entropy(logits[:2], y[:2]).item()
    manual_acc0 = ((logits[:2].argmax(dim=1) == y[:2]).float().mean()).item()
    assert stratum0["nll"] == pytest.approx(manual_nll0)
    assert stratum0["accuracy"] == pytest.approx(manual_acc0)
    assert stratum0["accuracy"] == pytest.approx(1.0)  # both correct by construction

    stratum1 = df[df["context_strength"] == 1.0].iloc[0]
    assert stratum1["accuracy"] == pytest.approx(0.0)  # both wrong by construction


class _ConstantScorer(nn.Module):
    """Test double: ignores item features entirely, returns a fixed score
    per position (broadcast over the batch). Used to test evaluate.py's
    plumbing (does it slice/aggregate correctly) independent of any real
    model's learned behavior."""

    def __init__(self, position_scores: list[float]):
        super().__init__()
        self.position_scores = torch.tensor(position_scores)

    def forward(self, X, mask):
        batch = X.shape[0]
        scores = self.position_scores[: X.shape[1]].unsqueeze(0).expand(batch, -1).clone()
        return scores.masked_fill(~mask, float("-inf"))


@pytest.fixture(scope="module")
def small_df():
    cfg = SyntheticConfig(n_categories=2, items_per_category=2, seed=0)
    df, items, triads = generate_benchmark(
        cfg=cfg, strengths=(0.0, 1.0), n_sets_per_strength=200, seed=0,
    )
    return df, cfg


def test_decoy_shift_by_strength_plumbing(small_df):
    df, cfg = small_df
    max_set_size = df.groupby("choice_set_id").size().max()
    # uniform scorer: every position scores identically, so predicted
    # P(target) should just reflect 1/set_size on average, not react to
    # the decoy at all -- a clean way to check the function reports
    # exactly what a known, trivial model would produce.
    model = _ConstantScorer([0.0] * max_set_size)

    result = decoy_shift_by_strength("uniform", model, df, cfg.n_categories, max_set_size)

    assert set(result["decoy_strength"]) == {0.0, 1.0}
    assert (result["model"] == "uniform").all()
    # a uniform scorer's predicted share should be small and similar
    # between treated/control (no reason for it to differ meaningfully)
    assert (result["shift"].abs() < 0.1).all()


def test_true_decoy_shift_by_strength_matches_manual(small_df):
    df, cfg = small_df
    result = true_decoy_shift_by_strength(df)

    target_rows = df[df["role"] == "target"]
    for gamma in [0.0, 1.0]:
        treated = target_rows[(target_rows["scenario"] == "decoy_treated")
                               & (target_rows["decoy_strength"] == gamma)]
        control = target_rows[(target_rows["scenario"] == "decoy_control")
                               & (target_rows["decoy_strength"] == gamma)]
        expected_shift = treated["chosen"].mean() - control["chosen"].mean()
        row = result[result["decoy_strength"] == gamma].iloc[0]
        assert row["shift"] == pytest.approx(expected_shift)
        assert row["model"] == "True (empirical)"


def test_write_comparison_report_produces_readable_markdown(tmp_path):
    metrics_df = pd.DataFrame([
        dict(model="Bayes-optimal", context_strength=0.0, n=100, nll=1.40, accuracy=0.45),
        dict(model="MNL", context_strength=0.0, n=100, nll=1.43, accuracy=0.44),
        dict(model="DeepMNL", context_strength=0.0, n=100, nll=1.43, accuracy=0.44),
        dict(model="Set Transformer", context_strength=0.0, n=100, nll=1.43, accuracy=0.44),
        dict(model="Bayes-optimal", context_strength=2.0, n=50, nll=1.20, accuracy=0.55),
        dict(model="MNL", context_strength=2.0, n=50, nll=1.68, accuracy=0.33),
        dict(model="DeepMNL", context_strength=2.0, n=50, nll=1.64, accuracy=0.33),
        dict(model="Set Transformer", context_strength=2.0, n=50, nll=1.57, accuracy=0.33),
    ])
    decoy_df = pd.DataFrame([
        dict(model="True (empirical)", decoy_strength=0.0, p_treated=0.08, p_control=0.08, shift=0.0),
        dict(model="MNL", decoy_strength=0.0, p_treated=0.10, p_control=0.10, shift=0.0),
        dict(model="DeepMNL", decoy_strength=0.0, p_treated=0.10, p_control=0.10, shift=0.0),
        dict(model="Set Transformer", decoy_strength=0.0, p_treated=0.11, p_control=0.10, shift=0.01),
        dict(model="True (empirical)", decoy_strength=2.0, p_treated=0.39, p_control=0.07, shift=0.31),
        dict(model="MNL", decoy_strength=2.0, p_treated=0.10, p_control=0.11, shift=-0.01),
        dict(model="DeepMNL", decoy_strength=2.0, p_treated=0.10, p_control=0.11, shift=-0.01),
        dict(model="Set Transformer", decoy_strength=2.0, p_treated=0.12, p_control=0.10, shift=0.02),
    ])
    cfg = SyntheticConfig()
    out_path = tmp_path / "comparison_report.md"

    write_comparison_report(metrics_df, decoy_df, out_path, cfg)

    text = out_path.read_text()
    assert "Set Transformer" in text
    assert "Bayes-optimal" in text
    assert "decoy-shift" in text.lower() or "decoy_strength" in text.lower()
    assert "1.4300" in text or "1.43" in text  # some MNL NLL value made it in
    assert "## Summary" in text


def test_write_bakery_section_appends_without_clobbering_synthetic_content(tmp_path):
    out_path = tmp_path / "comparison_report.md"
    out_path.write_text("# Some report\n\nSynthetic results here.\n")

    bakery_metrics = pd.DataFrame([
        dict(model="MNL", n=1000, nll=2.10, accuracy=0.20),
        dict(model="DeepMNL", n=1000, nll=2.05, accuracy=0.21),
        dict(model="Set Transformer", n=1000, nll=2.02, accuracy=0.22),
    ])

    write_bakery_section(bakery_metrics, out_path, n_items=50, n_sets=1000)
    text = out_path.read_text()

    assert "Synthetic results here." in text  # original content preserved
    assert BAKERY_SECTION_HEADER in text
    assert text.count(BAKERY_SECTION_HEADER) == 1
    assert "2.0200" in text  # Set Transformer's NLL made it in

    # rerunning should not duplicate the section
    write_bakery_section(bakery_metrics, out_path, n_items=50, n_sets=1000)
    text2 = out_path.read_text()
    assert text2.count(BAKERY_SECTION_HEADER) == 1
    assert "Synthetic results here." in text2


def test_write_bakery_section_flags_negligible_spread_instead_of_naming_a_winner(tmp_path):
    """A 0.0004-nat spread isn't a meaningful ranking -- naming a "winner"
    at that precision would overstate noise as signal. The report should
    say so explicitly rather than just picking the argmin."""
    out_path = tmp_path / "comparison_report.md"
    bakery_metrics = pd.DataFrame([
        dict(model="MNL", n=3600, nll=1.7512, accuracy=0.2150),
        dict(model="DeepMNL", n=3600, nll=1.7508, accuracy=0.2081),
        dict(model="Set Transformer", n=3600, nll=1.7508, accuracy=0.2114),
    ])

    write_bakery_section(bakery_metrics, out_path, n_items=50, n_sets=24000)
    text = out_path.read_text()

    assert "statistically indistinguishable" in text
    assert "Lowest NLL:" not in text  # the "meaningful ranking" phrasing shouldn't appear
