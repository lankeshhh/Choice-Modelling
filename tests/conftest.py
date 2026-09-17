import pytest

from src.data.synthetic import SyntheticConfig, generate_benchmark
from src.utils import set_seed, build_padded_tensors, build_true_utility_tensor, grouped_split


@pytest.fixture(scope="session")
def benchmark():
    """Shared small synthetic benchmark for model tests (MNL, DeepMNL, Set
    Transformer). Session-scoped: generated once and reused across every
    test file, since it's read-only and regenerating/refitting against it
    repeatedly is pure overhead."""
    set_seed(0)
    cfg = SyntheticConfig(n_categories=4, items_per_category=4, seed=0)
    df, items, triads = generate_benchmark(
        cfg=cfg, strengths=(0.0, 0.5, 1.0, 2.0), n_sets_per_strength=1500, seed=0,
    )
    tensors, meta = build_padded_tensors(df, cfg.n_categories)
    true_u = build_true_utility_tensor(df, tensors[0].shape[1])
    train_ids, val_ids, test_ids = grouped_split(meta, seed=0)
    return dict(cfg=cfg, df=df, items=items, tensors=tensors, meta=meta, true_u=true_u,
                train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
