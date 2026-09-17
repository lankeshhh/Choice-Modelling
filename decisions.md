# Design decisions log

Short, dated notes on non-obvious choices and why. Newest at the bottom.

## 2026-09-07 — Synthetic data generator (`src/data/synthetic.py`)

**Context-effect mechanism: asymmetric-dominance decoy, not nested-logit /
correlated errors.**
Chose a deterministic decoy boost (`boost_i(S) = decoy_strength` iff `i` is
the designated target item `A` and its dominated decoy `D_A` is in the same
set `S`) over a nested-logit-style correlated-error generative process.
Reasons:
- It's the mechanism the spec names directly (asymmetric dominance /
  similarity effect).
- It's falsifiable without fitting any model: fix `A`, `B`, compare the
  `P(A)/P(B)` odds ratio with vs. without `D_A` in the set. Nested logit
  would require fitting to check anything.
- At `decoy_strength = 0` (or decoy absent), the data is *exactly* a
  correctly-specified MNL — gives a clean "MNL should be near-optimal here"
  anchor for later sanity checks on the models themselves.
- The choice rule stays a standard Gumbel-argmax (softmax) at the per-set
  level throughout — the violation is that MNL/DeepMNL only ever observe
  `x_i = (price_i, quality_i, category_i)`, so they structurally cannot
  compute `boost_i(S)`, which depends on *other* items in the set. The
  softmax mechanics aren't broken; the *input* a feature-only model has
  access to is insufficient.

**Trade-off pair (A, B) is constructed, not discovered by chance.**
`A` (cheap, low quality) and `B` (pricey, high quality) are built with
`trade_off_price_gap` and an auto-derived `trade_off_quality_gap` chosen so
`beta_price * gap_price ≈ beta_quality * gap_quality` — i.e. A and B have
approximately equal structural utility `V` before any decoy is added. This
guarantees a genuine, contested trade-off (not one item quietly dominating
the other), so the decoy has something real to tip.

**Decoy `D_A` is dominated on *both* attributes.**
`price_D = price_A + decoy_delta_price`, `quality_D = quality_A -
decoy_delta_quality`, both deltas positive by config. This makes `V_D < V_A`
by construction — no random draw could accidentally make the decoy
*attractive*, which would confound the experiment.

**Bug caught during sanity-checking, fixed before commit: filler-count
confound.** First implementation fixed total set size and derived filler
count as `size - len(core)`. Since `decoy_treated` sets have a 3-item core
(A, B, D) vs. `decoy_control`'s 2-item core (A, B), this gave
`decoy_treated` sets one fewer filler competitor *on average* — inflating
raw `P(A)` and `P(B)` together in `decoy_treated` for a reason having
nothing to do with the decoy effect. Caught it because the raw-`P(A)` gap
persisted even at `decoy_strength = 0`, where it should be zero. Confirmed
via the `P(A)/P(B)` odds ratio (which IIA says must be invariant to this
kind of confound) that the effect was in the "fewer competitors" channel,
not a real IIA violation. Fixed by drawing filler count independently of
whether the decoy is included, so `decoy_control` and `decoy_treated` sets
now share the same filler distribution and differ by exactly one item
(`D_A`), which is what "the decoy is either presented or not" should mean.
Re-verified at scale (see below) that the confound is gone.

**Empirical verification at scale (not just the unit tests).**
Generated large one-off runs (10-60k choice sets) outside the test suite to
check statistics that only stabilize with enough samples:
- At `decoy_strength = 0`, `P(A)/P(B)` odds ratio: 0.400 (control) vs. 0.415
  (treated), z = -0.97 on raw `P(A)` — not distinguishable from no effect at
  n≈9k sets/arm. Confirms the filler-count fix actually worked, not just
  passed a small unit test.
- At `decoy_strength ∈ {0.5, 1.0, 2.0}`, the odds-ratio multiplier
  (treated / control) was `x1.53 / x2.69 / x6.75` against a closed-form
  prediction of `exp(γ) = 1.65 / 2.72 / 7.39` — matches closely. This is the
  expected result for a flat additive utility boost under a softmax choice
  rule (`odds *= exp(boost)`), and having the empirical numbers land close
  to it (n=20k sets/level) is strong evidence the simulator's Gumbel/softmax
  mechanics are implemented correctly, not just "roughly plausible."

**Sample size: `n_sets_per_strength` raised from 2000 to 6000.**
With `decoy_fraction=0.3` split ~50/50 between control/treated, 2000 sets
per γ-level left only ~300 sets per scenario per level — too thin for a
low-variance stratified comparison later (this is what produced visibly
noisy numbers in the first real-scale run before the confound fix, on top
of the confound itself). 6000/level gives ~900 sets per scenario per level,
24000 total sets across the 4 default γ levels.

**Unit tests (`tests/test_synthetic.py`), what each guards against:**
- `test_item_universe_shape_and_roles`: triad geometry (decoy dominated on
  both attributes; A/B is a genuine trade-off, not dominance).
- `test_dataset_schema_and_set_integrity`: one chosen item per set, set
  sizes in bounds, no duplicate items in a set, decoy items only ever
  appear in `decoy_treated` rows.
- `test_choice_probabilities_match_closed_form_softmax`: Monte Carlo over a
  fixed structural set recovers `softmax(V + boost)` within tolerance —
  validates the Gumbel-argmax mechanics directly, independent of any model.
- `test_decoy_breaks_iia_in_predicted_direction`: the textbook IIA test.
  Originally written wrong (checked raw `P(A)` gap == 0 at γ=0); running it
  immediately caught that adding a third alternative always perturbs raw
  probabilities even under true IIA, since the softmax denominator changes.
  Rewrote to check the `P(A)/P(B)` odds ratio instead, which IIA actually
  constrains. Left this failure-then-fix as an example of why the "run it
  and check the number, don't assume" step matters even for the test code
  itself, not just the model code.
- `test_generate_benchmark_covers_all_strengths`: sweep produces the
  expected γ levels, one block of choice sets per level, unique set ids
  across blocks.
