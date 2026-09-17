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

## 2026-09-07 — Classical MNL (`src/models/mnl.py`, `src/utils.py`)

**Two fits on purpose.** `fit_mnl_pytorch` (Adam + cross-entropy, the same
training loop DeepMNL and the Set Transformer will reuse) and
`fit_mnl_scipy` (closed-form NLL + analytic gradient, BFGS via
`scipy.optimize.minimize`) are solving the *same* convex MLE problem two
different ways. Requiring them to agree is the check that the shared
harness's gradient-descent training is actually doing correct MLE, not
just producing plausible-looking numbers.

**Bug 1 (real, not a tolerance issue): category one-hot + bias is not
identified.** First cut used a full `n_categories`-wide one-hot block plus
a `Linear(..., bias=True)`. PyTorch and scipy converged to the same NLL to
machine precision but to *different* individual category weights (off by
~0.1-0.3). Diagnosed by checking `w_pytorch - w_scipy` per category weight
and finding all four differences were the same constant (~-0.105) — the
signature of an exactly flat direction in parameter space, not
optimization noise. Reasoning: a full K-of-K one-hot block sums to exactly
1 in every row, so shifting every category weight by the same constant `c`
shifts every item's score by that same `c`, and softmax within a choice
set is invariant to adding a constant to every item in the set. This holds
*whether or not* there's a separate bias term — removing just the bias
(first attempted fix) was necessary but not sufficient. Fixed by using
`n_categories - 1` dummy columns (category 0 as reference, encoded
all-zero) together with no bias term. This isn't a hack; it's the standard
resolution, and it reflects a real, general fact about discrete choice
models: only utility *differences* between alternatives are ever
identifiable from choice data, never absolute levels (equivalently: there
is no free normalization without an outside option of fixed utility).
Re-verified: with the fix, a full-convergence PyTorch fit and the BFGS fit
agree to NLL within 1e-7 and max weight difference `6e-5` on a real
16.8k-set training split.

**Bug 2 (in the training loop's early-stopping logic, not the model):**
after fixing the identifiability bug, PyTorch and scipy *still* disagreed
by up to ~0.12 on category weights no matter how many epochs were
requested (2000, 8000, 20000 all gave bit-identical output). Cause:
`fit_mnl_pytorch` only checkpoints `best_state` when validation NLL
improves by more than `1e-5`, then reloads that checkpoint at the end.
That's correct behavior for real training against a genuine validation
set (it's what makes early stopping a regularizer). But the MLE-agreement
test had passed the *training* set as its own "validation" set, and once
per-epoch improvement dropped below `1e-5` (which happens quickly on a
convex problem), progress silently froze — thousands of further epochs
changed nothing because the reload always reverted to the same frozen
snapshot. Fixed by giving the agreement test its own plain convergence
loop (`_fit_mnl_to_convergence` in `tests/test_mnl.py`) with no
checkpoint/reload logic, rather than changing `fit_mnl_pytorch` itself --
its early-stopping behavior is correct and desired for the real
train/val/test harness; it was just the wrong tool for an MLE-convergence
check specifically.

**Bayes-optimal benchmark at `context_strength == 0`.** Added
`true_utility` to the synthetic generator's output (the pre-noise
effective utility `V + boost`) so `bayes_optimal_nll` can compute
cross-entropy against the *true* generating softmax at the realized
choices -- an unbiased estimate of that distribution's entropy, i.e. the
best expected NLL any model could achieve. On the full default benchmark
(24k sets, 16.8k/3.6k/3.6k train/val/test), MNL's held-out NLL at
`context_strength == 0` (3,202 test sets) was 1.4345 against a
Bayes-optimal 1.4331 -- a gap of 0.0014 nats, i.e. MNL is essentially at
the theoretical floor exactly where it's correctly specified. On the
`context_strength > 0` strata (the decoy-treated slices), NLL rose to
1.56-1.68 and accuracy fell from 0.44 to 0.33-0.41 -- real degradation
where MNL is structurally missing the boost term, not a suspiciously clean
result (strata are small here, ~130-140 test sets each, so the exact
numbers aren't perfectly monotonic in γ -- expected sampling noise at this
sample size, not evidence of a problem).
