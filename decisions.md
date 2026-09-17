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

**Coefficient recovery, not just NLL, at `context_strength == 0`.**
Near-Bayes-optimal NLL doesn't by itself prove the individual coefficients
were recovered -- a model that's misspecified in a way that doesn't cost
much likelihood could still land on the wrong weights. Added
`mnl_hessian`/`mnl_standard_errors` to `mnl.py`: the closed-form observed
Fisher information for conditional/multinomial logit,
`H = sum_n X_n^T (diag(p_n) - p_n p_n^T) X_n` (summed-NLL scale; notably
independent of y, since observed and expected information coincide for
this canonical-exponential-family case). Verified against finite
differences of the analytic gradient first (max abs diff 2e-9 on a small
sample) before trusting it for anything, same pattern as the gradient
check.

Fit MNL via `fit_mnl_scipy` on the full `context_strength == 0` subset of
the default benchmark (21,332 of 24,000 sets -- every plain and
decoy_control set across all γ blocks, plus the entire γ=0 block including
its "decoy_treated" sets, since a boost of 0 has no effect) and compared
fitted coefficients to the true generative values: `cfg.beta_price`,
`cfg.beta_quality`, and the *realized* per-category effects `alpha[c]`
from `build_item_universe` (not the config's `category_effect_std` scalar
-- alpha is drawn per-seed). Since category is encoded as K-1 dummies
relative to category 0, the comparanda for the fitted category weights are
`alpha[c] - alpha[0]`, not `alpha[c]` itself.

| param | true | fitted | se | 95% CI | z |
|---|---|---|---|---|---|
| beta_price | -0.1200 | -0.1233 | 0.0098 | [-0.1426, -0.1041] | -0.34 |
| beta_quality | 1.0000 | 1.0004 | 0.0131 | [0.9746, 1.0262] | 0.03 |
| alpha[1]-alpha[0] | 1.3865 | 1.3745 | 0.0332 | [1.3094, 1.4397] | -0.36 |
| alpha[2]-alpha[0] | 1.3025 | 1.2893 | 0.0302 | [1.2301, 1.3485] | -0.44 |
| alpha[3]-alpha[0] | 1.0202 | 0.9885 | 0.0318 | [0.9262, 1.0508] | -1.00 |
| alpha[4]-alpha[0] | 0.7618 | 0.7659 | 0.0303 | [0.7066, 0.8252] | 0.14 |

All six parameters land inside their 95% CI with `|z| <= 1.00` -- no
systematic bias in any direction, consistent with pure sampling noise
rather than a real gap. Cross-checked the Hessian-based SEs against a
200-resample bootstrap (resample choice sets with replacement, refit each
time): bootstrap/Hessian SE ratios were 0.99-1.08 across all six
parameters, i.e. the two independent SE estimates agree closely, so the
asymptotic-normality assumption behind the Hessian-based CIs isn't doing
anything suspicious here. Added
`test_mnl_recovers_true_coefficients_at_zero_context_strength` as a
permanent regression test (on the smaller test fixture, so a looser
`|z| < 4` threshold than the large-sample table above) -- this is the kind
of thing that could silently break (e.g. if a future refactor reintroduced
the one-hot identifiability bug) without moving NLL enough to be caught by
the Bayes-optimal check alone.

## 2026-09-07 — DeepMNL (`src/models/deep_mnl.py`)

**Refactor first: extracted `train_choice_model` into `utils.py`.**
`fit_mnl_pytorch`'s Adam-plus-early-stopping loop is exactly what DeepMNL
(and later the Transformer) needs too, so pulled it into a generic trainer
parameterized by the model instance, before it existed twice.
`fit_mnl_pytorch` is now a two-line wrapper around it. Differences in
results across the three models are then guaranteed to come from model
structure, not from accidentally-different training procedures.

**`patience=None` added to `train_choice_model`.** The same
checkpoint-freeze issue documented under MNL above (only checkpointing on
>1e-5 validation improvement, then reloading that checkpoint at the end)
resurfaced in DeepMNL's own "does this model actually learn" smoke test,
for the identical reason: passing the training set as its own "validation"
set. Rather than patch around it a third time (it will hit the
Transformer's smoke test too), fixed it at the source: `patience=None`
skips checkpointing entirely and returns the model's actual final state
after running all requested epochs. `fit_mnl_pytorch`'s and
`fit_deep_mnl`'s real early-stopping behavior (patience=int, the default)
is unchanged and still correct for genuine train/val use.

**Architecture.** `utility_i = MLP(x_i)`, a 2-hidden-layer (width 32, ReLU)
network, same weights shared across items and across the set, applied
independently per item -- `nn.Linear`/`nn.ReLU` broadcast over the leading
(batch, set) dimensions, so no reshaping is needed and there is no
computational path for one item's features to reach another item's score.
No identifiability concerns here the way there were for MNL's linear
weights (nobody is meant to read individual MLP weights for meaning) --
DeepMNL is a flexible black-box utility approximator by design, so a
normal bias-including MLP is fine.

**Verified the "no interaction" claim directly, not just assumed it from
the architecture.** `test_score_is_independent_of_other_items_in_set`
takes a batch, perturbs every item's features *except* position 0, and
asserts position 0's score is bit-identical before and after. This is the
literal, checkable version of the ablation claim DeepMNL exists to make,
and it's an exact test (not a statistical one) since it follows from the
computation graph, not from how well training converged.

**Test-threshold bug (not a model bug): initial smoke-test bound was just
a guess.** First cut of `test_deep_mnl_fits_training_data` asserted NLL
should drop by at least 0.3 nats from random init; it failed at an
observed drop of ~0.208. Checked interactively across epochs
{150,500,1000} x lr {0.01,0.02} and the converged NLL was stable at
~1.524 in every case (not an under-training artifact -- confirmed
`patience=None` was working correctly here too). The 0.3 threshold was
simply an unfounded guess; lowered to 0.15, comfortably below the observed
stable value.

**Full-benchmark result (24k sets, same split as MNL): DeepMNL tracks MNL
almost exactly at every context_strength level**, which is the expected
and correct ablation outcome, not a null result to be concerned about:

| context_strength | n | MNL NLL | MNL acc | DeepMNL NLL | DeepMNL acc |
|---|---|---|---|---|---|
| 0.0 | 3202 | 1.4345 | 0.4441 | 1.4346 | 0.4413 |
| 0.5 | 138 | 1.6137 | 0.3696 | 1.6102 | 0.3768 |
| 1.0 | 131 | 1.5591 | 0.4122 | 1.5510 | 0.4122 |
| 2.0 | 132 | 1.6778 | 0.3258 | 1.6412 | 0.3333 |

At `context_strength=0`, DeepMNL's gap to Bayes-optimal is +0.0015 --
essentially identical to MNL's own +0.0014 -- confirming an MLP with
enough capacity recovers a genuinely linear truth just as well as the
correctly-specified linear model, with no advantage and no disadvantage.
Both models degrade together as context_strength rises, since neither can
see the decoy item. This is the whole point of building DeepMNL as a
separate step: it isolates "does nonlinearity help" (no, because the truth
here is linear) from "does seeing the rest of the assortment help" (not
yet tested -- that's what the Set Transformer is for). A large DeepMNL
advantage over MNL anywhere in this table would have been the surprising,
investigate-before-trusting result; this isn't that.

`test_deep_mnl_misses_decoy_effect_like_mnl` checks the same story on
held-out data at the test-fixture scale: DeepMNL's learned P(A) shift from
decoy presence is required to be < 0.05 in absolute value (expected ~0),
confirming the architectural guarantee actually shows up in trained
behavior, not just in a hand-constructed forward-pass probe.
