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

## 2026-09-07 — Set Transformer (`src/models/set_transformer.py`)

**Architecture and structural verification** (see the commit message /
module docstring for the full detail): linear item embedding, 2
TransformerEncoder layers (d_model=32, 4 heads), masked with
src_key_padding_mask, no positional encoding (a choice set is unordered,
so permutation equivariance is the correct inductive bias -- checked
directly in `test_permutation_equivariance`, not assumed). Also checked
directly: perturbing only padded slots doesn't change valid-item scores
(masking polarity is an easy inversion bug), and perturbing *other* real
items *does* change item 0's score (the direct contrast to DeepMNL's
independence proof) -- confirming the architecture actually has the
computational capability the whole benchmark is built to test for.

**The core research-question result -- reported honestly, not the clean
story initially hoped for.**

On the full 24k-set default benchmark, stratified test NLL/accuracy
(Transformer trained to genuine convergence -- see below):

| context_strength | n | MNL NLL | DeepMNL NLL | Transformer NLL | MNL acc | DeepMNL acc | Transformer acc |
|---|---|---|---|---|---|---|---|
| 0.0 | 3202 | 1.4345 | 1.4346 | 1.4339 | 0.4441 | 0.4413 | 0.4413 |
| 0.5 | 138 | 1.6137 | 1.6102 | 1.6063 | 0.3696 | 0.3768 | 0.3623 |
| 1.0 | 131 | 1.5591 | 1.5510 | 1.5407 | 0.4122 | 0.4122 | 0.3893 |
| 2.0 | 132 | 1.6778 | 1.6412 | 1.5731 | 0.3258 | 0.3333 | 0.3258 |

NLL is modestly, consistently better for the Transformer as
context_strength rises. Accuracy is *not* a clean win -- it's actually
slightly worse than MNL/DeepMNL at 0.5 and 1.0, and only ties MNL at 2.0.
This mixed picture is reported as-is rather than emphasizing only the NLL
column, which would overstate the result.

**The predicted-probability-shift diagnostic (the sharper test of "did it
actually learn the mechanism") shows the same story more starkly.**
Measuring the model's predicted P(A) with vs. without the decoy present
(the same diagnostic used for DeepMNL, now against the true injected
shift too):

First pass (300 epochs, standard patience=25 -- did NOT trigger early
stopping, i.e. not yet converged):

| decoy_strength | true shift | Transformer's predicted shift | fraction captured |
|---|---|---|---|
| 0.0 | +0.0047 | +0.0082 | (n/a, true effect ~0) |
| 0.5 | +0.0459 | +0.0129 | ~28% |
| 1.0 | +0.1054 | +0.0069 | ~7% |
| 2.0 | +0.3144 | +0.0087 | ~3% |

Before trusting this, checked whether it was simply undertrained: reran
with epochs=1500, patience=100. This time early stopping *did* trigger
(451 epochs), and val_nll was flat across the last 10 checkpoints
(1.4844-1.4845) -- genuine convergence, not an artifact of stopping too
early:

| decoy_strength | true shift | Transformer's predicted shift (converged) | fraction captured |
|---|---|---|---|
| 0.0 | +0.0047 | +0.0184 | (n/a) |
| 0.5 | +0.0459 | +0.0225 | ~49% |
| 1.0 | +0.1054 | +0.0169 | ~16% |
| 2.0 | +0.3144 | +0.0188 | ~6% |

More training helped somewhat at the low end but the qualitative picture
is unchanged, and one pattern stands out: **the learned shift is nearly
flat across all four decoy_strength levels (0.017-0.023) despite the true
effect spanning a 67x range (0.005-0.314).** The model learned "a decoy is
present -> apply a small, roughly constant boost," not "apply a boost
calibrated to how strong this regime's effect actually is."

**Working hypothesis for why, and why it's a benchmark-design fact, not
(only) a training/capacity failure:** `decoy_strength` is not an
observable input feature anywhere in x_i -- items in a decoy_strength=2.0
block and a decoy_strength=0.5 block look statistically identical to the
model; only the *conditional outcome frequencies* differ, and those
frequencies are pooled across all four blocks in one cross-entropy
objective the model cannot condition on regime. The Bayes-optimal
response *for this model class, given this input*, may genuinely be close
to a single pooled-average boost rather than four different ones -- the
model isn't necessarily failing to find an achievable optimum, it may be
close to the best achievable optimum *without observing the regime*. A
second, compounding factor: decoy_treated sets are a small minority of
training data (~3.75% per decoy_strength block), so the aggregate
cross-entropy loss has limited gradient signal to sharpen this specific
sub-pattern relative to getting everything else right. Neither hypothesis
was tested to isolation (e.g. by exposing decoy_strength as an explicit
feature to see if the ceiling rises) -- flagged as a natural next step,
not done here to avoid quietly changing the benchmark's information
content mid-comparison.

**Checked at the smaller scale the automated tests actually use, and the
signal doesn't reliably survive.** The pytest fixture (6,000 sets, ~1,500
per decoy_strength level) is deliberately small for fast tests. Re-ran the
same diagnostic there: MNL/DeepMNL shifts were -0.0071 to +0.0007 (noise,
as expected -- structurally guaranteed to be unresponsive, per their own
tests), and the Transformer's shifts were -0.0027 to +0.0037 -- *not*
reliably distinguishable from that noise floor, and not even consistently
positive. The modest signal found on the 24k-set benchmark needs that
scale of data to show up at all.

**Decision: no committed pytest assertion for the decoy-shift magnitude or
comparison.** Writing one against the small fixture would mean either
asserting something not reliably true (it isn't reliably positive there)
or setting a threshold so loose it would also pass for MNL/DeepMNL's pure
noise -- neither is a meaningful test. This finding is documented here
instead of encoded as a regression test. What *is* tested and committed:
the architectural capability (`test_score_depends_on_other_items_in_set`,
already true at initialization, no data scale required) and the
structural guarantee that MNL/DeepMNL cannot respond at all
(`test_score_is_independent_of_other_items_in_set` /
`test_deep_mnl_misses_decoy_effect_like_mnl`) -- both robust regardless of
sample size, unlike the Transformer's *quantitative* response magnitude.

**Headline, stated honestly rather than oversold:** the Set Transformer
shows a real, directionally-correct, but incomplete recovery of the
injected context effect at this data scale and model capacity. It is
never worse than MNL/DeepMNL on NLL and modestly better as the effect
strengthens, its predicted probabilities move in the right direction where
MNL/DeepMNL structurally cannot move at all, but it captures roughly
6-50% of the true effect magnitude (worse at higher decoy_strength, where
it matters most) and doesn't calibrate to how strong the effect actually
is in a given regime. This is not the clean "Transformer wins decisively"
story a more optimistic framing might expect -- and per the original
spec's own instruction not to force the result, it's reported as exactly
that: a partial, honest validation of the core hypothesis, not a complete
one.

**Follow-up: reconsidered the "no committed test" decision after a
multi-seed check, and it was worth reconsidering.** The raw-shift
comparison above is confounded in a way worth naming precisely: a
decoy_treated set has one more competing alternative than its matched
decoy_control set *by construction* (the decoy itself is the extra item --
this was the whole point of the earlier filler-count fix, see the
synthetic-data-generator section above). Adding any extra alternative
mechanically dilutes raw P(A) through the softmax denominator, regardless
of whether that alternative carries any genuine attraction effect. A
perfectly-fit MNL *should* show a negative raw shift from this alone
whenever the boost is small relative to the dilution -- that's correct
model behavior, not a detection failure. This explains why several raw
shifts (including some of the Transformer's own) came out negative in the
tables above: the metric mixes two effects (dilution, always present and
negative-ish; genuine attraction boost, positive and specific to whichever
model can compute it) rather than isolating the second one.

That reframes the right test: not "is the Transformer's raw shift
positive" (confounded, and not reliably true) but "does the Transformer
show more net positive shift than MNL/DeepMNL, on top of the same shared
dilution baseline they all experience." Checked this relative claim across
4 independent data seeds at fixture scale (this repo's fixture seed=0,
plus 1/2/3 run interactively, each fitting fresh MNL/DeepMNL/Transformer
instances):

| seed | MNL mean shift | DeepMNL mean shift | Transformer mean shift | TF - MNL | TF - DeepMNL |
|---|---|---|---|---|---|
| 0 | -0.0034 | -0.0032 | -0.0003 | +0.0031 | +0.0028 |
| 1 | -0.0024 | -0.0027 | +0.0009 | +0.0033 | +0.0036 |
| 2 | -0.0151 | -0.0120 | -0.0077 | +0.0074 | +0.0043 |
| 3 | -0.0061 | -0.0090 | -0.0048 | +0.0013 | +0.0042 |

The relative margin (Transformer exceeds both baselines) held in all 4
seeds without exception, with a minimum margin of +0.0013 (vs MNL) and
+0.0028 (vs DeepMNL) -- small, but a real, reproducible, non-arbitrary-
direction effect, unlike the raw shift's sign which flips seed to seed.
This is different in kind from a claim about magnitude recovery (still
not supported, see above): it's the claim that the Transformer learns
*some* real incremental context-sensitivity MNL/DeepMNL cannot learn at
all, which the architecture guarantees is possible and this data confirms
actually happens, even if only partially.

Added `test_set_transformer_decoy_shift_exceeds_baselines`, asserting
`mean_shift(Transformer) - mean_shift(MNL) > 0.0005` and the same vs.
DeepMNL -- comfortably under every observed margin (smallest was +0.0013)
so there's real headroom, not a threshold sitting at the edge of what was
actually seen. Also factored the repeated "predicted P(target) over a
choice-set subset" logic (previously duplicated in the DeepMNL test and
about to be duplicated a third time here) into
`utils.predicted_target_share`, used by both test files now.

## 2026-09-07 — Shared harness and comparison report (`src/train.py`, `src/evaluate.py`)

**Split responsibility: `evaluate.py` holds pure metric functions that
take already-computed logits/predictions, never a model or raw data.**
This makes every function in it cheap to unit test (`tests/test_evaluate.py`)
without retraining anything -- hand-constructed logits, a trivial
"constant scorer" test double standing in for a real model, small
synthetic dataframes. `train.py` is the orchestration script: generates
the benchmark, trains all three models on identical splits with the
hyperparameters already validated in this log (in particular the Set
Transformer's "extended" config, epochs=1500/patience=100, confirmed
earlier to reach genuine convergence rather than the under-converged
first-pass config), evaluates them plus the Bayes-optimal reference, and
writes `results/metrics.csv`, `results/decoy_shifts.csv`, and
`results/comparison_report.md`.

Two test-writing mistakes caught by actually running the tests (both
fixed before commit, consistent with the pattern throughout this log):
the first stratified-metrics test asserted "both correct"/"both wrong"
for hand-picked logits where I'd miscalculated argmax by hand twice in a
row (`[0.0, 2.0]` favors index 1, not 0; a tie `[1.0, 1.0]` with
`torch.argmax` breaks toward the first index) -- the function's own
output was correct both times, matching an independently-computed manual
NLL/accuracy; the fixture just didn't say what I intended.

**Bayes-optimal NLL drops sharply at high decoy_strength (1.3335 at
decoy_strength=2.0, vs. 1.4331 at decoy_strength=0)** -- worth noting
since it looks surprising at a glance. This is correct, not a bug: a
strong boost pushes the true softmax distribution toward near-certainty
on the target item, which lowers entropy (and therefore the Bayes-optimal
NLL) even though the *raw* choice-prediction problem intuitively sounds
"harder" with more going on in the set.

**Ran the real pipeline (not just tests) on the full 24k-set default
benchmark before committing**, and it reproduced the numbers already
validated interactively in this log almost exactly (same seed=0
throughout this whole project): NLL at context_strength=0
(1.4345/1.4346/1.4339 for MNL/DeepMNL/Transformer, Bayes-optimal 1.4331),
NLL at context_strength=2.0 (1.6778/1.6412/1.5731), and the Set
Transformer's decoy-shift diagnostic values (+0.0184/+0.0225/+0.0169/
+0.0188 across decoy_strength 0/0.5/1/2) matching the "extended,
converged" run to 4 decimal places. No surprises, nothing requiring
further investigation -- committed `results/metrics.csv`,
`results/decoy_shifts.csv`, and `results/comparison_report.md` as the
formal, reproducible version of everything discussed qualitatively
above. Model checkpoints (`checkpoints/*.pt`) are gitignored, same
reasoning as `data/` -- regenerate via `python -m src.train` rather than
committing binary artifacts.

## 2026-09-07 — Real-data validation: Bakery (`src/data/bakery.py`, `src/train_bakery.py`)

**Which dataset, and why.** "Bakery" was named in the original project
spec alongside "Expedia" for later real-data work, with no file or URL
given. Rather than guess, searched for what's canonically meant in the
discrete-choice/assortment literature and found
[Benson, Kumar & Tomkins (WSDM 2018), "A Discrete Choice Model for Subset
Selection"](https://github.com/arbenson/discrete-subset-choice), which
bundles a `bakery.txt` dataset (the well-known Extended Bakery basket
data: 75,000 transactions, 50 unique items). Verified this before
building anything on top of it (cloned the repo, read the actual data
format, checked size and cardinality) rather than trusting the search
summary alone. Vendored the raw file unmodified at
`src/data/external/bakery.txt` (742KB, small and stable enough to commit
directly, unlike the gitignored generated `data/`) with a `SOURCE.md`
carrying attribution and the paper citation.

**Two real structural gaps from the synthetic setting, and how each was
resolved -- both documented up front rather than discovered as surprises
partway through, since both were foreseeable from just reading the raw
data format before writing any conversion code:**

1. **No assortment in the raw data at all.** Each line of `bakery.txt` is
   a whole *basket* (subset selection) -- items purchased together, not
   "an assortment was shown, one item was chosen." Considered and rejected
   a "leave-one-out within the basket" conversion (treat one basket item
   as chosen, the rest as the choice set) because it's conceptually wrong:
   every item in a basket was actually purchased, so treating co-purchased
   items as "rejected alternatives" misrepresents what happened. Instead:
   for each basket, one item is chosen uniformly at random as the
   observed choice, and the rest of the choice set is filled with items
   *not* in that basket, sampled with probability proportional to
   `popularity^0.75` (standard unigram-negative-sampling practice, e.g.
   word2vec/implicit-feedback recommenders) rather than uniformly --
   uniform negatives would make discrimination trivially easy and measure
   mostly "did the model memorize which items are rare," not choice
   behavior. This is a real, named approximation: we don't know what a
   shopper actually saw and rejected, only what they didn't buy that
   trip. `test_negatives_exclude_original_basket_items` checks the one
   hard invariant this construction must satisfy (a negative is never
   something actually in the original basket) by reconstructing the exact
   basket per `choice_set_id` and checking directly, not a weaker proxy.

2. **No item attributes.** `bakery.txt` has only integer item IDs -- no
   price, quality, or category. To reuse `featurize()` unchanged (per the
   goal: only the data loading should change), `category` is set to the
   item's own identity (each item is its own singleton category, i.e. a
   per-item fixed effect via the same K-1 dummy encoding used for
   synthetic categories) and `price_z`/`quality_z` are set to `0.0` for
   every row -- present so `featurize()`'s column access doesn't break,
   genuinely uninformative since there's no real price/quality signal to
   put there. This is stated plainly in the loader's docstring and the
   report rather than left for a reader to discover by inspecting the
   data.

**Scale-matched to the synthetic benchmark on purpose.** Subsampled to
24,000 transactions (from 75,000 available) specifically to match the
synthetic experiment's scale, so a real-vs-synthetic comparison isn't
confounded by "more data helps everyone regardless of architecture."

**The result, run once for real and reported as-is:**

| model | n | NLL | accuracy |
|---|---|---|---|
| MNL | 3600 | 1.7512 | 0.2150 |
| DeepMNL | 3600 | 1.7508 | 0.2081 |
| Set Transformer | 3600 | 1.7508 | 0.2114 |

**All three models are statistically indistinguishable** -- NLL spread is
0.0004 nats, accuracy spread 0.7 points, both far smaller than anything
seen in the synthetic benchmark's `context_strength > 0` strata. The Set
Transformer's synthetic-data advantage has effectively disappeared here.
This was corrected for in the auto-generated report too:
`write_bakery_section`'s first draft named a "winner" by raw NLL argmin,
which was technically true (Set Transformer lowest by 0.0004) but
misleading at that precision -- overstating noise as a finding. Fixed to
state the spread explicitly and call it indistinguishable below a 0.01-nat
threshold, with a dedicated test
(`test_write_bakery_section_flags_negligible_spread_instead_of_naming_a_winner`)
so this doesn't regress.

**Why, mechanistically, not just "real data is different":** this is
explicable, not mysterious, and was anticipated during the loader's design
(see point 1 above) rather than discovered as a surprise after the fact.
Negatives are sampled i.i.d. from a fixed marginal popularity
distribution, independent of which item is chosen or what else is in the
set. That means a negative's identity carries no information beyond
overall item popularity -- which MNL's per-item fixed effects already
capture completely. There is no cross-item substitution or context
structure of the kind the synthetic decoy effect deliberately injects, so
there is nothing for attention over the set to exploit; the Bayes-optimal
solution to *this specific construction* is a saturated per-item
popularity model, which all three architectures can equally represent.
This is a genuinely informative negative result, not a failure to find
something that was there: it shows that whether a set-aware model can
have any advantage depends critically on whether the *choice-set
construction* preserves real substitution structure, not just on whether
the underlying data is "real" vs. "synthetic." A more realistic
choice-set construction (e.g. negatives drawn from items that empirically
co-occur with or substitute for the chosen item, rather than population
marginal popularity) might recover some of the synthetic advantage --
noted as a natural next step, not attempted here to avoid quietly
tuning the construction toward a more favorable result after seeing this
one.
