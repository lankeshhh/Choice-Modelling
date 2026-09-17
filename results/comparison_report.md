# Choice modeling comparison: MNL vs. DeepMNL vs. Set Transformer

Synthetic benchmark with an injected asymmetric-dominance (decoy) context effect. See `decisions.md` for the full generation mechanism, sanity checks, and per-model findings; this report is the auto-generated numeric summary.

Config: 5 categories, 4 filler items/category, sets of size 4-8, decoy_fraction=0.3.

## Held-out NLL by context_strength

| context_strength | n | Bayes-optimal | MNL | DeepMNL | Set Transformer |
|---|---|---|---|---|---|
| 0.0 | 3202 | 1.4331 | 1.4345 | 1.4346 | 1.4339 |
| 0.5 | 138 | 1.5989 | 1.6137 | 1.6102 | 1.6063 |
| 1.0 | 131 | 1.5188 | 1.5591 | 1.5510 | 1.5407 |
| 2.0 | 132 | 1.3335 | 1.6778 | 1.6412 | 1.5731 |

## Held-out accuracy by context_strength

| context_strength | n | MNL | DeepMNL | Set Transformer |
|---|---|---|---|---|
| 0.0 | 3202 | 0.4441 | 0.4413 | 0.4413 |
| 0.5 | 138 | 0.3696 | 0.3768 | 0.3623 |
| 1.0 | 131 | 0.4122 | 0.4122 | 0.3893 |
| 2.0 | 132 | 0.3258 | 0.3333 | 0.3258 |

## Decoy-shift diagnostic: predicted P(target) with vs. without the decoy

Raw shift is confounded by denominator dilution (decoy_treated sets have one more competing alternative than decoy_control sets by construction) -- compare shifts *across models*, not against zero. See `decisions.md` for the full explanation.

| decoy_strength | True (empirical) | MNL | DeepMNL | Set Transformer |
|---|---|---|---|---|
| 0.0 | +0.0047 | -0.0037 | -0.0035 | +0.0184 |
| 0.5 | +0.0459 | +0.0008 | +0.0013 | +0.0225 |
| 1.0 | +0.1054 | -0.0045 | -0.0047 | +0.0169 |
| 2.0 | +0.3144 | -0.0021 | -0.0020 | +0.0188 |

## Summary

At the highest context_strength stratum (2.0), held-out NLL was MNL=1.6778, DeepMNL=1.6412, Set Transformer=1.5731.
Mean decoy-shift margin (Set Transformer minus MNL) across all decoy_strength levels: +0.0215.
This is a partial, honest result, not a clean win: the Set Transformer is directionally correct and modestly better on NLL as the injected effect strengthens, but does not fully recover the true effect magnitude and its response does not calibrate to decoy_strength. See `decisions.md` for the full investigation, including the multi-seed robustness check and the working hypothesis for why (decoy_strength is not an observable input feature, and decoy-treated examples are a thin slice of training data).

## Real-data validation: Bakery

Same models, same shared harness (`featurize`, `build_padded_tensors`, `grouped_split`, `train_choice_model`), same train/val/test discipline as the synthetic experiment above -- only the data loading differs. This checks whether the Set Transformer's advantage from the controlled synthetic experiment holds up on real purchase data, reported honestly whatever the result turns out to be. See `src/data/bakery.py` and `decisions.md` for the full data-construction methodology and its limitations -- in particular, this is basket (subset-selection) data with a constructed choice set, not a dataset of actual presented assortments, and items carry no real attributes (price/quality are unavailable; `category` is each item's own identity).

Data: [Benson, Kumar & Tomkins (WSDM 2018)](https://github.com/arbenson/discrete-subset-choice) bakery basket dataset, 50 items, 24,000 choice sets (subsampled to match the synthetic benchmark's scale).

No Bayes-optimal reference and no decoy-shift diagnostic here -- there is no known true utility for real data, and no injected effect to check recovery of.

| model | n | nll | accuracy |
|---|---|---|---|
| DeepMNL | 3600 | 1.7508 | 0.2081 |
| MNL | 3600 | 1.7512 | 0.2150 |
| Set Transformer | 3600 | 1.7508 | 0.2114 |

NLL spread across all three models is 0.0004 nats (Set Transformer lowest, MNL highest) -- statistically indistinguishable, not a meaningful ranking. Compare this to the synthetic benchmark's much larger, systematic margins above: on this real-data construction, the Set Transformer's advantage has effectively disappeared.
