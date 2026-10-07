# 13_hgbr_tuning_msl40_l2reg01 — HGBR tuning: min_samples_leaf=40, l2_regularization=0.1

## Question / hypothesis

Does increasing tree-leaf regularization (larger min_samples_leaf=40 and
l2_regularization=0.1) at the baseline learning_rate/iterations reduce
CV RMSE by preventing individual leaves from fitting noise in small patient subgroups?

## Motivation

The residual-by-target-quartile pattern (Q1: mean −9.2, Q4: mean +11.5)
shows systematic under-prediction at low scores and over-prediction at
high scores. This is a classic sign that individual trees are overfitting
to extreme target values. Two regularisation handles exist in HGBR:
- `min_samples_leaf`: require more samples per leaf → wider, smoother
  predictions, less overfitting on small subgroups.
- `l2_regularization`: penalize leaf weight magnitude → shrinks extreme
  leaf values toward zero.

The default `min_samples_leaf=20` was designed for "big datasets"; with
44 590 visits across 5 576 patients and multiple visits per patient, a
larger min_samples_leaf may better prevent fitting to per-patient noise.

Feature set: **05 feature set** (9 numeric) — same as 11-12, per pre-declared winner rule (05 wins).
Sourcing strategy: B7 carry-forward (compact tuning batch — tree complexity).

## Method

Feature set: pre-declared winner of 10 vs 05 (same as 11 and 12).
Estimator: `HistGradientBoostingRegressor(learning_rate=0.1, max_iter=200,
min_samples_leaf=40, l2_regularization=0.1, random_state=0,
early_stopping=False)`. Other parameters at sklearn 1.9.1 defaults.

CV: identical `GroupKFold(n_splits=5)` fixed patient splits as 03–12.

## Risks

- `min_samples_leaf=40` may be too large for small patient subgroups
  (e.g. rare gene categories), causing underfitting for those groups.
- `l2_regularization=0.1` is a moderate value; too large would over-shrink
  all leaves. The sklearn docs note this parameter was added in 1.2; 0.1
  is a conservative starting point.
- The residual pattern may not be reducible via regularization alone if
  it reflects genuine target distribution asymmetry (floor at 0).

## Outcome

Mean RMSE 7.363917, std 0.138674 (fixed GroupKFold-5).
05_duration ref: 7.375098 (Δ = −0.011181).
Leaf regularization alone (min_samples_leaf=40, l2=0.1) improves over baseline
but not as much as lr tuning (12: 7.342917).

## Status

- **State:** done
- **Approved by user on:** current session (user-authorized batch)
- **Headline result:** RMSE 7.363917 (fold std 0.138674, fixed GroupKFold-5). Δ=−0.011 vs 05. Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47995
- **Implication for next iteration:** Regularization helps but lr tuning is more effective. 12 is the champion.
