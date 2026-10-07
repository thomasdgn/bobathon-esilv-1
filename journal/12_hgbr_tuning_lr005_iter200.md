# 12_hgbr_tuning_lr005_iter200 — HGBR tuning: lr=0.05, max_iter=200 (less total budget)

## Question / hypothesis

Does a reduced learning rate (lr=0.05) with the same number of iterations
(max_iter=200) improve CV RMSE, or does the reduced budget (fewer total
leaf updates) hurt performance relative to lr=0.1, max_iter=200?

## Motivation

Experiment 11 tests lr=0.05 with doubled iterations (same total budget).
This experiment tests the complementary question: what happens with the
reduced learning rate but the SAME iteration count? This isolates the
effect of the learning rate itself from the effect of more trees. If 12
matches or beats 11, it suggests the model is not constrained by the
number of trees at 200; if 12 is worse than 11 but better than the
lr=0.1 baseline, the benefit comes from the slower learning rate AND
additional trees. The comparison between 11 and 12 (same budget vs less
budget) is informative without being part of an unbounded sweep.

Feature set: **05 feature set** (9 numeric) — same as 11, per pre-declared winner rule (05 wins).

Sourcing strategy: B7 carry-forward (compact tuning batch).

## Method

Feature set: pre-declared winner of 10 vs 05 (same as 11).
Estimator: `HistGradientBoostingRegressor(learning_rate=0.05, max_iter=200,
random_state=0, early_stopping=False)`. All other parameters at sklearn 1.9.1
defaults.

CV: identical `GroupKFold(n_splits=5)` fixed patient splits as 03–11.

## Risks

- With fewer total "steps" (lr=0.05 × 200 = 10 units vs lr=0.1 × 200 = 20
  units of total leaf scaling), underfitting is possible if the baseline
  200 iterations with lr=0.1 was near optimal depth.
- If 12 is worse than the lr=0.1 baseline but better than dummy, it still
  provides useful information: the model is "budget-limited" at 200 trees.

## Outcome

Mean RMSE 7.342917, std 0.137556 (fixed GroupKFold-5).
05_duration ref: 7.375098 (Δ = −0.032181). Best fixed-split result.
Reducing lr to 0.05 without compensating with more iterations still improves RMSE —
the model was budget-sufficient at 200 iterations; slower lr helps.
See 15 for seed42 consistency check.

## Status

- **State:** done
- **Approved by user on:** current session (user-authorized batch)
- **Headline result:** RMSE 7.342917 (fold std 0.137556, fixed GroupKFold-5). Δ=−0.032 vs 05. **Best fixed-split result.** Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47964
- **Implication for next iteration:** Provisional fixed-split champion. Better than all prior configs. Confirmed consistent on seed42 splits (exp 15).
