# 11_hgbr_tuning_lr005_iter400 — HGBR tuning: lr=0.05, max_iter=400, feature set from winner of 10 vs 05

## Question / hypothesis

Does reducing the learning rate to 0.05 and doubling max_iter to 400
(same total boosting "budget") lower CV RMSE relative to the default
lr=0.1, max_iter=200 used in 05–10?

## Motivation

The canonical HGBR tuning strategy is to lower the learning rate and
compensate with more iterations so the total shrinkage budget remains
comparable. With lr=0.1 and 200 trees (05 baseline), each tree contributes
0.1 of its leaf values. With lr=0.05 and 400 trees, each tree contributes
0.05 — same total path length but finer steps, which can reduce overfitting
on the training folds while improving generalisation on the validation fold.
This is a well-documented first tuning step for HGBR/GBM (sklearn docs §
histogram_based_gradient_boosting, Backlog row B7).

Feature set: **05 feature set** (9 numeric: the 8 original + years_since_diagnosis).
Pre-declared rule: winner of 10 vs 05 → 05 wins (RMSE 7.375098 vs 7.376945);
05 feature set applied to all tuning experiments 11-14.

Sourcing strategy: B7 (Backlog row).

## Method

Feature set: determined by the 10 vs 05 result (pre-declared above).
Estimator: `HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400,
random_state=0, early_stopping=False)`. All other parameters at sklearn 1.9.1
defaults (max_leaf_nodes=31, max_depth=None, min_samples_leaf=20,
l2_regularization=0.0).

CV: identical `GroupKFold(n_splits=5)` fixed patient splits as 03–10.

## Risks

- 400 iterations may overfit if 200 was already past the optimum for this
  dataset with lr=0.1. Early stopping is disabled to keep the evaluation
  deterministic; if RMSE rises vs 10/05 it indicates overfitting.
- Runtime will roughly double (400 vs 200 iterations). Accepted.

## Outcome

Mean RMSE 7.359378, std 0.129610 (fixed GroupKFold-5).
05_duration ref: 7.375098 (Δ = −0.015720).
lr=0.05 with same total budget (lr × iter = 20) as baseline reduces fold RMSE.
Experiment 12 (lr=0.05, iter=200, less budget) achieves lower RMSE still.

## Status

- **State:** done
- **Approved by user on:** current session (user-authorized batch)
- **Headline result:** RMSE 7.359378 (fold std 0.129610, fixed GroupKFold-5). Δ=−0.016 vs 05. Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47926
- **Implication for next iteration:** lr=0.05 improves vs lr=0.1 baseline. Experiment 12 (same lr, fewer iter) wins even more.
