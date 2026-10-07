# 14_hgbr_tuning_lr005_iter400_msl40 — HGBR tuning: lr=0.05, max_iter=400, min_samples_leaf=40

## Question / hypothesis

Does combining slower learning (lr=0.05, 400 iterations) with stronger
leaf regularization (min_samples_leaf=40) yield a further RMSE improvement
over any of experiments 11–13 individually?

## Motivation

Experiments 11–12 explore the learning rate/iteration tradeoff; experiment
13 explores tree complexity regularization. This experiment combines the
best elements: slower, smoother learning with more robust leaves. If the
previous experiments show lr=0.05/iter=400 wins over the baseline AND
min_samples_leaf=40 wins independently, combining them is the natural
next step. The combination is pre-declared here (not post-hoc).

Note: this config is declared before 11–13 results are known. If the
direction of 11–13 does not support this combination, the experiment still
runs (pre-commitment ensures no selection bias from intermediate results).

Feature set: **05 feature set** (9 numeric) — same as 11-13, per pre-declared winner rule (05 wins).
Sourcing strategy: B7 carry-forward (compact tuning batch — combined).

## Method

Feature set: pre-declared winner of 10 vs 05 (same as 11–13).
Estimator: `HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400,
min_samples_leaf=40, random_state=0, early_stopping=False)`. l2_regularization
at default (0.0) — isolating the interaction of lr/iter with leaf size.

CV: identical `GroupKFold(n_splits=5)` fixed patient splits as 03–13.

## Risks

- Over-regularization if min_samples_leaf=40 is too aggressive combined
  with slower learning. Both levers reduce model flexibility; their
  interaction may cause underfitting.
- 400 iterations with a larger min_samples_leaf is slower. Accepted.

## Outcome

Mean RMSE 7.354230, std 0.136884 (fixed GroupKFold-5).
05_duration ref: 7.375098 (Δ = −0.020868).
The combination is better than each component alone at the lr=0.1 base,
but 12 (lr=0.05, iter=200, no msl change) is still better (7.342917).
The larger min_samples_leaf slightly hurts when combined with lr=0.05.

## Status

- **State:** done
- **Approved by user on:** current session (user-authorized batch)
- **Headline result:** RMSE 7.354230 (fold std 0.136884, fixed GroupKFold-5). Δ=−0.021 vs 05. Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/48001
- **Implication for next iteration:** 12 remains champion. Combined lr+msl does not outperform lr alone.
