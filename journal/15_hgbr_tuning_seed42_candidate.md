# 15_hgbr_tuning_seed42_candidate — 12-config on shuffled GroupKFold(seed=42) vs 09 baseline

## Question / hypothesis

Does the RMSE improvement of experiment 12 (lr=0.05, max_iter=200) hold on the
shuffled seed=42 splits used in experiments 08/09? Is 12 the provisional champion
over the current best (05 RMSE 7.375098 fixed, 09 RMSE 7.367803 seed=42)?

## Motivation

Experiment 12 achieved the best fixed-split RMSE (7.342917), beating 05 (7.375098)
by Δ=−0.032181. To check consistency with the seed=42 partition, 12's configuration
is run on the same shuffled GroupKFold(n_splits=5, shuffle=True, random_state=42)
splits as 08/09. Paired fold differences against 09 reveal whether the tuning gain
(lr=0.05, iter=200) is consistent across both CV partitions.

Both CV partitions use the same training data; this is selection data, not an
independent holdout. No significance claims are made.

Sourcing strategy: prescribed by the batch (seed42 diagnostic for champion).

## Method

- Configuration: `HistGradientBoostingRegressor(learning_rate=0.05, max_iter=200,
  random_state=0, early_stopping=False)` — identical to experiment 12.
- Feature set: 05 feature set (9 numeric).
- Splitter: `GroupKFold(n_splits=5, shuffle=True, random_state=42)` — same as 08/09.
- Val-patient set pairing with 09 is verified by assertion (fold-for-fold same patients).
- No submission CSV — diagnostics only.
- Both fixed and seed42 CV splits are reused selection data; whichever is lower
  across both is the provisional champion. No causal attribution.

## Paired fold results vs 09

| Fold | 15 (12-config, seed=42) | 09 (05-config, seed=42) | Paired diff (15−09) |
|---|---|---|---|
| 0 | 7.046954 | 7.053030 | −0.006076 |
| 1 | 7.430126 | 7.460417 | −0.030291 |
| 2 | 7.314448 | 7.329882 | −0.015434 |
| 3 | 7.437987 | 7.468069 | −0.030082 |
| 4 | 7.512762 | 7.527617 | −0.014855 |
| **Mean** | **7.348455** | **7.367803** | **−0.019348** |
| Std (ddof=1) | 0.182854 | 0.190191 | — |

12-config beats 09 on all 5 folds. Mean paired diff = −0.019.

## Champion decision

| Partition | 12-config | Prior best | Δ |
|---|---|---|---|
| Fixed splits (03-07 base) | 7.342917 | 7.375098 (05) | −0.032 |
| Seed=42 splits | 7.348455 | 7.367803 (09) | −0.019 |

12-config (lr=0.05, max_iter=200) is the provisional champion on both CV
partitions. Both partitions are reused selection data, not independent holdout.
No statistical significance claims. Provisional champion selected balancing
both scores.

## Status

- **State:** done
- **Approved by user on:** current session (user-authorized batch)
- **Headline result:** RMSE 7.348455 (fold std 0.182854, seed=42 GroupKFold). Paired mean diff vs 09 = −0.019 (12-config better on all 5 folds). Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/48025
- **Implication for next iteration:** 12-config (lr=0.05, iter=200) is provisional champion on both CV partitions. OOF residual diagnostics show systematic target-range bias; error-driven feature engineering is the priority.
