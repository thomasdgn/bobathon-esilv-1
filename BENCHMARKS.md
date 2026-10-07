# BENCHMARKS

All results use `GroupKFold(n_splits=5)` with `groups=patient_id` (zero patient overlap per fold).
RMSE is reported via `report.metrics.rmse(aggregate=None)` with mean and std computed from fold values (ddof=1).
`IDs` column refers to whether patient_id is excluded from features (always Yes).
Fixed splits = deterministic GroupKFold(n_splits=5, shuffle=False).
Seed=42 splits = GroupKFold(n_splits=5, shuffle=True, random_state=42).

## Fixed-split results (GroupKFold deterministic, comparable across 03-07, 10-14)

| Exp | Short name | Features | Model config | Mean RMSE | Std (ddof=1) | IDs | Hub key |
|---|---|---|---|---|---|---|---|
| 01 | dummy | — | DummyRegressor(mean) | ≈16.50 | 0.34 | Yes | — |
| 02 | ridge | 8 numeric | Ridge(alpha=1) + median impute + scale | ≈10.44 | 0.20 | Yes | 46906 |
| 03 | hgbr | 8 numeric | HGBR(rs=0, iter=200, es=False) | 7.433591 | 0.139028 | Yes | 47082 |
| 04 | hgbr_categories | 8 numeric + cohort+gene (OrdinalEncoder) | HGBR(rs=0, iter=200, es=False) | 7.431442 | 0.120716 | Yes | 47381 |
| **05** | **hgbr_duration** | **9 numeric (+duration)** | **HGBR(rs=0, iter=200, es=False)** | **7.375098** | **0.143036** | **Yes** | **47400** |
| 06 | hgbr_missing | 8+3 missing indicators | HGBR(rs=0, iter=200, es=False) | 7.433591 | 0.139028 | Yes | 47412 |
| 07 | hgbr_combined | 9 numeric + OrdinalEnc categories | HGBR(rs=0, iter=200, es=False) | 7.381945 | 0.132252 | Yes | 47430 |
| 10 | hgbr_native_categories | 9 numeric + native Categorical | HGBR(from_dtype, rs=0, iter=200, es=False) | 7.376945 | 0.131788 | Yes | 47912 |
| 11 | hgbr_tuning_lr005_iter400 | 9 numeric | HGBR(lr=0.05, iter=400, rs=0, es=False) | 7.359378 | 0.129610 | Yes | 47926 |
| 13 | hgbr_tuning_msl40_l2reg01 | 9 numeric | HGBR(msl=40, l2=0.1, rs=0, iter=200, es=False) | 7.363917 | 0.138674 | Yes | 47995 |
| 14 | hgbr_tuning_lr005_iter400_msl40 | 9 numeric | HGBR(lr=0.05, iter=400, msl=40, rs=0, es=False) | 7.354230 | 0.136884 | Yes | 48001 |
| **12** | **hgbr_tuning_lr005_iter200** | **9 numeric** | **HGBR(lr=0.05, iter=200, rs=0, es=False)** | **7.342917** | **0.137556** | **Yes** | **47964** |

**Current fixed-split champion: 12_hgbr_tuning_lr005_iter200 — RMSE 7.342917**

## Seed=42 splits (GroupKFold shuffle=True, random_state=42 — 08, 09, 15)

| Exp | Short name | Features | Model config | Mean RMSE | Std (ddof=1) | Hub key |
|---|---|---|---|---|---|---|
| 08 | hgbr_robust_baseline | 8 numeric | HGBR(rs=0, iter=200, es=False) | 7.437984 | 0.190985 | 47443 |
| 09 | hgbr_robust_candidate | 9 numeric (+duration) | HGBR(rs=0, iter=200, es=False) | 7.367803 | 0.190191 | 47461 |
| **15** | **hgbr_tuning_seed42_candidate** | **9 numeric (+duration)** | **HGBR(lr=0.05, iter=200, rs=0, es=False)** | **7.348455** | **0.182854** | **48025** |

**Current seed=42 champion: 15_hgbr_tuning_seed42_candidate — RMSE 7.348455**

### Paired fold comparison 15 vs 09 (same seed=42 splits, fold-for-fold)

| Fold | 15 (12-config) | 09 (05-config) | Diff (15−09) |
|---|---|---|---|
| 0 | 7.046954 | 7.053030 | −0.006076 |
| 1 | 7.430126 | 7.460417 | −0.030291 |
| 2 | 7.314448 | 7.329882 | −0.015434 |
| 3 | 7.437987 | 7.468069 | −0.030082 |
| 4 | 7.512762 | 7.527617 | −0.014855 |
| **Mean** | **7.348455** | **7.367803** | **−0.019348** |

12-config beats 09 (05-config) on all 5 folds. Both are reused selection data.

## Provisional champion (balancing both partitions)

**12_hgbr_tuning_lr005_iter200** (HGBR, lr=0.05, max_iter=200, all other params default):
- Fixed splits: 7.342917 (Δ=−0.032 vs 05)
- Seed=42 splits: 7.348455 (Δ=−0.019 vs 09)

Both partitions are reused selection data, not independent holdout. No significance
claims. Provisional champion chosen by consistent improvement across both.

## OOF residual diagnostics for champion (exp 12, fixed splits)

Subgroup conditioning (OOF only; no in-sample residuals; no causal attribution):

| Subgroup | Count | Residual mean | Residual std | OOF RMSE |
|---|---|---|---|---|
| Cohort A | 39636 | +0.086 | 7.448 | 7.448 |
| Cohort B | 4954 | −0.578 | 6.423 | 6.449 |
| off not missing | 25677 | −0.004 | 6.630 | 6.630 |
| off missing | 18913 | +0.034 | 8.214 | 8.214 |
| on not missing | 31370 | +0.034 | 7.933 | 7.933 |
| on missing | 13220 | −0.042 | 5.709 | 5.709 |
| gene not null | 30158 | −0.101 | 7.342 | 7.342 |
| gene null | 14432 | +0.248 | 7.344 | 7.348 |
| Target Q1 (low, 0–24) | 11225 | −2.705 | 5.134 | 5.803 |
| Target Q2 | 11122 | −3.033 | 6.127 | 6.836 |
| Target Q3 | 11121 | +0.297 | 6.804 | 6.810 |
| Target Q4 (high, 54–110) | 11122 | +5.513 | 7.666 | 9.442 |

Key observations (no causal attribution):
- Systematic under-prediction at low target (Q1–Q2), over-prediction at high (Q4).
- off-missing visits have higher OOF RMSE (8.21 vs 6.63) and higher mean target (44.4 vs 32.4).
- on-missing visits have lower OOF RMSE (5.71 vs 7.93) and lower mean target (21.3 vs 44.3).
- Cohort B is a smaller group (11 %) with lower RMSE than Cohort A.

## Robustness notes

- Results on fixed splits (03-07, 10-14) and seed=42 splits (08, 09, 15) are NOT directly
  comparable: different patient-to-fold assignments.
- Both CV partitions use the same training data. A truly held-out test set would be required
  to confirm generalisation.
- No statistical significance claims are made from CV comparisons.

## Backlog items (open)

| # | Item | Source |
|---|---|---|
| B6 | ~~Use HGBR native categorical~~ | done as 10 — no gain vs 05 |
| B7 | ~~Tune max_iter/learning_rate around 05~~ | done as 11-14; 12 wins |
| B8 | Error-driven features: encode off/on missingness pattern more expressively (visit-level or patient-level) | OOF diagnostics: off_missing has higher RMSE |
| B9 | Target-range features: investigate disease severity proxies for extreme scores | OOF diagnostics: Q1/Q4 residual asymmetry |
| B10 | lr=0.05 with iter=300-500 (finer search around 12) | 11 (iter=400) is worse than 12 (iter=200) — diminishing returns signal |
