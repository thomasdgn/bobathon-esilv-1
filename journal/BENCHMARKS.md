# BENCHMARKS

Persistent cross-experiment comparison. Updated after each completed experiment.
CV strategy is identical across all rows: GroupKFold(n_splits=5), groups=patient_id,
same precomputed splits, skore.evaluate with iter(cv_splits).

## Summary table

| Stem | Model | Features | Mean RMSE | Fold std | vs Dummy | Hub report |
|---|---|---|---|---|---|---|
| `01_dummy` | DummyRegressor(mean) | 8 numeric (ignored) | 16.50 | 0.34 | baseline | [46521](https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/46521) |
| `02_ridge` | Ridge(alpha=1) + median impute + StandardScaler | 8 numeric | 10.437 | 0.197 | −36.7 % | [46906](https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/46906) |
| `03_hgbr` | HistGradientBoostingRegressor(max_iter=200, early_stopping=False) | 8 numeric, native NaN | 7.4336 | 0.1390 | −54.9 % | [47082](https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47082) |
| `04_hgbr_categories` | HGBR + OrdinalEncoder(cohort, gene) | 8 numeric + cohort + gene | 7.4314 | 0.1207 | −55.0 % | [47381](https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47381) |
| `05_hgbr_duration` ⭐ | HGBR + years_since_diagnosis | 8 numeric + years_since_dx | **7.3751** | 0.1430 | −55.3 % | [47400](https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47400) |
| `06_hgbr_missing` | HGBR + binary missing indicators (off, on, ledd) | 8 numeric + 3 indicators | 7.4336 | 0.1390 | −54.9 % | [47412](https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47412) |
| `07_hgbr_combined` | HGBR + years_since_diagnosis + OrdinalEncoder(cohort, gene) | 9 numeric + cohort + gene | 7.3819 | 0.1323 | −55.3 % | [47430](https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47430) |

⭐ Best CV result on fixed GroupKFold-5 splits.

### Robustness check (shuffled GroupKFold, n_splits=5, shuffle=True, random_state=42)

Both arms use **identical** shuffled patient fold assignments; all five fold val-patient
sets are confirmed identical between the two arms. This is a second independent CV split
on the same training data — NOT an untouched holdout. Paired differences measure
split-to-split consistency only; no significance claims are made.

| Stem | Config | Mean RMSE | Fold std | Hub report |
|---|---|---|---|---|
| `08_hgbr_robust_baseline` | 03_hgbr config, shuffled seed=42 | 7.4380 | 0.1910 | [47443](https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47443) |
| `09_hgbr_robust_candidate` | 05_duration config, shuffled seed=42 | 7.3678 | 0.1902 | [47461](https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47461) |

## Per-fold RMSE detail

### Fixed GroupKFold (deterministic, same splits as 01–07)

| Stem | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Std (ddof=1) |
|---|---|---|---|---|---|---|---|
| `01_dummy` | 16.585 | 16.880 | 15.948 | 16.470 | 16.613 | 16.499 | 0.343 |
| `02_ridge` | 10.441 | 10.758 | 10.307 | 10.250 | 10.427 | 10.437 | 0.197 |
| `03_hgbr` | 7.46895 | 7.58634 | 7.28006 | 7.29680 | 7.53580 | 7.433591 | 0.139028 |
| `04_hgbr_categories` | 7.47030 | 7.56339 | 7.26615 | 7.35112 | 7.50626 | 7.431442 | 0.120716 |
| `05_hgbr_duration` | 7.39538 | 7.53804 | 7.19850 | 7.26301 | 7.48056 | 7.375098 | 0.143036 |
| `06_hgbr_missing` | 7.46895 | 7.58634 | 7.28006 | 7.29680 | 7.53580 | 7.433591 | 0.139028 |
| `07_hgbr_combined` | 7.40200 | 7.52939 | 7.19661 | 7.30928 | 7.47245 | 7.381945 | 0.132252 |

### Shuffled GroupKFold (seed=42 robustness check, 08 vs 09 paired)

| Stem | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Std (ddof=1) |
|---|---|---|---|---|---|---|---|
| `08_hgbr_robust_baseline` | 7.11668 | 7.51180 | 7.41637 | 7.55613 | 7.58894 | 7.437984 | 0.190985 |
| `09_hgbr_robust_candidate` | 7.05303 | 7.46042 | 7.32988 | 7.46807 | 7.52762 | 7.367803 | 0.190191 |
| **Paired diff (09−08)** | −0.06365 | −0.05138 | −0.08649 | −0.08806 | −0.06132 | **−0.07018** | — |

All fold values are measured from `report.metrics.rmse(aggregate=None)` and
cross-checked against the skore aggregate mean (abs diff < 1e-4). Paired diff
mean = −0.07018: candidate reduces RMSE on all 5 folds under the shuffled split.

## Submission CSVs

| File | Rows | Nonfinite | Range |
|---|---|---|---|
| `submissions/01_dummy.csv` | 11 013 | 0 | 37.47 (constant) |
| `submissions/02_ridge.csv` | 11 013 | 0 | 2.05 – 101.27 |
| `submissions/03_hgbr.csv` | 11 013 | 0 | 2.77 – 90.29 |
| `submissions/04_hgbr_categories.csv` | 11 013 | 0 | 3.12 – 87.46 |
| `submissions/05_hgbr_duration.csv` | 11 013 | 0 | 0.69 – 90.23 |
| `submissions/06_hgbr_missing.csv` | 11 013 | 0 | 2.77 – 90.29 |
| `submissions/07_hgbr_combined.csv` | 11 013 | 0 | 1.60 – 89.85 |

All CSVs: header `Index,target`, Index order matches `sample_submission.csv` exactly,
zero duplicate IDs, zero train/test patient overlap confirmed.
Experiments 08 and 09 are robustness-check CV only — no submission CSVs produced.

## Key findings

- **Ridge vs Dummy:** Linear additive signal from the 8 numeric features (especially
  `off` at Pearson 0.89 and `on` at 0.69) cuts RMSE by 36.7 %. Median imputation
  lets Ridge use all columns despite 30–79 % missingness in some features.
- **HGBR vs Ridge (experiments 02 and 03):** HGBR (03) improves RMSE on every fold
  (7.28–7.59 vs 10.25–10.76 for Ridge). These two experiments differ in both the
  estimator class and the missing-value strategy (native NaN in HGBR vs median
  imputation in Ridge); the contributions of each change are confounded and cannot
  be separated from this pair alone. Ablations 04–06 vary the feature set within
  HGBR only; they do not address the estimator/imputation confound from 02 vs 03.
- **Feature ablations (04–07): all within HGBR, fixed GroupKFold-5 splits, vs 03_hgbr = 7.433591:**
  - `04_hgbr_categories` (cohort + gene via OrdinalEncoder): RMSE 7.431442, Δ = −0.002 (~0.03 %)
  - `05_hgbr_duration` (years_since_diagnosis): RMSE 7.375098, Δ = **−0.059** (−0.79 %)
  - `06_hgbr_missing` (binary indicators off/on/ledd): RMSE 7.433591, Δ = 0.000 (no measured benefit in this configuration; does not generalise to other estimators or feature sets)
  - `07_hgbr_combined` (duration + categories): RMSE 7.381945, Δ = −0.052 (slightly worse than 05 alone; difference is small relative to fold variation; cause undetermined from this data)
- **Best on fixed splits:** `05_hgbr_duration` at RMSE 7.3751 — 55.3 % below dummy.
  Selection is on the same partition used for all ablations; ranking may differ on
  another partition.
- **Robustness check (shuffled seed=42, paired 08 vs 09):** Both arms use the same
  training patients with fold boundaries reassigned by a different shuffle seed.
  The candidate (05 config) has lower RMSE than the baseline (03 config) on all 5
  paired folds (mean paired diff −0.070). The direction of improvement is the same
  as on the fixed splits. This is a second CV partition of the same training data,
  not an independent holdout; both partitions share the same patients and training
  corpus. The higher fold std on the shuffled partition (0.190 vs 0.139 for the fixed
  splits) is a property of this particular patient grouping, not a measure of
  reliability. No statistical significance claim is made from two CV partitions.
- **Limitations:**
  - Ablation selections and combined experiment all evaluated on the same fixed CV
    partition. Selection is exploratory; a held-out final test was not performed.
  - OrdinalEncoder assigns arbitrary lexicographic integer codes; the -2 sentinel
    is an ordinary integer split point, not HGBR's native NaN mechanism.
  - The robustness check uses the same training data with changed group boundaries.
    Paired fold differences measure split-to-split consistency, not independent
    replication. No significance claims are made.
- **Next directions (not pursued in this batch):** native categorical support via
  HGBR's `categorical_features` parameter (avoids arbitrary ordinal codes); tuning
  `max_iter` or `learning_rate` around the best feature set.
