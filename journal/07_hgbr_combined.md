# 07_hgbr_combined — HGBR + years_since_diagnosis + cohort + gene

## Question / hypothesis

Combining the two individually useful additions from ablations 04 and 05
(categorical features `cohort`/`gene` and derived `years_since_diagnosis`)
will produce lower CV RMSE than either alone, while the missingness
indicators from 06 are excluded because they gave zero improvement.

## Motivation

Ablation results on fixed GroupKFold(n_splits=5, patient-grouped) splits:

| Experiment | Mean RMSE | Delta vs 03_hgbr |
|---|---|---|
| `03_hgbr` | 7.433591 | baseline |
| `04_hgbr_categories` | 7.431442 | −0.002149 (marginal, ~0.03 %) |
| `05_hgbr_duration` | 7.375098 | −0.058493 (clear, −0.79 %) |
| `06_hgbr_missing` | 7.433591 | 0.000000 (no benefit — excluded) |

`years_since_diagnosis` is the strongest single addition; `cohort`/`gene`
adds a small marginal delta. Combining them may compound the benefits.
Adaptive selection: 06 is excluded because its measured RMSE is identical
to the baseline to 6 decimal places in this configuration.

Sourcing strategy: adaptive combination of B4 + B5 results.

## Method

Keep the identical eight numeric features and identical HGBR configuration
(random_state=0, max_iter=200, early_stopping=False).
Add three columns:
- `years_since_diagnosis = age − age_at_diagnosis` (stateless arithmetic;
  NaN when `age_at_diagnosis` is NaN, handled natively by HGBR).
- `cohort` via fold-fitted `OrdinalEncoder` (no NaN; lexicographic integer
  codes).
- `gene` via fold-fitted `OrdinalEncoder` (32 % NaN → `encoded_missing_value=-2`
  integer sentinel; HGBR splits on -2 as an ordinary integer).

Pipeline: `ColumnTransformer` (passthrough nine numeric, OrdinalEncoder
two categorical) → `HistGradientBoostingRegressor`.

Same GroupKFold(n_splits=5) patient splits as 03..06, passed as
`iter(cv_splits)` to `skore.evaluate`. Metrics read via
`report.metrics.rmse(aggregate=None)`.

## Risks

- The marginal delta from categories (0.002 RMSE) may add noise rather
  than signal when combined; the combined result may not be better than
  05 alone.
- All three ablations (04, 05, 06) and the combined experiment (07) were
  evaluated on the same fixed CV split. Selection is exploratory; the
  ranking of configurations on this partition may differ on other partitions.
- Limitation of OrdinalEncoder: same arbitrary-order caveat as 04.

## Outcome note

Measured mean RMSE 7.381945, vs 03_hgbr 7.433591 (Δ = −0.052) and vs
05_hgbr_duration 7.375098 (+0.007). The combined result is slightly worse
than 05 alone on this fixed CV partition. The difference between 07 and 05
(0.007 RMSE) is small relative to fold-to-fold variation; both observations
are on the same single partition and cannot determine whether the difference
reflects added noise from categories or CV noise. This is a within-HGBR
feature ablation; no estimator comparison is made here.

## Status

- **State:** done
- **Approved by user on:** current session
- **Headline result:** RMSE 7.381945 (fold std 0.132252, GroupKFold-5 fixed). Slightly worse than 05 alone. See BENCHMARKS.md.
- **Implication for next iteration:** 05_hgbr_duration selected as robustness
  candidate; 07 not taken forward to shuffled-split comparison.
