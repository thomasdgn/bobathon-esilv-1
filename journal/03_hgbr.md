# 03_hgbr

<!--
Design note for experiments/03_hgbr.py (same stem, one to one with
the script). Written and approved before creating the script.
-->

## Question / hypothesis

Does `HistGradientBoostingRegressor` (native NaN handling, no
imputation) beat Ridge (mean RMSE 10.437) on the same eight numeric
features under identical patient-grouped GroupKFold-5 CV? Does
preserving the missingness pattern as signal — rather than filling it
with the median — help the model?

## Motivation

- **Sourcing strategy:** user (lab guide Day 2 + explicit session request)
- **Source(s):**
  - `docs/DAY2.md` § 4 — prescribes HGBR with native NaN handling,
    `random_state=0`, `max_iter=200`; explicitly says "do NOT
    median-fill NaNs" because missingness is information.
  - EDA finding (`data/eda.md`): `time_since_intake_on/off` have
    near-zero Pearson but elevated Cramér V → non-linear effect that
    tree-based models can capture. `off` (42 % missing) and `on`
    (30 % missing) are the strongest predictors.
  - Backlog item B3: "Gradient-boosted trees (HistGradientBoosting)
    with native missing-value support."
- **Why this matters:** Two competing hypotheses — (a) the non-linear
  timing and missingness patterns add signal beyond a linear model;
  (b) the eight features are already mostly linear, and HGBR gains
  nothing. This experiment disambiguates.

## Method

- **Files touched:** `experiments/03_hgbr.py` only
- **Estimator:** `HistGradientBoostingRegressor(random_state=0, max_iter=200, early_stopping=False)`.
  `early_stopping=False` disables internal row-level validation split
  so all training rows in each fold are used for tree growth.
  Native NaN support: no imputation step, no `SimpleImputer`.
- **Feature columns used:** same eight numeric columns as `01_dummy`
  and `02_ridge`: `sexM`, `age_at_diagnosis`, `age`, `ledd`,
  `time_since_intake_on`, `time_since_intake_off`, `on`, `off`.
  No categoricals. IDs (`Index`, `patient_id`) never enter the feature
  matrix.
- **Cross-validation:** identical precomputed `GroupKFold(n_splits=5)`
  splits on `patient_id` as `01_dummy` and `02_ridge`. Passed as
  `iter(cv_splits)` to `skore.evaluate`.
- **Metric extraction:** `report.metrics.rmse(aggregate=None)` for
  five fold values; cross-checked against `report.metrics.rmse()`
  aggregate mean/std.
- **Hub key:** `03_hgbr` in project `bobathon-esilv`, workspace `teamATN`.
- **Submission file:** `submissions/03_hgbr.csv` (refit full model on
  all training rows, predict X_test, header `Index,target`).
- **Out of scope:** hyperparameter sweep, categorical features, feature
  engineering.

## Risks / things that could invalidate the result

- `early_stopping=False` is required: with `early_stopping="auto"` or
  `True`, sklearn carves out an internal validation set from each
  training fold, leaking row ordering into the stopping criterion and
  reducing the effective training set size.
- GroupKFold splits must be identical to prior experiments for direct
  RMSE comparison. The same deterministic `GroupKFold(n_splits=5)`
  call on the same `X`, `y`, `groups` produces the same splits.

## Acceptance criteria

1. Experiment file runs to completion without errors.
2. Hub report published under key `03_hgbr`; URL recorded here.
3. `submissions/03_hgbr.csv` written: 11 013 rows, header `Index,target`,
   no nonfinite values, Index aligns with `sample_submission.csv`.
4. Per-fold RMSE reported for all five folds, mean and fold std logged.
5. Comparison vs dummy (≈16.50) and Ridge (≈10.437) logged.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** Mean RMSE 7.43359 (fold std 0.13903, GroupKFold-5, patient-grouped)
  - Fold 0: 7.46895
  - Fold 1: 7.58634
  - Fold 2: 7.28006
  - Fold 3: 7.29680
  - Fold 4: 7.53580
  - Runtime: ~7s
  - Beats dummy by 54.9 % (16.50 → 7.43); beats Ridge by 28.8 % (10.44 → 7.43); submissions/03_hgbr.csv validated (11 013 rows, no nonfinite, range 2.77–90.29)
- **Hub URL:** https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47082
