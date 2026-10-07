# 02_ridge

<!--
Design note for experiments/02_ridge.py (same stem, one to one with
the script). Written and approved before creating the script.
-->

## Question / hypothesis

Does a Ridge regression on the eight numeric features (with median
imputation and standard scaling) beat the dummy-mean floor of RMSE ≈
16.50 under patient-grouped GroupKFold-5 cross-validation?

## Motivation

- **Sourcing strategy:** user (lab guide Day 2 + explicit session request)
- **Source(s):**
  - `docs/DAY2.md` § 1–3 — prescribes Ridge with median imputation as
    the first non-trivial linear baseline; § 3 mandates GroupKFold for
    honest patient-grouped CV.
  - EDA finding (`data/eda.md`): `off` (Pearson 0.89) and `on` (0.69)
    are the strongest predictors; `age` (0.31) and `ledd` (0.30) also
    add signal. Median imputation lets Ridge consume those despite high
    missingness.
  - Backlog item B1: "Use `off` + `on` with imputation and `age` as
    features in a simple linear model."
- **Why this matters:** Establishes the first real-model RMSE and
  quantifies how much linear additive signal the numeric features carry
  before any feature engineering or non-linear model.

## Method

- **Files touched:** `experiments/02_ridge.py` only
- **Estimator:** `sklearn.pipeline.make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))`
  — all three steps fitted inside each fold to prevent imputation
  leakage.
- **Feature columns used:** same eight numeric columns as `01_dummy`:
  `sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`,
  `time_since_intake_off`, `on`, `off`. No categoricals; no IDs.
- **Cross-validation:** `GroupKFold(n_splits=5)` with `groups=patient_id`.
  Splits pre-computed as `list(GroupKFold(n_splits=5).split(X, y,
  groups=groups))` and passed as `iter(cv_splits)` to `skore.evaluate`,
  matching the `01_dummy` pattern exactly.
- **Hub key:** `02_ridge` in project `bobathon-esilv`, workspace `teamATN`.
- **Submission file:** `submissions/02_ridge.csv` (refit full pipeline on
  all training rows, predict X_test, header `Index,target`).
- **Out of scope:** hyperparameter sweep, feature engineering, categoricals.

## Risks / things that could invalidate the result

- Imputation must happen inside the pipeline, not before `split()`, to
  avoid leaking the fold-validation median into training statistics.
  `make_pipeline` guarantees this.
- The same precomputed `cv_splits` as `01_dummy` must be used so fold
  boundaries are identical and RMSEs are directly comparable.

## Acceptance criteria

1. Experiment file runs to completion without errors.
2. Hub report published under key `02_ridge`; URL recorded here.
3. `submissions/02_ridge.csv` written: 11 013 rows, header `Index,target`,
   no nonfinite values, Index aligns with `sample_submission.csv`.
4. Mean RMSE < 16.50 (must beat the dummy floor).

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** Mean RMSE 10.43654 (fold std 0.19677, GroupKFold-5, patient-grouped)
  - Fold 0: 10.44143
  - Fold 1: 10.75767
  - Fold 2: 10.30669
  - Fold 3: 10.25008
  - Fold 4: 10.42683
  - Runtime: ~2s
  - Beats dummy by 36.7 % (16.50 → 10.44); submissions/02_ridge.csv validated (11 013 rows, no nonfinite, range 2.05–101.27)
- **Hub URL:** https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/46906
