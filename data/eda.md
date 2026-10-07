# EDA: Parkinson's Disease True-OFF Motor Score

Exploratory data analysis run on 2026-10-07. Raw data: four CSVs in `data/`.
Raw data files are read-only; this file and the HTML reports are the only outputs.

Interactive reports:
- [data/eda_x_train.html](eda_x_train.html)
- [data/eda_visits.html](eda_visits.html)
- [data/eda_x_test.html](eda_x_test.html)

---

## Dataset at a glance

| Table | Rows | Columns | Notes |
|---|---|---|---|
| `X_train.csv` | 44 590 | 12 | Feature table for training |
| `y_train.csv` | 44 590 | 2 | Target (`Index`, `target`) |
| `visits` (joined) | 44 590 | 13 | `X_train` ⋈ `y_train` on `Index` — the modelling table |
| `X_test.csv` | 11 013 | 12 | Feature table for prediction |
| `sample_submission.csv` | 11 013 | 2 | Template: `Index`, `target` |

**Join integrity:** `Index` is unique in both `X_train` (0 duplicates) and `y_train` (0 duplicates). The inner join on `Index` yields exactly 44 590 rows with 0 unmatched rows on either side — perfect 1:1 coverage.

**Submission alignment:** `sample_submission.csv` and `X_test.csv` share exactly the same 11 013 `Index` values — zero entries in one that are absent from the other.

**Patients:** 5 576 distinct patients in train, 1 395 in test. **Overlap: 0** — the holdout is fully patient-disjoint, matching the challenge spec.

**Visits per patient (train):** mean 8.0, median 7, max 12. Each patient has up to 12 longitudinal visits at irregular intervals.

**Task:** regression — predict `target` (debiased true OFF MDS-UPDRS total motor score) for every visit. Grouping variable: `patient_id`. Evaluation metric: visit-level RMSE.

---

## Per-column findings

All 12 feature columns are present in both train and test with identical names — **no train/test column mismatch**.

| Column | Dtype | Train missing | Train miss % | Test miss % | n_unique (train) | Notes |
|---|---|---|---|---|---|---|
| `Index` | Int64 | 0 | 0 % | 0 % | 44 590 | Row identifier — **never feed to model** |
| `patient_id` | String | 0 | 0 % | 0 % | 5 576 | Group column — **never feed to model as a feature** |
| `cohort` | String | 0 | 0 % | 0 % | 2 | Low-cardinality categorical (values: A, B) |
| `sexM` | Int64 | 0 | 0 % | 0 % | 2 | Binary (0/1) |
| `gene` | String | 14 432 | **32.4 %** | 32.0 % | 4 | High missingness; categorical with 4 levels |
| `age_at_diagnosis` | Float64 | 2 319 | **5.2 %** | 4.6 % | 563 | Low missingness |
| `age` | Float64 | 0 | 0 % | 0 % | 1 082 | Always present |
| `ledd` | Float64 | 16 338 | **36.6 %** | 38.6 % | 1 320 | Levodopa equivalent daily dose — see cohort breakdown below |
| `time_since_intake_on` | Float64 | 20 680 | **46.4 %** | 47.8 % | 64 | Drug timing for ON assessment |
| `time_since_intake_off` | Float64 | 35 121 | **78.8 %** | 79.1 % | 178 | Drug timing for OFF assessment — **most-missing column** |
| `on` | Float64 | 13 220 | **29.6 %** | 31.2 % | 80 | Measured ON MDS-UPDRS motor score |
| `off` | Float64 | 18 913 | **42.4 %** | 40.8 % | 93 | Measured OFF MDS-UPDRS motor score |

### Joint missingness: `on` and `off` (train)

Computed directly from the data — not inferred from marginals:

| Condition | Count | % of train |
|---|---|---|
| `on` missing | 13 220 | 29.65 % |
| `off` missing | 18 913 | 42.42 % |
| **Both `on` and `off` missing** | **0** | **0.00 %** |

**Key finding:** every training visit has at least one of `on` or `off` present. There are no fully-unscored visits in train (both missing = 0). The same pattern holds in test (both missing = 0 out of 11 013).

The same computation applies to test: `on` missing 3 439 (31.23 %), `off` missing 4 494 (40.81 %), both missing 0.

### LEDD missingness by cohort (train)

| Cohort | Visits | LEDD missing | Miss % |
|---|---|---|---|
| A | 39 636 | 14 228 | 35.9 % |
| B | 4 954 | 2 110 | 42.6 % |

Both cohorts have substantial LEDD missingness. Cohort B has a somewhat higher rate (~7 pp more), but neither cohort is clean. This is consistent with LEDD being an optional clinical measurement rather than a systematic recording gap specific to one cohort.

**Top 3 most-missing columns:** `time_since_intake_off` (78.8 %), `time_since_intake_on` (46.4 %), `off` (42.4 %). `ledd` is fourth at 36.6 %.

No datetime columns were detected. Temporal structure is encoded implicitly through `age` (current age), `age_at_diagnosis`, and the intake-delay columns.

---

## Target distribution

- **Column:** `target` (true OFF MDS-UPDRS motor score)
- **Dtype:** Float64
- **Missing:** 0 % — the target is present for all 44 590 training visits
- **Range:** 0.0 – 109.5 (theoretical maximum 132; the synthetic data does not reach it)
- **Mean:** 37.5 | **Median:** 37.3 | **Std:** 16.5
- **IQR:** 25.6 – 49.3 (spread of 23.7 points)
- **Histogram (bin counts, ~11-point bins from 0 to 109.5):** 2 477, 5 641, 9 764, 10 703, 8 844, 5 203, 1 757, 180, 13, 8

The distribution is **approximately bell-shaped with a slight right tail** — the bulk of visits cluster in the 20–55 range, with a long tail up to ~110. The distribution is continuous (867 unique values). RMSE is a natural metric; no log transform is required at baseline.

The **standard deviation of ≈16.5** sets the dummy-mean reference: a model that predicts only the training mean achieves RMSE ≈ 16.5 on data drawn from the same distribution. Any model that uses the features must beat this floor on grouped held-out patients to demonstrate genuine learning.

---

## Structure signals

**Datetime columns:** none detected. No explicit date or timestamp column — temporal context comes from `age`, `age_at_diagnosis`, and intake-delay columns.

**High-cardinality / identifier columns:**

| Column | Unique ratio |
|---|---|
| `Index` | 1.000 — pure row ID |
| `patient_id` | 0.125 — 5 576 distinct patients across 44 590 visits |
| `ledd` | 0.030 — continuous numeric |
| `age` | 0.024 — continuous numeric |

`patient_id` repeats across multiple rows (mean 8 visits per patient). This is a **grouped panel dataset** — the correct CV strategy is `GroupKFold` with `groups=patient_id` to prevent any patient's visits from appearing on both sides of a fold boundary.

---

## Associations with `target`

Sorted by Pearson correlation (absolute value). All association values are descriptive statistics; they do not by themselves prove or disprove leakage — that requires reasoning about how each column is generated and whether its value is knowable at prediction time.

| Feature | Cramér V | Pearson r | Note |
|---|---|---|---|
| `off` | 0.495 | **0.886** | Strongest predictor — but **42 % missing** |
| `on` | 0.403 | **0.688** | Second-strongest — but **30 % missing** |
| `ledd` | 0.236 | 0.298 | Levodopa dose — **37 % missing** |
| `age` | 0.114 | 0.310 | Current age — always present |
| `age_at_diagnosis` | 0.051 | 0.133 | Diagnosis age — 5 % missing |
| `time_since_intake_off` | 0.092 | 0.008 | Cramér V non-trivial despite near-zero Pearson → non-linear relationship |
| `time_since_intake_on` | 0.177 | 0.0002 | Same pattern — Cramér V elevated, Pearson ≈ 0 |
| `cohort` | 0.090 | n/a | Categorical |
| `gene` | 0.052 | n/a | Categorical |
| `sexM` | 0.023 | −0.001 | Very weak |
| `Index` | 0.026 | −0.004 | Row-number artifact — no predictive value expected |
| `patient_id` | 0.028 | n/a | Group label artifact |

**Other strong pairwise links:**
- `age_at_diagnosis` ↔ `age`: Pearson 0.94 — near-collinear; `age − age_at_diagnosis` likely encodes disease duration more directly.
- `on` ↔ `off`: Pearson 0.87 — measured ON and OFF scores are correlated, as expected clinically.

### Leakage assessment

Leakage cannot be ruled out or confirmed by correlation alone. The assessment below is based on the data-generating process described in `docs/CONTEXT.md`:

- **`off` (Pearson 0.89 with `target`):** `off` is the *measured* (biased) OFF motor score; `target` is the *debiased* version. The challenge is precisely to learn this debiasing. `off` is a legitimate feature: it is available at visit time and carries signal the model should use. Its high correlation with `target` reflects shared biology, not leakage. However, since `off` is missing in 42 % of visits, the model cannot rely on it alone.
- **`Index`:** A row counter. Correlation with `target` is negligible (Pearson −0.004, Cramér V 0.026). Drop from feature set regardless.
- **`patient_id`:** A group label. Its Cramér V (0.028) is low on the full dataset, but patient effects may be real (each patient has a trajectory). It should be used as the CV grouping variable, not as a feature.
- **No implausibly perfect association (Pearson ≥ 0.99 or Cramér V ≥ 0.99) was detected.** This is a necessary check, not sufficient to clear all leakage risks.

---

## Train/test column differences

**Identical column sets** in both `X_train` and `X_test`. No leakage via extra columns in train and no missing columns in test.

---

## Modelling implications

| Finding | Implication |
|---|---|
| `patient_id` repeats; test holdout is by patient; join is 1:1 with full coverage | **Use `GroupKFold(n_splits=5)` with `groups=patient_id`** — random row splits would leak patient trajectories into validation folds and produce optimistically biased RMSE estimates |
| No datetime column | `TimeSeriesSplit` is not needed at baseline; `GroupKFold` is sufficient |
| Target is continuous, bell-shaped, std ≈ 16.5 | RMSE is the right primary metric; no transform needed at baseline; dummy-mean floor ≈ 16.5 |
| Every visit has at least one of `on`/`off` present (joint both-missing = 0) | No need to handle the case of a completely unscored visit — but individual `on` or `off` missingness still requires imputation |
| `off` has Pearson 0.89 with `target` but is 42 % missing; `on` is 30 % missing | Use both as features with an imputer (or a model with native missing-value support); missingness-indicator columns may add signal in later experiments |
| `time_since_intake_on/off` have near-zero Pearson but elevated Cramér V | Non-linear effect; tree-based models likely extract more signal than linear ones |
| `ledd`, `time_since_intake_on/off`, `off`, `on`, `gene` all have ≥ 30 % missing | Any learner must handle missing values explicitly (imputation or native support) |
| `age` and `age_at_diagnosis` are near-collinear (Pearson 0.94) | Consider `age − age_at_diagnosis` as a "years since diagnosis" derived feature in later experiments |
| `Index` and `patient_id` are identifiers | Drop them from the feature matrix |
| Std(target) ≈ 16.5 | Dummy-mean baseline RMSE ≈ 16.5; this is the floor every real model must beat |
| LEDD missingness: cohort A 35.9 %, cohort B 42.6 % | Both cohorts have substantial LEDD missingness; cohort B is slightly worse but neither is cleanly imputable |

---

## Open questions

1. **`cohort` values (A and B):** only 2 distinct values — what do they represent clinically? Cohort B is smaller (4 954 vs 39 636 visits) and has slightly higher LEDD missingness.
2. **Visit ordering:** is `Index` a monotonically increasing visit counter within a patient? If so, visit order could be recovered for temporal feature engineering.
3. **Rater ID:** `CONTEXT.md` mentions a `rater_id` column, but it is absent from the CSVs — possibly removed from the synthetic dataset.
4. **`gene` missingness mechanism:** is it missing at random, or does missingness correlate with cohort, age at diagnosis, or disease severity? This could determine whether the missingness indicator itself is informative.
