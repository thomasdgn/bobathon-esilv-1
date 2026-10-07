# 08_hgbr_robust_baseline — 03_hgbr on shuffled GroupKFold(n_splits=5, shuffle=True, random_state=42)

## Question / hypothesis

Does the CV performance of the `03_hgbr` configuration change substantially
when patient group boundaries are reassigned by a different shuffle seed?

## Motivation

The fixed deterministic GroupKFold used in experiments 03–07 produces one
specific partition of the 5 576 training patients into five groups.
Different group boundaries could in principle yield a noticeably different
mean RMSE. Running the same 03_hgbr configuration on an independently
shuffled partition (seed 42) provides a second data point for how sensitive
the aggregate CV estimate is to the choice of fold boundaries.

This is the **baseline arm** of the paired comparison. The paired candidate
arm (09_hgbr_robust_candidate) runs the 05_hgbr_duration configuration on
the **same shuffled patient groups**, so fold-level differences between the
two arms reflect the feature-set change rather than different patient
assignments.

Sourcing strategy: prescribed by the batch (robustness check step).

## Method

Configuration is **identical to 03_hgbr**:
- Features: same eight numeric columns.
- Estimator: `HistGradientBoostingRegressor(random_state=0, max_iter=200, early_stopping=False)`.
- Splitter: `GroupKFold(n_splits=5, shuffle=True, random_state=42)` — same
  patient-level grouping constraint, different assignment of patients to folds.

The script asserts that the shuffled split differs from the original
deterministic split (confirmed: `True`).

No submission CSV is produced. This experiment is a CV diagnostics run only.

## Scope and limitations

- **Same training data and patients** as 03–07. The fold boundaries differ;
  the underlying data do not. This is a second CV partition of the same
  training corpus, not an independent holdout or new data.
- **Higher fold std** on this split (0.190 vs 0.139 on the fixed split).
  This reflects the particular patient mix in the shuffled folds; it does
  not imply that one split is more reliable than the other.
- **No statistical significance claims** are made from the comparison of two
  CV partitions of the same data.
- CV folds do estimate held-out-patient performance (each fold's val set
  contains patients absent from training), but neither split constitutes a
  final untouched holdout.

## Results

| Fold | RMSE |
|---|---|
| 0 | 7.116676 |
| 1 | 7.511801 |
| 2 | 7.416372 |
| 3 | 7.556134 |
| 4 | 7.588937 |
| **Mean** | **7.437984** |
| Std (ddof=1) | 0.190985 |

Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47443

## Status

- **State:** done
- **Approved by user on:** current session
- **Headline result:** RMSE 7.437984 (fold std 0.190985, shuffled GroupKFold seed=42).
  Paired with 09_hgbr_robust_candidate for fold-level differences.
- **Implication for next iteration:** See BENCHMARKS.md § Robustness check.
