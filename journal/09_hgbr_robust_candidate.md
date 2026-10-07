# 09_hgbr_robust_candidate — 05_hgbr_duration on shuffled GroupKFold(n_splits=5, shuffle=True, random_state=42)

## Question / hypothesis

Does the improvement from `years_since_diagnosis` (observed in 05_hgbr_duration
vs 03_hgbr on the fixed splits) also appear when both configurations are
evaluated on a different patient grouping (shuffled seed 42)?

## Motivation

Experiment 05_hgbr_duration showed a mean RMSE reduction of −0.059 vs
03_hgbr on the fixed deterministic splits. Both results come from a single
CV partition, so the difference could in principle reflect the specific
patient mix of those folds rather than a genuine signal from
`years_since_diagnosis`. Running both configurations on an independently
shuffled partition and examining **paired fold differences** (same patients
in each corresponding fold) provides additional evidence about whether the
improvement is partition-specific.

This is the **candidate arm** of the paired comparison. The baseline arm
(08_hgbr_robust_baseline) runs 03_hgbr on the same shuffled patient groups.

Sourcing strategy: prescribed by the batch (robustness check step).

## Method

Configuration is **identical to 05_hgbr_duration**:
- Features: eight original numeric columns plus `years_since_diagnosis = age − age_at_diagnosis`.
- Estimator: `HistGradientBoostingRegressor(random_state=0, max_iter=200, early_stopping=False)`.
- Splitter: `GroupKFold(n_splits=5, shuffle=True, random_state=42)` — **same
  parameters as 08_hgbr_robust_baseline** so the five fold val-patient sets
  are identical between the two arms.

The script asserts that the val-patient set for each fold matches exactly
between 08 and 09 (confirmed: all 5 folds PASSED).

No submission CSV is produced. This experiment is a CV diagnostics run only.

## Scope and limitations

- **Same training data and patients** as 03–08. Only the fold boundaries
  (patient group assignment) differ from experiments 03–07; the candidate
  arm shares these boundaries with 08.
- **Paired fold differences** (09 fold RMSE minus 08 fold RMSE) are negative
  on all five folds, consistent with the fixed-split result. This adds
  evidence for the improvement, but both CV partitions use the same training
  data. A truly held-out final test would be required to confirm generalisation.
- **No statistical significance claims** are made. Two CV partitions of the
  same data are not independent experiments.
- The higher fold std on the shuffled split (0.190 vs 0.143 for 05's fixed
  split) reflects the particular patient mix; it is a property of this
  partition, not of the feature set.

## Results

| Fold | Candidate (09) | Baseline (08) | Paired diff (09−08) |
|---|---|---|---|
| 0 | 7.053030 | 7.116676 | −0.063646 |
| 1 | 7.460417 | 7.511801 | −0.051384 |
| 2 | 7.329882 | 7.416372 | −0.086490 |
| 3 | 7.468069 | 7.556134 | −0.088065 |
| 4 | 7.527617 | 7.588937 | −0.061320 |
| **Mean** | **7.367803** | **7.437984** | **−0.070181** |
| Std (ddof=1) | 0.190191 | 0.190985 | — |

Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47461

## Status

- **State:** done
- **Approved by user on:** current session
- **Headline result:** RMSE 7.367803 (fold std 0.190191, shuffled GroupKFold seed=42).
  Paired mean fold diff vs 08 = −0.070 (candidate lower on all 5 folds).
- **Implication for next iteration:** The direction of improvement from
  years_since_diagnosis is consistent across both CV partitions. See
  BENCHMARKS.md § Robustness check for full interpretation and limitations.
