# 05_hgbr_duration — HGBR + years_since_diagnosis feature

## Question / hypothesis

Adding the derived feature `years_since_diagnosis = age − age_at_diagnosis`
to the original eight numeric features will lower CV RMSE if disease
duration carries signal not already present in the two raw age columns.

## Motivation

EDA flagged `age` and `age_at_diagnosis` as near-collinear: their
difference (disease duration in years) may capture progression in a more
direct, interpretable form.  Backlog row B5 / B2 explicitly listed this
derivation.  HGBR already uses `age` and `age_at_diagnosis` individually,
so this ablation tests whether their interaction expressed as a difference
adds unique signal.  `age_at_diagnosis` is missing for 2 319 / 44 590
rows (5.2 %); HGBR handles NaN natively so no imputer is needed even
when the derived feature is NaN.

Sourcing strategy: B5 carry-forward.

## Method

Keep the identical eight numeric features and identical HGBR configuration
(random_state=0, max_iter=200, early_stopping=False).
Add a single derived column `years_since_diagnosis = age − age_at_diagnosis`
(a stateless, pure-Python column addition; NaN when `age_at_diagnosis` is
NaN).  No other features are added.  The total feature matrix is nine
numeric columns; all handled natively by HGBR.

Derivation is done once before the CV loop (same value every fold) because
it is a stateless arithmetic combination of two existing columns with no
statistics fitted from data.  The result is appended to `X`; no pipeline
wrapper needed.

Same GroupKFold(n_splits=5) patient splits as 03_hgbr, passed as
`iter(cv_splits)` to `skore.evaluate`.  Metrics read via
`report.metrics.rmse(aggregate=None)`.

## Risks

- If `age` and `age_at_diagnosis` already allow HGBR to reconstruct the
  difference implicitly (a single split on each), the added column may
  give negligible improvement — accepted.
- When `age_at_diagnosis` is NaN, `years_since_diagnosis` is also NaN;
  this adds a second NaN indicator for that row beyond the original
  `age_at_diagnosis` column, which may or may not be beneficial.

## Outcome note

Measured mean RMSE 7.375098, vs 03_hgbr 7.433591 (Δ = −0.059, −0.79 %).
This is a feature ablation within the same HGBR estimator; both experiments
use identical model configuration. The improvement is observed on this fixed
CV partition; a different patient grouping could in principle yield a different
result. See the robustness check (experiments 08/09) for a second partition.

## Status

- **State:** done
- **Approved by user on:** current session
- **Headline result:** RMSE 7.375098 (fold std 0.143036, GroupKFold-5 fixed). Best on fixed splits. See BENCHMARKS.md.
- **Implication for next iteration:** best single-addition result. Combined
  with categories in 07_hgbr_combined; robustness checked in 09.
