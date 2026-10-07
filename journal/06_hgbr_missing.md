# 06_hgbr_missing — HGBR + binary missingness indicators

## Question / hypothesis

Adding explicit binary missing-indicator columns for `off`, `on`, and
`ledd` (the three features with the highest missingness) to the original
eight numeric features will lower CV RMSE if the missingness pattern
carries signal beyond what this HGBR configuration already uses.

## Motivation

HGBR handles NaN by learning a binary split direction (go left or right
when the feature is missing) at each tree node — it uses missingness
implicitly during tree growth.  The ablation tests whether encoding
missingness explicitly as three added binary columns gives a different or
complementary signal in this specific feature set.  EDA showed `off` at
42 % missing and `on` at 30 % missing are the two highest-signal features;
`ledd` at 35–43 % (by cohort) also has substantial missingness.

Sourcing strategy: B2 / B5 carry-forward.

## Method

Keep the identical eight numeric features and identical HGBR configuration
(random_state=0, max_iter=200, early_stopping=False).
Add three binary columns:
  - `off_missing`  = 1 if `off` is NaN, else 0
  - `on_missing`   = 1 if `on` is NaN, else 0
  - `ledd_missing` = 1 if `ledd` is NaN, else 0

These are stateless derivations; no pipeline wrapper needed.  The total
feature matrix is eleven numeric columns; all handled natively by HGBR.

Same GroupKFold(n_splits=5) patient splits as 03_hgbr, passed as
`iter(cv_splits)` to `skore.evaluate`.  Metrics read via
`report.metrics.rmse(aggregate=None)`.

## Risks

- HGBR's tree-growth mechanism already uses missingness implicitly, so the
  explicit indicator columns may provide no additional information in this
  configuration — accepted as an ablation.
- Missingness indicators are fully determined by the original columns;
  there is no leakage risk from the stateless derivation.

## Outcome note

Measured mean RMSE 7.433591, identical to 03_hgbr to 6 decimal places. This
is a feature ablation within the same HGBR estimator. The equal score shows
no measured benefit from explicit binary indicators in this tested
configuration. It does not establish that HGBR's implicit mechanism captures
all missingness signal in general; other configurations (different estimators,
different feature sets) could show a different result.

## Status

- **State:** done
- **Approved by user on:** current session
- **Headline result:** RMSE 7.433591 (fold std 0.139028, GroupKFold-5 fixed). No measured benefit. See BENCHMARKS.md.
- **Implication for next iteration:** excluded from 07_hgbr_combined due to
  zero measured delta. Generalisation to other configurations untested.
