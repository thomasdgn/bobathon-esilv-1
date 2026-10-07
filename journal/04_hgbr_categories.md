# 04_hgbr_categories — HGBR + categorical features (cohort, gene)

## Question / hypothesis

Adding `cohort` (A/B, always present) and `gene` (4 categories, 32 %
missing) to the eight numeric features will lower CV RMSE if either
variable carries signal beyond what the numeric features already
capture.

## Motivation

EDA showed cohort A vs B has different LEDD missingness patterns (35.9 %
vs 42.6 %) and structural differences in the data collection protocol.
`gene` has four distinct values (No Mutation, LRRK2+, GBA+, OTHER+) with
32 % NaN; genetic background may have a disease-progression effect on the
true-OFF score independent of the numeric scores. Both are categorical
(low cardinality), so they need encoding — HGBR supports only numeric
inputs, so a fold-fitted encoder is required to avoid leakage.

Sourcing strategy: B4 (Backlog row).

## Method

Keep the identical eight numeric features and identical HGBR
configuration (random_state=0, max_iter=200, early_stopping=False).
Add `cohort` and `gene` via a `sklearn.preprocessing.OrdinalEncoder`
fitted per fold (inside the Pipeline) so category vocabularies are
learned only from the training fold.  `OrdinalEncoder` maps each
category to an integer by lexicographic sort order — this ordering is
arbitrary and carries no meaning, but HGBR can find useful splits from
arbitrary integer codes.  NaN in `gene` is mapped to `encoded_missing_value=-2`
(an integer sentinel), making the column fully numeric before HGBR sees
it; HGBR therefore treats missing `gene` as integer code -2 via ordinary
numeric splits, NOT via its native NaN-handling mechanism.
`unknown_value=-1` handles any category unseen during fold training,
also as a plain integer.  Limitation: sentinel values -1 and -2 are
arbitrary choices whose position on the integer axis affects splits.

The pipeline is a `sklearn.pipeline.Pipeline` with two steps:
1. `ColumnTransformer`: passthrough eight numeric, OrdinalEncoder two
   categorical (handle_unknown="use_encoded_value", unknown_value=-1,
   encoded_missing_value=-2).
2. `HistGradientBoostingRegressor` (same config as 03_hgbr).

Same GroupKFold(n_splits=5) patient splits as 03_hgbr, passed as
`iter(cv_splits)` to `skore.evaluate`.  Metrics read via
`report.metrics.rmse(aggregate=None)`.

## Risks

- If `cohort` and `gene` carry no signal beyond the numeric features,
  the additional dimensionality may slightly inflate RMSE or leave it
  flat — accepted, as this is an ablation.
- OrdinalEncoder vocabularies built per fold: with only two cohort
  values, the vocabulary is always identical; the fold-fitting is more
  relevant for `gene`.
- The sentinel approach (`unknown_value=-1`, `encoded_missing_value=-2`)
  maps unknown categories and NaN to fixed negative integers.  HGBR
  treats these codes as ordinary numeric split candidates — this is the
  intended behaviour, but the arbitrary choice of -1/-2 (rather than,
  say, -0.5 between valid codes) may affect which splits are chosen.

## Outcome note

Measured mean RMSE 7.431442, vs 03_hgbr 7.433591 (Δ = −0.002). This is a
feature ablation within the same HGBR estimator; it does not separate the
contribution of categories from any Ridge-vs-HGBR or imputation comparison.
The small delta is consistent with either marginal signal or CV noise on this
partition; a different grouping could produce a different ordering.

## Status

- **State:** done
- **Approved by user on:** current session
- **Headline result:** RMSE 7.431442 (fold std 0.120716, GroupKFold-5 fixed). See BENCHMARKS.md.
- **Implication for next iteration:** marginal delta; combined result (07) is
  available. See BENCHMARKS.md for full comparison.
