# 10_hgbr_native_categories — HGBR + years_since_diagnosis + native Categorical cohort/gene

## Question / hypothesis

Does using HGBR's native categorical support (`categorical_features="from_dtype"`)
for `cohort` and `gene`, combined with the `years_since_diagnosis` feature from
experiment 05, outperform both 05 (numeric-only) and 07 (OrdinalEncoder encoding)?

## Motivation

Experiment 04 added `cohort` and `gene` via fold-fitted `OrdinalEncoder`
(`unknown_value=-1`, `encoded_missing_value=-2`) and yielded only Δ=−0.002 vs
03_hgbr. Experiment 07 combined this with `years_since_diagnosis` and reached
RMSE 7.381945 — worse than 05 alone (7.375098). A documented limitation of the
OrdinalEncoder approach (noted in Backlog row B6) is that NaN in `gene` (32.3 %
missing) is encoded as the arbitrary integer −2 and HGBR splits on it as a
numeric value. HGBR's native categorical support routes NaN through its own
missing-value mechanism (learns at each split whether missing goes left or right
based on gain), which should represent gene missingness more faithfully.

Sourcing strategy: B6 (Backlog row, limitation noted from 04).

## Method

Feature set: nine numeric columns (identical to 05) plus `cohort_cat` and
`gene_cat` as `pd.Categorical` dtype columns. Eleven features total.

- Numeric nine: `sexM, age_at_diagnosis, age, ledd, time_since_intake_on,
  time_since_intake_off, on, off, years_since_diagnosis`.
- `cohort_cat`: `pd.Categorical(visits["cohort"])` — 2 categories (A, B), no NaN.
- `gene_cat`: `pd.Categorical(visits["gene"])` — 4 categories
  (GBA+, LRRK2+, No Mutation, OTHER+), 14 432/44 590 NaN (32.3 %).

Estimator: `HistGradientBoostingRegressor(categorical_features="from_dtype",
random_state=0, max_iter=200, early_stopping=False)` — identical to 05 in all
other parameters.

CV: identical deterministic `GroupKFold(n_splits=5)` fixed patient splits as
experiments 03–07. **08 and 09 used a different splitter (GroupKFold with
shuffle=True, random_state=42) and are not comparable on per-fold RMSE.**

### Fold-local vocabulary semantics (verified by probe 2026-10-09)

The global `pd.Categorical` construction before CV does **not** violate the
fold-local fitting constraint. This was verified by the following test:

- A `pd.Categorical` with categories `['A','B','C']` was passed to HGBR.fit()
  with training data containing only `['A','B']` values.
- After fit, `_preprocessor.named_transformers_["encoder"].categories_[0]`
  was `['A', 'B']` — `C` was **not** included.
- Predictions for `C` values at inference were numerically identical to
  predictions for `NaN` values, confirming `C` was routed as missing.
- `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=nan)`:
  any value not in training vocabulary is encoded as NaN at predict time,
  then routed through HGBR's native missing mechanism.

**Conclusion**: HGBR's internal `OrdinalEncoder` is fitted from **training
values only**, not from the dtype's `.cat.categories` metadata. The global
`pd.Categorical` is syntactic glue to trigger `from_dtype` detection; it does
not contaminate the fitted vocabulary. Fold-local fitting is satisfied.

The global dtype is nonetheless retained because: (a) it is required for
`from_dtype` to detect the columns, and (b) without a global dtype pandas would
not propagate a `category` dtype to fold slices obtained by `.iloc[]`.

Per-fold gene category coverage check (from CV run):
All 4 gene categories appear in every training fold. No fold has an absent
category that would be treated as missing for all its training data.

### Native categorical semantics vs OrdinalEncoder (07)

| Aspect | 07 (OrdinalEncoder) | 10 (native Categorical) |
|---|---|---|
| NaN in gene | Mapped to integer −2 (numeric) | Routed through HGBR native missing mechanism |
| Unseen categories at predict | Mapped to −1 (numeric) | Encoded as NaN → HGBR missing path |
| Vocabulary source | OrdinalEncoder per fold (training values) | HGBR internal OrdinalEncoder (training values) |
| Vocabulary drift between folds | None (both use training values only) | None (both use training values only) |

Both approaches are fold-local; the difference is in what integer/value
NaN-gene rows receive. In 07, NaN becomes −2 which HGBR splits numerically;
in 10, NaN is preserved as NaN and routed through HGBR's dedicated missing path.

## Risks

- If HGBR was already handling the −2 encoding of NaN-gene effectively in 07
  (treating it like a threshold feature), native may show negligible difference.
- With 4 unique categories for gene and max_bins=255, there is no binning
  truncation risk.

## Outcome

- Mean RMSE 7.376945, std 0.131788 (fixed GroupKFold-5)
- 05_hgbr_duration: 7.375098 (Δ = +0.001847 — 10 is **marginally worse**)
- 07_hgbr_combined (OrdinalEncoder): 7.381945 (Δ = −0.004999 — 10 is better)
- Native categorical for cohort/gene does not improve over 05 (numeric only).
  The native missing-value routing for gene does not outweigh the addition of
  cohort/gene categorical signal in this feature set.
- Feature set for tuning batch (11-14): **05 feature set** (9 numeric only).

No causal attribution is made. The marginal RMSE difference (+0.001847) may
reflect this specific CV partition.

## Status

- **State:** done
- **Approved by user on:** current session (user-authorized batch)
- **Headline result:** RMSE 7.376945 (fold std 0.131788, fixed GroupKFold-5). Marginally worse than 05 (+0.002). Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47912
- **Implication for next iteration:** 05 feature set (9 numeric) is the feature
  base for the tuning batch (11-14). Native categorical offers no additional
  gain vs 05 numeric-only on these fixed splits.
