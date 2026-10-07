# %% [markdown]
# # 10_hgbr_native_categories — HGBR + years_since_diagnosis + native Categorical cohort/gene
#
# Ablation: add cohort and gene via HGBR native categorical support
# (categorical_features="from_dtype") to the 05_hgbr_duration feature set.
#
# VOCABULARY SEMANTICS (verified by probe 2026-10-09):
# HGBR with categorical_features="from_dtype" fits its internal OrdinalEncoder
# from TRAINING VALUES ONLY at fit() time. The pandas Categorical dtype metadata
# (.cat.categories) is NOT used to populate the fitted vocabulary.
# Proof: when a category exists in dtype but is absent from training values,
# it is NOT stored in OrdinalEncoder.categories_, and predictions for that
# value are identical to NaN predictions (it is routed as missing).
# This means the fold-local fitting constraint is satisfied: each CV fold
# fits its OrdinalEncoder vocabulary only from the training split rows.
# The global Categorical dtype is syntactic glue needed for from_dtype
# detection; it does not contaminate the fitted vocabulary.
# For this dataset specifically: all 4 gene categories (GBA+, LRRK2+,
# No Mutation, OTHER+) and both cohort categories (A, B) appear in every
# training fold, so there is no cross-fold vocabulary asymmetry.
#
# NaN in gene_cat (32.3 %) is preserved as NaN — HGBR routes NaN in a
# Categorical column through its native missing-value mechanism, NOT mapped
# to an arbitrary integer as in 07_hgbr_combined's OrdinalEncoder.
#
# CV: GroupKFold(n_splits=5) — same deterministic fixed patient splits as
# experiments 03, 04, 05, 06, 07. Note: 08 and 09 used a DIFFERENT splitter
# (GroupKFold shuffle=True, random_state=42) and are NOT comparable to this
# fixed split.

# %%
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import skore
from skore import Project, login

# -- paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

# -- credentials (workspace only, no secrets printed) -----------------------
cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
HUB_WORKSPACE = cfg["workspace"]  # "teamATN"
HUB_KEY = "10_hgbr_native_categories"
PROJECT_NAME = "bobathon-esilv"

# -- load data ---------------------------------------------------------------
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

visits = X_train_raw.merge(y_train_raw, on="Index")

# -- feature engineering -----------------------------------------------------
# 1. Derived numeric: years_since_diagnosis (stateless arithmetic).
#    NaN when age_at_diagnosis is NaN; handled natively by HGBR.
# 2. Categorical dtype: cohort and gene are cast to pd.Categorical.
#    This is required for HGBR categorical_features="from_dtype" detection.
#    The vocabulary (OrdinalEncoder.categories_) is fit from training values
#    at fit() time only — NOT from this dtype metadata. Global dtype is
#    only needed so sklearn can detect which columns to treat as categorical.
#    Any value present in dtype but absent from training rows is treated as
#    missing at prediction time (OrdinalEncoder unknown_value=nan).

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add years_since_diagnosis and cast cohort/gene to Categorical dtype."""
    out = df.copy()
    out["years_since_diagnosis"] = out["age"] - out["age_at_diagnosis"]
    # pd.Categorical dtype is required for HGBR from_dtype detection.
    # Vocabulary is fitted from training values only (not from these categories).
    out["cohort_cat"] = pd.Categorical(out["cohort"])
    out["gene_cat"] = pd.Categorical(out["gene"])  # NaN kept as NaN
    return out

visits_feat = prepare_features(visits)

# Nine numeric + two categorical
NUMERIC_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
    "years_since_diagnosis",
]
CATEGORICAL_COLS = ["cohort_cat", "gene_cat"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

X = visits_feat[FEATURE_COLS]
y = visits_feat["target"]
groups = visits_feat["patient_id"]

# Verify categorical dtype is set
assert X["cohort_cat"].dtype.name == "category", "cohort_cat must be Categorical dtype"
assert X["gene_cat"].dtype.name == "category", "gene_cat must be Categorical dtype"
print(f"cohort_cat categories (dtype): {list(X['cohort_cat'].cat.categories)}")
print(f"gene_cat categories (dtype): {list(X['gene_cat'].cat.categories)}")
print(f"gene_cat null count: {X['gene_cat'].isnull().sum()} / {len(X)}")
print("Note: HGBR internal OrdinalEncoder fits vocabulary from training values,")
print("      not from these dtype categories. Unseen categories become NaN.")

# -- cross-validation setup --------------------------------------------------
# Deterministic GroupKFold(n_splits=5) — same fixed patient splits as 03-07.
# Different from 08-09 which used GroupKFold(shuffle=True, random_state=42).
gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X, y, groups=groups))

# -- validate: zero patient overlap per fold ---------------------------------
print("\n=== Fold patient-overlap check ===")
for fold_idx, (tr, val) in enumerate(cv_splits):
    train_patients = set(groups.iloc[tr])
    val_patients = set(groups.iloc[val])
    overlap = train_patients & val_patients
    print(f"  Fold {fold_idx}: train {len(train_patients)} patients, "
          f"val {len(val_patients)} patients, overlap {len(overlap)}")
    assert len(overlap) == 0, f"Fold {fold_idx} has patient leakage!"

# Prove fold-local vocabulary: all gene categories appear in every training fold.
# If any gene category were absent from a training fold, that fold's fitted
# OrdinalEncoder would not include it (it would be treated as missing in val).
# This is acceptable for HGBR (NaN handling), but we document the situation.
gene_cats = list(X["gene_cat"].cat.categories)
print("\n=== Per-fold gene category coverage ===")
for fold_idx, (tr, val) in enumerate(cv_splits):
    train_gene_vals = set(X.iloc[tr]["gene_cat"].dropna().unique())
    missing_from_train = set(gene_cats) - train_gene_vals
    print(f"  Fold {fold_idx}: gene cats in training values: {sorted(train_gene_vals)}, "
          f"absent: {sorted(missing_from_train) or 'none'}")

# -- model -------------------------------------------------------------------
# categorical_features="from_dtype": HGBR detects Categorical dtype columns.
# Internal OrdinalEncoder fits from training values only (fold-local).
# NaN in gene_cat is routed through HGBR native missing mechanism.
# Same random_state, max_iter, early_stopping as experiments 03-09.
hgbr = HistGradientBoostingRegressor(
    categorical_features="from_dtype",
    random_state=0,
    max_iter=200,
    early_stopping=False,
)

# -- evaluate ----------------------------------------------------------------
t0 = time.perf_counter()
report = skore.evaluate(hgbr, X, y, splitter=iter(cv_splits))
elapsed = time.perf_counter() - t0

print("\n=== Cross-validation results (fixed GroupKFold n=5, groups=patient_id) ===")

per_fold_df = report.metrics.rmse(aggregate=None)
print(per_fold_df.to_string())
fold_vals = per_fold_df.values.ravel()

agg_df = report.metrics.rmse()
print("\nAggregate:")
print(agg_df.to_string())

print(f"\nPer-fold RMSE : {fold_vals}")
print(f"Mean RMSE     : {np.mean(fold_vals):.6f}")
print(f"Fold std      : {np.std(fold_vals, ddof=1):.6f}")
print(f"Runtime       : {elapsed:.1f}s")
print(f"05_duration ref (fixed splits): 7.375098 (mean), 0.143036 (std)")
print(f"07_combined   (OrdinalEncoder): 7.381945 (mean), 0.132252 (std)")

# Sanity: fold-derived mean must match aggregate
agg_mean = float(agg_df.values[0, 0])
assert abs(agg_mean - np.mean(fold_vals)) < 1e-4, (
    f"Aggregate mean {agg_mean} != fold mean {np.mean(fold_vals)}"
)
print("Cross-check PASSED: aggregate mean matches fold-derived mean")

# -- save report locally before Hub upload ----------------------------------
# CrossValidationReport has no standalone save() method. Persist via a
# dedicated local skore workspace before Hub push to avoid retraining if
# the upload is interrupted.
LOCAL_WORKSPACE = PROJECT_ROOT / "scratch" / "skore_local"
project_local = Project(name=PROJECT_NAME, mode="local",
                        workspace=str(LOCAL_WORKSPACE))
project_local.put(HUB_KEY, report)
print(f"\nReport saved locally: key={HUB_KEY}, workspace={LOCAL_WORKSPACE}")

# -- push to Hub -------------------------------------------------------------
login(mode="hub")
project = Project(name=PROJECT_NAME, mode="hub", workspace=HUB_WORKSPACE)
project.put(HUB_KEY, report)
print(f"Hub report published: key={HUB_KEY}, workspace={HUB_WORKSPACE}")

# -- full-data refit and submission ------------------------------------------
X_test_feat = prepare_features(X_test)

hgbr_full = HistGradientBoostingRegressor(
    categorical_features="from_dtype",
    random_state=0,
    max_iter=200,
    early_stopping=False,
)
hgbr_full.fit(X, y)
preds = hgbr_full.predict(X_test_feat[FEATURE_COLS])

# -- submission validation ---------------------------------------------------
assert not np.any(np.isnan(preds)), "Predictions contain NaN"
assert not np.any(np.isinf(preds)), "Predictions contain Inf"
assert len(preds) == len(sample_sub), (
    f"Row count mismatch: got {len(preds)}, expected {len(sample_sub)}"
)
assert list(X_test["Index"]) == list(sample_sub["Index"]), (
    "X_test Index order does not match sample_submission"
)
sub = sample_sub[["Index"]].copy()
sub["target"] = preds
assert sub["Index"].nunique() == len(sub), "Duplicate Index values in submission"

print(f"\nSubmission validation passed:")
print(f"  rows={len(sub)}, unique IDs={sub['Index'].nunique()}")
print(f"  target range: [{sub['target'].min():.2f}, {sub['target'].max():.2f}]")
print(f"  nonfinite count: {(~np.isfinite(sub['target'].values)).sum()}")

sub.to_csv(SUBMISSIONS_DIR / f"{HUB_KEY}.csv", index=False)
print(f"Submission written -> submissions/{HUB_KEY}.csv  (rows: {len(sub)})")
