# %% [markdown]
# # 02_ridge — Ridge regression baseline
#
# Median imputation → StandardScaler → Ridge(alpha=1) on the eight
# numeric features.  All three steps fit inside each fold to prevent
# imputation leakage.
#
# CV strategy: identical GroupKFold(n_splits=5) patient-grouped splits
# as 01_dummy so RMSEs are directly comparable.

# %%
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import skore
from skore import Project, login

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── credentials (non-secret fields only printed) ─────────────────────────────
cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
HUB_WORKSPACE = cfg["workspace"]   # "teamATN"
HUB_KEY = "02_ridge"
PROJECT_NAME = "bobathon-esilv"

# ── load data ────────────────────────────────────────────────────────────────
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

visits = X_train_raw.merge(y_train_raw, on="Index")

# ── feature matrix and target ────────────────────────────────────────────────
# Same eight numeric columns as 01_dummy; IDs never enter the feature matrix.
FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
X = visits[FEATURE_COLS]
y = visits["target"]
groups = visits["patient_id"]

# ── cross-validation setup ───────────────────────────────────────────────────
# Identical GroupKFold splits to 01_dummy — same seed/order, directly comparable.
gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X, y, groups=groups))

# ── validate: zero patient overlap per fold ──────────────────────────────────
print("=== Fold patient-overlap check ===")
for fold_idx, (tr, val) in enumerate(cv_splits):
    train_patients = set(groups.iloc[tr])
    val_patients = set(groups.iloc[val])
    overlap = train_patients & val_patients
    print(f"  Fold {fold_idx}: train {len(train_patients)} patients, "
          f"val {len(val_patients)} patients, overlap {len(overlap)}")
    assert len(overlap) == 0, f"Fold {fold_idx} has patient leakage!"

# ── pipeline ─────────────────────────────────────────────────────────────────
# SimpleImputer + StandardScaler + Ridge all fit inside each fold.
ridge_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    Ridge(alpha=1.0),
)

# ── evaluate ─────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
report = skore.evaluate(ridge_pipeline, X, y, splitter=iter(cv_splits))
elapsed = time.perf_counter() - t0

print("\n=== Cross-validation results (GroupKFold n=5, groups=patient_id) ===")

# Per-fold RMSE: aggregate=None returns one column per split, NOT mean/std.
# Averaging the aggregate (mean, std) row would be wrong.
per_fold_df = report.metrics.rmse(aggregate=None)
print(per_fold_df.to_string())
fold_vals = per_fold_df.values.ravel()

# Aggregate cross-check
agg_df = report.metrics.rmse()
print("\nAggregate:")
print(agg_df.to_string())

print(f"\nPer-fold RMSE : {fold_vals}")
print(f"Mean RMSE     : {np.mean(fold_vals):.5f}")
print(f"Fold std      : {np.std(fold_vals, ddof=1):.5f}")
print(f"Runtime       : {elapsed:.1f}s")
print(f"Dummy ref     : 16.50")

# Sanity: fold-derived mean must match aggregate
agg_mean = float(agg_df.values[0, 0])
assert abs(agg_mean - np.mean(fold_vals)) < 1e-4, (
    f"Aggregate mean {agg_mean} != fold mean {np.mean(fold_vals)}"
)
print("Cross-check PASSED: aggregate mean matches fold-derived mean")

# ── push to Hub ──────────────────────────────────────────────────────────────
login(mode="hub")
project = Project(name=PROJECT_NAME, mode="hub", workspace=HUB_WORKSPACE)
project.put(HUB_KEY, report)
print(f"\nHub report published: key={HUB_KEY}, workspace={HUB_WORKSPACE}")

# ── full-data refit and submission ───────────────────────────────────────────
# Refit on ALL training visits then predict X_test aligned to sample_submission.
ridge_full = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    Ridge(alpha=1.0),
)
ridge_full.fit(X, y)

# Align test columns to FEATURE_COLS (same order, same set).
X_test_feat = X_test[FEATURE_COLS]
preds = ridge_full.predict(X_test_feat)

# ── submission validation ─────────────────────────────────────────────────────
assert not np.any(np.isnan(preds)), "Predictions contain NaN"
assert not np.any(np.isinf(preds)), "Predictions contain Inf"
assert len(preds) == len(sample_sub), (
    f"Row count mismatch: got {len(preds)}, expected {len(sample_sub)}"
)
# Verify X_test Index order matches sample_submission (not sample_sub vs itself)
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
