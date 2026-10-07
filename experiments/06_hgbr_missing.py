# %% [markdown]
# # 06_hgbr_missing — HGBR + binary missingness indicators
#
# Ablation: add three binary indicator columns (off_missing, on_missing,
# ledd_missing) to the original eight numeric features. HGBR already
# handles NaN natively; this test checks whether explicit binary
# encoding of missingness for the three highest-missing features
# provides additional signal.
#
# CV strategy: identical GroupKFold(n_splits=5) patient splits as
# 01_dummy..03_hgbr — directly comparable.

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
HUB_KEY = "06_hgbr_missing"
PROJECT_NAME = "bobathon-esilv"

# -- load data ---------------------------------------------------------------
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

visits = X_train_raw.merge(y_train_raw, on="Index")

# -- feature matrix and target -----------------------------------------------
# Original eight numeric features plus three binary missingness indicators.
# Derivation is stateless; no fold-fitting needed.
BASE_FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
MISSING_INDICATOR_COLS = ["off_missing", "on_missing", "ledd_missing"]


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with binary missingness indicator columns appended."""
    out = df.copy()
    out["off_missing"] = df["off"].isna().astype(int)
    out["on_missing"] = df["on"].isna().astype(int)
    out["ledd_missing"] = df["ledd"].isna().astype(int)
    return out


visits_feat = add_missing_indicators(visits)
FEATURE_COLS = BASE_FEATURE_COLS + MISSING_INDICATOR_COLS

X = visits_feat[FEATURE_COLS]
y = visits_feat["target"]
groups = visits_feat["patient_id"]

# -- cross-validation setup --------------------------------------------------
gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X, y, groups=groups))

# -- validate: zero patient overlap per fold ---------------------------------
print("=== Fold patient-overlap check ===")
for fold_idx, (tr, val) in enumerate(cv_splits):
    train_patients = set(groups.iloc[tr])
    val_patients = set(groups.iloc[val])
    overlap = train_patients & val_patients
    print(f"  Fold {fold_idx}: train {len(train_patients)} patients, "
          f"val {len(val_patients)} patients, overlap {len(overlap)}")
    assert len(overlap) == 0, f"Fold {fold_idx} has patient leakage!"

# -- model -------------------------------------------------------------------
hgbr = HistGradientBoostingRegressor(
    random_state=0,
    max_iter=200,
    early_stopping=False,
)

# -- evaluate ----------------------------------------------------------------
t0 = time.perf_counter()
report = skore.evaluate(hgbr, X, y, splitter=iter(cv_splits))
elapsed = time.perf_counter() - t0

print("\n=== Cross-validation results (GroupKFold n=5, groups=patient_id) ===")

per_fold_df = report.metrics.rmse(aggregate=None)
print(per_fold_df.to_string())
fold_vals = per_fold_df.values.ravel()

agg_df = report.metrics.rmse()
print("\nAggregate:")
print(agg_df.to_string())

print(f"\nPer-fold RMSE : {fold_vals}")
print(f"Mean RMSE     : {np.mean(fold_vals):.5f}")
print(f"Fold std      : {np.std(fold_vals, ddof=1):.5f}")
print(f"Runtime       : {elapsed:.1f}s")
print(f"03_hgbr ref   : 7.43359 (mean), 0.13903 (std)")

# Sanity: fold-derived mean must match aggregate
agg_mean = float(agg_df.values[0, 0])
assert abs(agg_mean - np.mean(fold_vals)) < 1e-4, (
    f"Aggregate mean {agg_mean} != fold mean {np.mean(fold_vals)}"
)
print("Cross-check PASSED: aggregate mean matches fold-derived mean")

# -- push to Hub -------------------------------------------------------------
login(mode="hub")
project = Project(name=PROJECT_NAME, mode="hub", workspace=HUB_WORKSPACE)
project.put(HUB_KEY, report)
print(f"\nHub report published: key={HUB_KEY}, workspace={HUB_WORKSPACE}")

# -- full-data refit and submission ------------------------------------------
X_test_feat = add_missing_indicators(X_test)[FEATURE_COLS]
hgbr_full = HistGradientBoostingRegressor(
    random_state=0,
    max_iter=200,
    early_stopping=False,
)
hgbr_full.fit(X, y)
preds = hgbr_full.predict(X_test_feat)

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
