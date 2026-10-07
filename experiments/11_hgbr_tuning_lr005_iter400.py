# %% [markdown]
# # 11_hgbr_tuning_lr005_iter400 — HGBR tuning: lr=0.05, max_iter=400
#
# Feature set: 05 feature set (9 numeric: 8 original + years_since_diagnosis).
# This is the winner of the 10 vs 05 ablation (05 RMSE 7.375098 < 10 RMSE 7.376945).
#
# Tuning: reduce learning_rate to 0.05 and double max_iter to 400.
# Same total "boosting budget" (lr × iter = 0.05 × 400 = 20 = 0.1 × 200),
# but finer steps per tree. Standard first HGBR tuning move.
#
# CV: fixed GroupKFold(n_splits=5) — same deterministic patient splits as 03-07, 10.
# (08-09 used a different splitter: GroupKFold shuffle=True, random_state=42.)

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
HUB_KEY = "11_hgbr_tuning_lr005_iter400"
PROJECT_NAME = "bobathon-esilv"

# -- load data ---------------------------------------------------------------
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

visits = X_train_raw.merge(y_train_raw, on="Index")

# -- feature engineering: 05 feature set ------------------------------------
# Same as 05_hgbr_duration: 8 numeric + years_since_diagnosis (stateless arithmetic).
# NaN when age_at_diagnosis is NaN; HGBR handles natively.

def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with years_since_diagnosis = age - age_at_diagnosis."""
    out = df.copy()
    out["years_since_diagnosis"] = out["age"] - out["age_at_diagnosis"]
    return out

visits_feat = add_duration(visits)

FEATURE_COLS = [
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

X = visits_feat[FEATURE_COLS]
y = visits_feat["target"]
groups = visits_feat["patient_id"]

# -- cross-validation setup --------------------------------------------------
# Deterministic GroupKFold(n_splits=5) — same fixed patient splits as 03-07, 10.
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
# Tuning: learning_rate=0.05, max_iter=400. All other params at defaults.
# Same total budget as 05 (lr=0.1, iter=200) but finer gradient steps.
hgbr = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=400,
    random_state=0,
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

# Sanity: fold-derived mean must match aggregate
agg_mean = float(agg_df.values[0, 0])
assert abs(agg_mean - np.mean(fold_vals)) < 1e-4, (
    f"Aggregate mean {agg_mean} != fold mean {np.mean(fold_vals)}"
)
print("Cross-check PASSED: aggregate mean matches fold-derived mean")

# -- save locally and push to Hub --------------------------------------------
LOCAL_WORKSPACE = PROJECT_ROOT / "scratch" / "skore_local"
project_local = Project(name=PROJECT_NAME, mode="local",
                        workspace=str(LOCAL_WORKSPACE))
project_local.put(HUB_KEY, report)
print(f"\nReport saved locally: key={HUB_KEY}")

login(mode="hub")
project = Project(name=PROJECT_NAME, mode="hub", workspace=HUB_WORKSPACE)
project.put(HUB_KEY, report)
print(f"Hub report published: key={HUB_KEY}, workspace={HUB_WORKSPACE}")

# -- full-data refit and submission ------------------------------------------
X_test_feat = add_duration(X_test)[FEATURE_COLS]
hgbr_full = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=400,
    random_state=0,
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
