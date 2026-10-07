# %% [markdown]
# # 09_hgbr_robust_candidate — 05_hgbr_duration on shuffled GroupKFold(n_splits=5, shuffle=True, random_state=42)
#
# Robustness check: run the best candidate configuration (05_hgbr_duration:
# 8 numeric + years_since_diagnosis) on the same shuffled GroupKFold
# seed=42 splits used in 08_hgbr_robust_baseline. Paired fold differences
# between 08 and 09 reveal whether the improvement from years_since_diagnosis
# holds under a different patient grouping.
#
# IMPORTANT: Same robustness-check caveats as 08. This is NOT an
# untouched holdout; it is a second independent CV split on the same
# training data. Paired differences describe split-to-split consistency,
# not generalisation to unseen data. No statistical significance claims
# are made from this paired comparison alone.
#
# Hub key: 09_hgbr_robust_candidate (distinct from 05_hgbr_duration).
# No submission CSV — diagnostics only.

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

# -- credentials (workspace only, no secrets printed) -----------------------
cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
HUB_WORKSPACE = cfg["workspace"]  # "teamATN"
HUB_KEY = "09_hgbr_robust_candidate"
PROJECT_NAME = "bobathon-esilv"

# -- load data ---------------------------------------------------------------
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")

visits = X_train_raw.merge(y_train_raw, on="Index")

# -- feature engineering: add years_since_diagnosis -------------------------
# Stateless arithmetic; NaN when age_at_diagnosis is NaN (HGBR handles natively).


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with years_since_diagnosis column appended."""
    out = df.copy()
    out["years_since_diagnosis"] = out["age"] - out["age_at_diagnosis"]
    return out


visits_feat = add_duration(visits)

# -- feature matrix (05_hgbr_duration: 8 numeric + years_since_diagnosis) ---
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
FEATURE_COLS = BASE_FEATURE_COLS + ["years_since_diagnosis"]

X = visits_feat[FEATURE_COLS]
y = visits_feat["target"]
groups = visits_feat["patient_id"]

# -- same shuffled GroupKFold splits as 08_hgbr_robust_baseline -------------
# MUST use identical splitter(n_splits=5, shuffle=True, random_state=42)
# so fold patient sets are comparable for paired fold differences.
gkf_robust = GroupKFold(n_splits=5, shuffle=True, random_state=42)
cv_splits_robust = list(gkf_robust.split(X, y, groups=groups))

# -- validate: zero patient overlap per fold ---------------------------------
print("=== Fold patient-overlap check (shuffled GroupKFold seed=42) ===")
for fold_idx, (tr, val) in enumerate(cv_splits_robust):
    train_patients = set(groups.iloc[tr])
    val_patients = set(groups.iloc[val])
    overlap = train_patients & val_patients
    print(f"  Fold {fold_idx}: train {len(train_patients)} patients, "
          f"val {len(val_patients)} patients, overlap {len(overlap)}")
    assert len(overlap) == 0, f"Fold {fold_idx} has patient leakage!"

# -- verify fold patient sets match 08 (same split, different feature set) --
# Cross-check: the val patient sets must be identical to those in 08 so
# paired differences are on the same patients.
X_base = visits[[
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]]
gkf_check = GroupKFold(n_splits=5, shuffle=True, random_state=42)
cv_base = list(gkf_check.split(X_base, y, groups=groups))
for fold_idx in range(5):
    base_val = set(groups.iloc[cv_base[fold_idx][1]])
    cand_val = set(groups.iloc[cv_splits_robust[fold_idx][1]])
    assert base_val == cand_val, (
        f"Fold {fold_idx}: candidate val patients differ from baseline val patients — "
        "paired comparison is invalid"
    )
print("Patient-set pairing check PASSED: all 5 fold val sets match 08_robust_baseline")

# -- model (05_hgbr_duration config) ----------------------------------------
hgbr = HistGradientBoostingRegressor(
    random_state=0,
    max_iter=200,
    early_stopping=False,
)

# -- evaluate ----------------------------------------------------------------
t0 = time.perf_counter()
report = skore.evaluate(hgbr, X, y, splitter=iter(cv_splits_robust))
elapsed = time.perf_counter() - t0

print("\n=== Robustness CV results — 05_duration config, shuffled splits ===")

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
print(f"05_duration fixed-split : 7.37510 (mean), 0.14304 (std)")

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
print(f"NOTE: This is the robustness candidate arm (shuffled seed=42 splits).")
print(f"      Compare with 08_hgbr_robust_baseline for paired fold differences.")
