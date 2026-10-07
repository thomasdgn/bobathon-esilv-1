# %% [markdown]
# # 08_hgbr_robust_baseline — 03_hgbr on shuffled GroupKFold(n_splits=5, shuffle=True, random_state=42)
#
# Robustness check: run the original 03_hgbr configuration on a different
# set of patient group assignments (shuffled GroupKFold seed 42) to verify
# that the fixed-split CV result is not an artefact of the specific fold
# boundaries. This is the BASELINE arm of the paired robustness comparison.
#
# IMPORTANT: This is a robustness check using a second independent
# CV split — it is NOT an untouched holdout. The same training data is
# used; only the fold boundaries differ. Paired fold differences between
# arms 08 and 09 describe how results change across two different patient
# groupings, not train/test performance on held-out data.
#
# Hub key: 08_hgbr_robust_baseline (distinct from 03_hgbr).
# No submission CSV produced — this is a CV diagnostics experiment only.

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
HUB_KEY = "08_hgbr_robust_baseline"
PROJECT_NAME = "bobathon-esilv"

# -- load data ---------------------------------------------------------------
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")

visits = X_train_raw.merge(y_train_raw, on="Index")

# -- feature matrix and target (identical to 03_hgbr) -----------------------
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

# -- shuffled GroupKFold splits (robustness split) ---------------------------
# shuffle=True, random_state=42 produces different patient group assignments
# than the original deterministic GroupKFold(n_splits=5, shuffle=False).
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

# -- validate: shuffled splits differ from original --------------------------
gkf_orig = GroupKFold(n_splits=5)
cv_splits_orig = list(gkf_orig.split(X, y, groups=groups))
orig_val_sets = [frozenset(groups.iloc[val]) for _, val in cv_splits_orig]
robust_val_sets = [frozenset(groups.iloc[val]) for _, val in cv_splits_robust]
any_different = any(a != b for a, b in zip(orig_val_sets, robust_val_sets))
print(f"\nSplit differs from original deterministic split: {any_different}")
assert any_different, "Shuffled split is identical to original — check seed"

# -- model (identical to 03_hgbr) --------------------------------------------
hgbr = HistGradientBoostingRegressor(
    random_state=0,
    max_iter=200,
    early_stopping=False,
)

# -- evaluate ----------------------------------------------------------------
t0 = time.perf_counter()
report = skore.evaluate(hgbr, X, y, splitter=iter(cv_splits_robust))
elapsed = time.perf_counter() - t0

print("\n=== Robustness CV results — 03_hgbr config, shuffled splits ===")

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
print(f"03_hgbr fixed : 7.43359 (mean), 0.13903 (std)")

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
print(f"NOTE: This is the robustness baseline arm (shuffled seed=42 splits).")
print(f"      Compare with 09_hgbr_robust_candidate for paired differences.")
