# %% [markdown]
# # 15_hgbr_tuning_seed42_candidate — 12-config (lr=0.05, iter=200) on shuffled GroupKFold(seed=42)
#
# Diagnostic: run experiment 12's configuration on the same shuffled seed=42
# GroupKFold splits used in experiments 08 (03-config baseline) and 09
# (05-config candidate). Paired fold differences against 09 check whether
# the tuning gain from lr=0.05 is consistent across both CV partitions.
#
# Both CV partitions are reused selection data; no significance claims.
# No submission CSV — diagnostics only.
#
# Splitter: GroupKFold(n_splits=5, shuffle=True, random_state=42)
# — different from fixed splits used in experiments 03-07, 10-14.

# %%
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import skore
from skore import Project, login

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
HUB_WORKSPACE = cfg["workspace"]  # "teamATN"
HUB_KEY = "15_hgbr_tuning_seed42_candidate"
PROJECT_NAME = "bobathon-esilv"

# -- load data ---------------------------------------------------------------
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")

visits = X_train_raw.merge(y_train_raw, on="Index")

# -- 05 feature set ----------------------------------------------------------
def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with years_since_diagnosis = age - age_at_diagnosis."""
    out = df.copy()
    out["years_since_diagnosis"] = out["age"] - out["age_at_diagnosis"]
    return out

visits_feat = add_duration(visits)

FEATURE_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off",
    "on", "off", "years_since_diagnosis",
]

X = visits_feat[FEATURE_COLS]
y = visits_feat["target"]
groups = visits_feat["patient_id"]

# -- shuffled GroupKFold seed=42 splits (same as 08/09) ----------------------
gkf_seed42 = GroupKFold(n_splits=5, shuffle=True, random_state=42)
cv_splits_seed42 = list(gkf_seed42.split(X, y, groups=groups))

# -- validate: zero patient overlap ------------------------------------------
print("=== Fold patient-overlap check (shuffled GroupKFold seed=42) ===", flush=True)
for fold_idx, (tr, val) in enumerate(cv_splits_seed42):
    train_patients = set(groups.iloc[tr])
    val_patients = set(groups.iloc[val])
    overlap = train_patients & val_patients
    print(f"  Fold {fold_idx}: train {len(train_patients)} patients, "
          f"val {len(val_patients)} patients, overlap {len(overlap)}", flush=True)
    assert len(overlap) == 0, f"Fold {fold_idx} has patient leakage!"

# -- verify val-patient pairing matches 09 (same splitter → same fold sets) --
# 09 used same GroupKFold(n_splits=5, shuffle=True, random_state=42) on same data.
# Re-instantiate from features equivalent to 09 (X is same data, just with duration added).
# The GroupKFold split is determined by groups only; adding a column to X doesn't change it.
X_base = visits[[
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]]
gkf_check = GroupKFold(n_splits=5, shuffle=True, random_state=42)
cv_09_check = list(gkf_check.split(X_base, y, groups=groups))
print("\nPaired fold check vs 09:", flush=True)
for fold_idx in range(5):
    val_09 = set(groups.iloc[cv_09_check[fold_idx][1]])
    val_15 = set(groups.iloc[cv_splits_seed42[fold_idx][1]])
    assert val_09 == val_15, (
        f"Fold {fold_idx}: val patients differ from 09 — paired comparison invalid"
    )
print("  PASSED — all 5 fold val-patient sets match 09's splits.", flush=True)

# -- model (12 config) -------------------------------------------------------
hgbr = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=200,
    random_state=0,
    early_stopping=False,
)

# -- evaluate ----------------------------------------------------------------
t0 = time.perf_counter()
report = skore.evaluate(hgbr, X, y, splitter=iter(cv_splits_seed42))
elapsed = time.perf_counter() - t0

print("\n=== CV results — 12-config, shuffled GroupKFold seed=42 ===", flush=True)
per_fold_df = report.metrics.rmse(aggregate=None)
print(per_fold_df.to_string(), flush=True)
fold_vals = per_fold_df.values.ravel()
agg_df = report.metrics.rmse()
print("\nAggregate:")
print(agg_df.to_string(), flush=True)

print(f"\nPer-fold RMSE : {fold_vals}", flush=True)
print(f"Mean RMSE     : {np.mean(fold_vals):.6f}", flush=True)
print(f"Fold std      : {np.std(fold_vals, ddof=1):.6f}", flush=True)
print(f"Runtime       : {elapsed:.1f}s", flush=True)
print(f"09 (05-config, seed=42): 7.367803 (mean), 0.190191 (std)", flush=True)
print(f"12 fixed-split          : 7.342917 (mean), 0.137556 (std)", flush=True)

# Sanity check
agg_mean = float(agg_df.values[0, 0])
assert abs(agg_mean - np.mean(fold_vals)) < 1e-4, \
    f"Aggregate mean {agg_mean} != fold mean {np.mean(fold_vals)}"
print("Cross-check PASSED: aggregate mean matches fold-derived mean", flush=True)

# Paired fold differences vs 09
fold_vals_09 = np.array([7.053030, 7.460417, 7.329882, 7.468069, 7.527617])
paired_diffs = fold_vals - fold_vals_09
print(f"\nPaired fold differences (15 − 09):", flush=True)
for i, d in enumerate(paired_diffs):
    print(f"  Fold {i}: {fold_vals[i]:.6f} − {fold_vals_09[i]:.6f} = {d:+.6f}", flush=True)
print(f"  Mean paired diff: {paired_diffs.mean():+.6f}", flush=True)
print(f"  Folds where 15 < 09: {(paired_diffs < 0).sum()}/5", flush=True)

# -- save locally and push to Hub --------------------------------------------
LOCAL_WORKSPACE = PROJECT_ROOT / "scratch" / "skore_local"
project_local = Project(name=PROJECT_NAME, mode="local",
                        workspace=str(LOCAL_WORKSPACE))
project_local.put(HUB_KEY, report)
print(f"\nReport saved locally: key={HUB_KEY}", flush=True)

login(mode="hub")
project = Project(name=PROJECT_NAME, mode="hub", workspace=HUB_WORKSPACE)
project.put(HUB_KEY, report)
print(f"Hub report published: key={HUB_KEY}, workspace={HUB_WORKSPACE}", flush=True)
