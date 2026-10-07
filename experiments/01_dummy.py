# %% [markdown]
# # 01_dummy — Dummy mean baseline
#
# Establishes the floor RMSE using `DummyRegressor(strategy="mean")`.
# The dummy always predicts the training mean of `target`, ignoring all
# features.  Its RMSE on held-out patients equals std(target) ≈ 16.5.
#
# CV strategy: GroupKFold(n_splits=5) with groups=patient_id so no
# patient's visits appear on both sides of a fold boundary.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold

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
HUB_KEY = "01_dummy"
PROJECT_NAME = "bobathon-esilv"

# ── load data ────────────────────────────────────────────────────────────────
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

visits = X_train_raw.merge(y_train_raw, on="Index")

# ── feature matrix and target ────────────────────────────────────────────────
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
# GroupKFold: no patient appears in both train and validation within a fold.
# Groups are pre-split and passed as a generator so skore receives concrete
# (train_idx, test_idx) pairs without needing to re-call split().
gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X, y, groups=groups))

# ── evaluate ─────────────────────────────────────────────────────────────────
dummy = DummyRegressor(strategy="mean")
report = skore.evaluate(dummy, X, y, splitter=iter(cv_splits))

print("=== Cross-validation results (GroupKFold n=5, groups=patient_id) ===")
print(report.metrics.rmse())

# ── push to Hub ──────────────────────────────────────────────────────────────
login(mode="hub")
project = Project(name=PROJECT_NAME, mode="hub", workspace=HUB_WORKSPACE)
project.put(HUB_KEY, report)

# ── submission file ──────────────────────────────────────────────────────────
# Refit on ALL training visits (no CV split) and predict the test set.
dummy_full = DummyRegressor(strategy="mean")
dummy_full.fit(X, y)

sub = sample_sub[["Index"]].copy()
sub["target"] = dummy_full.predict(X_test[FEATURE_COLS])
sub.to_csv(SUBMISSIONS_DIR / f"{HUB_KEY}.csv", index=False)
print(f"Submission written → submissions/{HUB_KEY}.csv  (rows: {len(sub)})")
print(f"Submission target values (all equal): {sub['target'].unique()}")
