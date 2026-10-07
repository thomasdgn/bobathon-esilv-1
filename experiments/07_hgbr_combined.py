# %% [markdown]
# # 07_hgbr_combined — HGBR + years_since_diagnosis + cohort + gene
#
# Combines the two ablations that improved over 03_hgbr:
#   - 05: years_since_diagnosis = age - age_at_diagnosis (clear gain -0.058)
#   - 04: cohort + gene via OrdinalEncoder (marginal gain -0.002)
# Excludes 06 (missing indicators) — zero RMSE improvement vs baseline.
#
# Adaptive selection: 06 excluded because its fold RMSE was identical to
# 03_hgbr (7.433591) to 6 decimal places.
#
# CV strategy: identical GroupKFold(n_splits=5) patient splits as 03..06.

# %%
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

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
HUB_KEY = "07_hgbr_combined"
PROJECT_NAME = "bobathon-esilv"

# -- load data ---------------------------------------------------------------
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

visits = X_train_raw.merge(y_train_raw, on="Index")

# -- feature engineering: add years_since_diagnosis -------------------------
# Stateless arithmetic; NaN when age_at_diagnosis is NaN (HGBR handles natively).


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with years_since_diagnosis column appended."""
    out = df.copy()
    out["years_since_diagnosis"] = out["age"] - out["age_at_diagnosis"]
    return out


visits_feat = add_duration(visits)

# -- feature matrix and target -----------------------------------------------
# Nine numeric (8 original + years_since_diagnosis) + 2 categorical.
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
CATEGORICAL_COLS = ["cohort", "gene"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

X = visits_feat[FEATURE_COLS]
y = visits_feat["target"]
groups = visits_feat["patient_id"]

# -- cross-validation setup --------------------------------------------------
# Identical GroupKFold splits to 03..06 — directly comparable.
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

# -- pipeline ----------------------------------------------------------------
# OrdinalEncoder fitted per fold: vocabulary learned only from training fold.
# Categories → lexicographic integer codes (arbitrary order, not ranked).
# unknown_value=-1: unseen cross-fold categories mapped to integer -1.
# encoded_missing_value=-2: NaN in 'gene' mapped to integer -2; HGBR splits
#   on -2 as an ordinary numeric value (not via native NaN mechanism).
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_COLS),
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            ),
            CATEGORICAL_COLS,
        ),
    ],
    remainder="drop",
)

hgbr = HistGradientBoostingRegressor(
    random_state=0,
    max_iter=200,
    early_stopping=False,
)

pipe = Pipeline([("prep", preprocessor), ("hgbr", hgbr)])

# -- evaluate ----------------------------------------------------------------
t0 = time.perf_counter()
report = skore.evaluate(pipe, X, y, splitter=iter(cv_splits))
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
print(f"05_duration   : 7.37510 (mean), 0.14304 (std)")

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
X_test_feat = add_duration(X_test)

preprocessor_full = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_COLS),
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            ),
            CATEGORICAL_COLS,
        ),
    ],
    remainder="drop",
)
hgbr_full = HistGradientBoostingRegressor(
    random_state=0,
    max_iter=200,
    early_stopping=False,
)
pipe_full = Pipeline([("prep", preprocessor_full), ("hgbr", hgbr_full)])
pipe_full.fit(X, y)

preds = pipe_full.predict(X_test_feat[FEATURE_COLS])

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
