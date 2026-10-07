# %% [markdown]
# # EDA: Parkinson's Disease True-OFF Motor Score
#
# Exploratory data analysis of the bobathon-esilv competition tables,
# run before designing a model.
#
# - **Raw data** is read-only — four CSVs under `data/` are loaded but
#   never modified.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): an
#   `eda_<table>.html` report per table, summarised in `eda.md`.
#
# Task: regression — predict `target` (debiased true OFF MDS-UPDRS motor
# score) for every patient visit. `patient_id` is the grouping variable.
# Metric: visit-level RMSE. Holdout is by patient (no train/test patient
# overlap).

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# data/eda.py lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here; raw data also lives here.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data

# %%
X_train = pd.read_csv(EDA_DIR / "X_train.csv")
y_train = pd.read_csv(EDA_DIR / "y_train.csv")
X_test = pd.read_csv(EDA_DIR / "X_test.csv")
sample_sub = pd.read_csv(EDA_DIR / "sample_submission.csv")

# Joined training table (target-bearing)
visits = X_train.merge(y_train, on="Index")

{  # noqa: B018
    "X_train": X_train.shape,
    "y_train": y_train.shape,
    "visits (joined)": visits.shape,
    "X_test": X_test.shape,
    "sample_submission": sample_sub.shape,
}

# %% [markdown]
# ## Patient counts and train/test overlap

# %%
train_patients = set(X_train["patient_id"].unique())
test_patients = set(X_test["patient_id"].unique())
visits_per_patient_train = X_train.groupby("patient_id").size()
{  # noqa: B018
    "train_patients": len(train_patients),
    "test_patients": len(test_patients),
    "overlap": len(train_patients & test_patients),
    "train_visits_per_patient_mean": round(visits_per_patient_train.mean(), 2),
    "train_visits_per_patient_max": int(visits_per_patient_train.max()),
    "train_visits_per_patient_median": int(visits_per_patient_train.median()),
}

# %% [markdown]
# ## Table overview — X_train
#
# Per-column dtype, missingness, cardinality.

# %%
report_xtrain = skrub.TableReport(X_train, title="X_train", verbose=0)
report_xtrain.write_html(EDA_DIR / "eda_x_train.html")

summary_xtrain = json.loads(report_xtrain.json())
n_rows_train = summary_xtrain.get("n_rows")
overview_xtrain = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_xtrain.get("columns", [])
]
{"n_rows": n_rows_train, "n_columns": len(overview_xtrain), "columns": overview_xtrain}  # noqa: B018

# %% [markdown]
# ## Table overview — visits (X_train joined with y_train)
#
# Target-bearing table: used for structure signals, target analysis,
# and associations.

# %%
report_visits = skrub.TableReport(visits, title="visits (train+target)", verbose=0)
report_visits.write_html(EDA_DIR / "eda_visits.html")

summary_visits = json.loads(report_visits.json())
n_rows_visits = summary_visits.get("n_rows")
overview_visits = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_visits.get("columns", [])
]
{"n_rows": n_rows_visits, "n_columns": len(overview_visits), "columns": overview_visits}  # noqa: B018

# %% [markdown]
# ## Table overview — X_test

# %%
report_xtest = skrub.TableReport(X_test, title="X_test", verbose=0)
report_xtest.write_html(EDA_DIR / "eda_x_test.html")

summary_xtest = json.loads(report_xtest.json())
overview_xtest = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_xtest.get("columns", [])
]
{"n_rows": summary_xtest.get("n_rows"), "n_columns": len(overview_xtest), "columns": overview_xtest}  # noqa: B018

# %% [markdown]
# ## Target analysis
#
# Distribution of `target` (true OFF MDS-UPDRS motor score) — spread,
# skew. Shapes the metric choice and whether CV should stratify.

# %%
TARGET = "target"
target_col = next(
    (col for col in summary_visits.get("columns", []) if col.get("name") == TARGET),
    None,
)
target_col  # noqa: B018

# %% [markdown]
# ## Train/test column differences
#
# Check for columns present in one split but not the other.

# %%
train_cols = set(X_train.columns)
test_cols = set(X_test.columns)
{  # noqa: B018
    "in_train_not_test": sorted(train_cols - test_cols),
    "in_test_not_train": sorted(test_cols - train_cols),
    "shared_cols": sorted(train_cols & test_cols),
}

# %% [markdown]
# ## Structure signals
#
# Datetime columns and high-cardinality id/group columns from the visits
# table. High unique-ratio columns that are not identifiers may need
# grouped CV.

# %%
datetime_cols = [
    col.get("name")
    for col in summary_visits.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows_visits
            if n_rows_visits
            else None,
        }
        for col in summary_visits.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:12]}  # noqa: B018

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations in the visits table.
# Strong feature↔target links are candidate predictors; implausibly
# perfect ones are leakage flags.

# %%
assoc = skrub.column_associations(visits)
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest_overall": rows[:10]}  # noqa: B018

# %% [markdown]
# ## Summary
#
# The findings and their modelling implications are written up in
# `data/eda.md`.
