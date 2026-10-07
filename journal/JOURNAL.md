# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** bobathon-esilv — Parkinson's disease true-OFF MDS-UPDRS motor score regression
- **Goal:** Predict the debiased true OFF motor score (0–132) for every patient visit; minimise visit-level RMSE on the patient-held-out test set.
- **Last experiment:** 09_hgbr_robust_candidate — done
- **Last result:** Robustness batch complete. Best CV result: `05_hgbr_duration` RMSE 7.3751 (fixed splits) — Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47400

<!--
Workspace decisions: one-time project-setup choices.
-->

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2026-10-07
  - env manager: pip+venv - recorded: 2026-10-07
  - agent feature: installed - recorded: 2026-10-07
  - optional features: none - recorded: 2026-10-07
  - package name (`src/<pkg>/`): parkinson - recorded: 2026-10-07
  - skore mode: hub - recorded: 2026-10-07
  - skore hub workspace: teamATN - recorded: 2026-10-07
  - skore mlflow tracking uri: n/a - recorded: 2026-10-07
  - student prior: some-sklearn - recorded: 2026-10-07
  - CV splitter family: GroupKFold - recorded: 2026-10-07

## Data understanding (EDA)

- **Status:** done - 2026-10-07
- **Summary:** 44 590 training visits across 5 576 patients (mean 8 visits/patient, max 12); 11 013 test visits across 1 395 patients with zero overlap. Join is 1:1 with full coverage; sample_submission aligns perfectly with X_test. Target is continuous (mean 37.5, std 16.5, range 0–110), bell-shaped — RMSE is right, dummy-mean floor ≈ 16.5. Joint both-`on`-and-`off`-missing count = 0 (every visit has at least one score). `off` (Pearson 0.89) and `on` (0.69) are the strongest predictors but are 42 % and 30 % missing; LEDD missingness by cohort: A 35.9 %, B 42.6 %. `patient_id` repeats → GroupKFold with patient groups required for honest CV.
- **Report:** [data/eda.md](../data/eda.md)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor(strategy="mean") patient-grouped 5-fold CV baseline | done | RMSE ≈ 16.50 (fold std 0.34, GroupKFold-5, patient-grouped) | [01_dummy.md](01_dummy.md) |
| `02_ridge` | Ridge(alpha=1) + median imputation + StandardScaler, 8 numeric features | done | RMSE ≈ 10.44 (fold std 0.20, GroupKFold-5) — Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/46906 | [02_ridge.md](02_ridge.md) |
| `03_hgbr` | HistGradientBoostingRegressor(max_iter=200, early_stopping=False), native NaN, 8 numeric features | done | RMSE 7.433591 (fold std 0.139028, GroupKFold-5) — Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47082 | [03_hgbr.md](03_hgbr.md) |
| `04_hgbr_categories` | Ablation: add cohort + gene via fold-fitted OrdinalEncoder to 03_hgbr | done | RMSE 7.431442 (fold std 0.120716) Δ=−0.002 vs 03 (negligible) — Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47381 | [04_hgbr_categories.md](04_hgbr_categories.md) |
| `05_hgbr_duration` | Ablation: add years_since_diagnosis = age − age_at_diagnosis to 03_hgbr | done | **RMSE 7.375098** (fold std 0.143036) Δ=**−0.059** vs 03 — Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47400 | [05_hgbr_duration.md](05_hgbr_duration.md) |
| `06_hgbr_missing` | Ablation: add binary indicators off_missing/on_missing/ledd_missing to 03_hgbr | done | RMSE 7.433591 (fold std 0.139028) Δ=0.000 (no benefit) — Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47412 | [06_hgbr_missing.md](06_hgbr_missing.md) |
| `07_hgbr_combined` | Combine 05 (duration) + 04 (categories); 06 excluded (zero gain) | done | RMSE 7.381945 (fold std 0.132252) Δ=−0.052 vs 03 (worse than 05 alone) — Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47430 | [07_hgbr_combined.md](07_hgbr_combined.md) |
| `08_hgbr_robust_baseline` | Robustness: 03_hgbr config on shuffled GroupKFold(n_splits=5,shuffle=True,random_state=42) | done | RMSE 7.437984 (fold std 0.190985, shuffled seed=42) — Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47443 | [08_hgbr_robust_baseline.md](08_hgbr_robust_baseline.md) |
| `09_hgbr_robust_candidate` | Robustness: 05_duration config on same shuffled GroupKFold seed=42 as 08; paired comparison | done | RMSE 7.367803 (fold std 0.190191) paired mean diff vs 08 = **−0.070** (candidate better all 5 folds) — Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/47461 | [09_hgbr_robust_candidate.md](09_hgbr_robust_candidate.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| B1 | Use `off` + `on` with imputation and `age` as features in a simple linear model | `skore:01_dummy` — **done as 02_ridge** |
| B2 | Derive `years_since_diagnosis = age − age_at_diagnosis`; add missingness indicators for `off`, `on`, `ledd` | EDA finding — **done as 05_hgbr_duration (duration) and 06_hgbr_missing (indicators)** |
| B3 | Gradient-boosted trees (HistGradientBoosting) with native missing-value support | EDA finding — **done as 03_hgbr** |
| B4 | Add categorical features (`cohort`, `gene`) via skrub TableVectorizer | EDA — **done as 04_hgbr_categories (OrdinalEncoder; TableVectorizer not used — limitation noted)** |
| B5 | Derive `years_since_diagnosis`; add missingness indicators | B2 carry-forward — **done as 05 and 06** |
| B6 | Try HGBR native categorical support (`categorical_features` parameter) to replace OrdinalEncoder arbitrary codes | 04 limitation noted in BENCHMARKS.md |
| B7 | Tune `max_iter` / `learning_rate` around the 05_hgbr_duration feature set | Next logical step after feature ablation batch |
