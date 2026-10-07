# 01_dummy

<!--
Design note for experiments/01_dummy.py (same stem, one to one with
the script). Written and approved before creating the script.
-->

## Question / hypothesis

Does the training mean of `target` (dummy mean) serve as a valid
floor reference for the RMSE metric on patient-grouped cross-validation?

## Motivation

- **Sourcing strategy:** user (lab guide Day 1 + explicit session request)
- **Source(s):**
  - `docs/DAY1.md` § "Dummy mean: a floor" — the guide prescribes a
    `DummyRegressor(strategy="mean")` as the mandatory first baseline;
    its RMSE equals the standard deviation of the target on any
    held-out sample from the same distribution.
  - User instruction: "prefer patient-grouped 5-fold CV for a
    trustworthy reference" — overrides the guide's default `splitter=0.2`
    row holdout for this experiment.
- **Why this matters:** Without a dummy-mean floor, it is impossible to
  know whether a later model's RMSE reflects genuine feature learning or
  a trivially achievable baseline. The floor also catches data-loading
  and metric-computation bugs before any real model is attempted.

## Method

- **Files touched:** `experiments/01_dummy.py` only
  (no `src/<pkg>/` changes — DummyRegressor needs no feature
  engineering)
- **Change versus baseline:** this *is* the baseline. The estimator is
  `sklearn.dummy.DummyRegressor(strategy="mean")`. It ignores all
  features and predicts a constant equal to the training mean of
  `target`.
- **Feature columns used:** `sexM`, `age_at_diagnosis`, `age`, `ledd`,
  `time_since_intake_on`, `time_since_intake_off`, `on`, `off`
  (as prescribed by the lab guide; passed to satisfy skore's X shape,
  not used by the dummy).
- **Cross-validation:** `GroupKFold(n_splits=5)` with `groups=patient_id`.
  Splits are pre-computed outside `skore.evaluate` via
  `list(GroupKFold(n_splits=5).split(X, y, groups=groups))` and passed
  as a generator so the splitter receives the group array directly,
  without relying on sklearn metadata routing through `skore.evaluate`.
- **Hub key:** `01_dummy` in project `bobathon-esilv`, workspace
  `teamATN`.
- **Submission file:** `submissions/01_dummy.csv` (refit on all train,
  predict X_test, header `Index,target`).
- **Out of scope for this experiment:** imputation, feature engineering,
  any non-trivial learner.

## Risks / things that could invalidate the result

- GroupKFold requires that no patient appears on both sides of a fold.
  If `groups` is not wired correctly into the splitter the RMSE will
  be optimistically biased (same as a random row holdout). Verified
  by checking patient overlap between folds in the run output.
- The dummy RMSE should be close to `std(target)` ≈ 16.5. A large
  deviation would indicate a data-loading bug.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** RMSE ≈ 16.50 (fold std 0.34 across 5 patient-grouped folds — this is the spread between folds, not a confidence interval); submission target constant 37.47; Hub: https://skore.probabl.ai/teamATN/bobathon-esilv/cross-validations/46521
- **Implication for next iteration:** The dummy floor is confirmed: RMSE ≈ std(target) ≈ 16.5, as expected. Any model using `off` (Pearson 0.89) and `on` (0.69) with imputation should substantially beat this. Future experiments must use the same patient-grouped folds (pre-computed `GroupKFold(n_splits=5)` splits on `patient_id`) so results are comparable; row-level holdouts are not an honest substitute for this dataset.
