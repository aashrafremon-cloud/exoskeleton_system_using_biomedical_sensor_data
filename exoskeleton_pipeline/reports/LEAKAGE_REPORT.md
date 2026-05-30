# LEAKAGE_REPORT

## Risk
Random train/test split assigns windows from the same trial to both sets → optimistic bias.

## Mitigation (implemented in train.py)
- `trial_id` per file pair
- **GroupKFold** (5 folds) for CV — groups = trial_id
- Hold-out: 20% of **trials** (not windows) for test metrics

## Verdict
Leakage controlled when using `train.py` GroupKFold + trial hold-out.
