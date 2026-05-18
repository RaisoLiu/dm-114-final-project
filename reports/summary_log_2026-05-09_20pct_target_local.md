# 20 Percent Improvement Target - Local Work

Date: 2026-05-09

## Target

Current best known Kaggle public score:

```text
0.8773
```

A 20 percent relative improvement means approximately:

```text
0.8773 * 0.80 = 0.7018
```

No further Kaggle submissions should be made without explicit approval because only one submission remains.

## Public Feedback Used

Submitted before the no-more-upload constraint was clarified:

```text
submission_blackout91_blend_cal02_fast64_gpu_shift25.csv  public: 0.8804
submission_blackout91_blend_cal02_fast64_gpu_shift35.csv  public: 0.8773
submission_blackout91_blend_valid1_fast64_gpu_shift35.csv public: 0.8921
```

The best public score is now `0.8773`. The latest-anchor validation candidate did not transfer to public leaderboard.

## Code Improvement

Added `--refit-all-for-test` to `scripts/train_gap_model.py`.

Before this change, `train_gap_model.py` selected settings on validation and then predicted test with the validation-trained model, leaving the validation rows unused for the final test model. The new option keeps validation metrics comparable, then refits the selected objective on train+validation rows before predicting test.

Validation:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python -m py_compile scripts/train_gap_model.py
```

Both passed.

## Local Experiments

### More sampled anchors

```text
reports/blackout91_blend_cal02_fast128_gpu.json
validation MAE: 0.320394
week_5 MAE:     0.370913
test mean:      0.868556
```

This was worse than the 64-sample best and is not recommended.

### Refit-all candidate

```text
reports/blackout91_blend_cal02_refitall_fast64_gpu.json
validation MAE: 0.317471
raw MAE:        0.331923
week_5 MAE:     0.372298
test mean:      0.935443
```

Shift candidates:

```text
submissions/submission_blackout91_blend_cal02_refitall_fast64_gpu_shift25.csv mean 1.185123
submissions/submission_blackout91_blend_cal02_refitall_fast64_gpu_shift30.csv mean 1.235008
submissions/submission_blackout91_blend_cal02_refitall_fast64_gpu_shift35.csv mean 1.284879
submissions/submission_blackout91_blend_cal02_refitall_fast64_gpu_shift40.csv mean 1.334743
```

### Hedge ensemble candidates

Ensembled the known public-best file with the refit-all shift30 file:

```text
known public-best: submissions/submission_blackout91_blend_cal02_fast64_gpu_shift35.csv
refit-all candidate: submissions/submission_blackout91_blend_cal02_refitall_fast64_gpu_shift30.csv
mean abs prediction difference: 0.097235
```

Validated ensemble files:

```text
submissions/submission_ensemble_publicbest_refitall_w35.csv mean 1.234294
submissions/submission_ensemble_publicbest_refitall_w50.csv mean 1.234459
submissions/submission_ensemble_publicbest_refitall_w65.csv mean 1.234624
submissions/submission_ensemble_publicbest_refitall_w80.csv mean 1.234789
```

## Recommendation For Last Submission

The safest final candidate is:

```text
submissions/submission_ensemble_publicbest_refitall_w50.csv
```

Reason:

- Keeps 50 percent of the known public-best submission (`0.8773`).
- Uses 50 percent of the improved refit-all model.
- Maintains the public-favorable prediction mean around `1.234`.
- Reduces risk compared with fully switching to an unsubmitted refit-all model.

This candidate is not guaranteed to reach the 20 percent target. Based on public feedback so far, a 20 percent jump is unlikely from global shifts alone; the refit-all/ensemble candidate is the most defensible final attempt found locally.
