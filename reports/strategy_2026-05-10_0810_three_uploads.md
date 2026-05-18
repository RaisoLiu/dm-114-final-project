# Strategy for 2026-05-10 08:10 Three Uploads

## Objective

Upload three prediction files by **2026-05-10 08:10 Asia/Taipei** and maximize the chance of a Kaggle public MAE below `0.8`.

Current best public score:

```text
submission_blackout91_blend_cal02_fast64_gpu_shift35.csv  0.8773
```

Target:

```text
< 0.8
```

This requires roughly `8.8%` improvement from the current best `0.8773`.

## What The Existing Evidence Says

### Public leaderboard feedback

| File | Role | Mean | Public MAE |
| --- | ---: | ---: | ---: |
| `submission_global_median.csv` | all-zero baseline | 0.000 | 1.2088 |
| `submission_last_train_score.csv` | label persistence baseline | 0.346 | 1.0884 |
| `submission_gap_zero_fast64.csv` | optimistic local validation model | 0.327 | 1.0839 |
| `submission_blackout91_blend_cal02_fast64_gpu_shift25.csv` | standard model + shift25 | 1.134 | 0.8804 |
| `submission_blackout91_blend_cal02_fast64_gpu_shift35.csv` | standard model + shift35 | 1.234 | 0.8773 |
| `submission_blackout91_blend_valid1_fast64_gpu_shift35.csv` | latest-anchor model | 1.197 | 0.8921 |

Main lessons:

- The public data needs a higher prediction distribution than the raw local models produce.
- Moving the standard model from shift25 to shift35 helped only slightly (`0.8804` to `0.8773`), so global shifting alone is not enough to reach `<0.8`.
- The latest-anchor validation setup looked excellent locally but did not transfer to public (`0.8921`), so it should not be trusted as a main final strategy.
- The useful public model family is the standard `blackout91` blend pipeline.

### Local validation feedback

| Report | Key Setting | Validation MAE | Week 5 MAE | Test Mean |
| --- | --- | ---: | ---: | ---: |
| `blackout91_blend_cal02_fast64_gpu.json` | current standard blend | 0.318530 | 0.370735 | 0.884284 |
| `blackout91_blend_cal02_refitall_fast64_gpu.json` | standard blend + final refit on train+valid | 0.317471 | 0.372298 | 0.935443 |
| `blackout91_blend_cal02_fast128_gpu.json` | more sampled anchors | 0.320394 | 0.370913 | 0.868556 |
| `blackout91_blend_cal100_fast64_gpu.json` | aggressive region calibration | 0.254453 | 0.284957 | 1.046754 |

Main lessons:

- `fast128` did not help.
- `refit-all` is the best conservative local improvement because it fixes the final training procedure.
- `cal100` is not a clean validation estimate because it uses validation residual calibration, but its predictions are meaningfully different and may be useful as a high-risk diversity candidate.

## Candidate Diversity

Mean absolute prediction difference versus the current public best file:

| Candidate | Mean | MAD vs public best | Corr vs public best |
| --- | ---: | ---: | ---: |
| `submission_ensemble_publicbest_refitall_w50.csv` | 1.234459 | 0.048618 | 0.99649 |
| `submission_blackout91_blend_cal02_refitall_fast64_gpu_shift30.csv` | 1.235008 | 0.097235 | 0.98641 |
| `submission_post_publicbest_sharp120_mkeep.csv` | 1.233909 | 0.115043 | derived from public best |
| `submission_post_refit30_sharp120_mkeep.csv` | 1.235008 | similar refit-sharpen family | derived from refit-all |
| `submission_post_publicbest_hsharp_opt_mkeep.csv` | 1.233909 | 0.173331 | 0.99721 |
| `submission_post_publicbest_hybrid_qsharp_opt_mkeep.csv` | 1.233909 | 0.334135 | 0.97255 |
| `submission_post_publicbest_hqmap_opt_mkeep.csv` | 1.233909 | 0.512569 | 0.96067 |
| `submission_post_publicbest_qmap100.csv` | 1.208808 | 0.685417 | 0.95313 |
| `submission_blackout91_blend_cal100_fast64_gpu_shift20.csv` | 1.245536 | 0.266955 | 0.92864 |
| `submission_gap_fast128_gpu_cal_shift40.csv` | 1.136082 | 0.430251 | 0.52335 |

Interpretation:

- `ensemble_w50` is safest but may only make a small move.
- `refit_shift30` is a stronger version of the same model family.
- `hybrid_qsharp` is the strongest current validation-postprocess candidate and keeps the public-best per-week means.
- `hqmap` is more aggressive than `hybrid_qsharp`; it has higher variance and upside but more overfit risk.
- `qmap100` is the most aggressive public-best ranking submission and directly uses the public all-zero score to set the public target mean.
- `cal100_shift20` remains a useful high-risk, higher-diversity backup candidate.
- `gap128_shift40` is too different and the gap-family public evidence is weak; reserve it only if we want a pure long-shot.

## Recommended Three Uploads

### Upload 1 - Public-Best Horizon-Specific Sharpening

```text
submissions/submission_post_publicbest_hsharp_opt_mkeep.csv
```

Reason:

- Starts from the current public-best submission (`0.8773`).
- Keeps the same public-favorable mean (`1.233909`).
- Sharpens each horizon separately, lowering low-risk predictions and raising high-risk predictions.
- Validation-prediction diagnostics selected scales `1.25, 1.25, 1.30, 1.35, 1.40` for weeks 1-5.
- This improved diagnostic validation MAE from `0.314839` to `0.287728` while keeping each horizon mean fixed.
- This directly tests whether the model ranking is useful but the current predictions are over-smoothed.

Expected role:

```text
Best evidence-based non-shift attempt to improve over 0.8773.
```

### Upload 2 - Hybrid Horizon QMap + Sharpening

```text
submissions/submission_post_publicbest_hybrid_qsharp_opt_mkeep.csv
```

Reason:

- Starts from the same current public-best submission and preserves each week's mean.
- Combines the two strongest validation-postprocess ideas: partial qmap and mild horizon-specific sharpening.
- Validation-prediction diagnostics selected qmap blends/scales:
  - week 1: blend `0.75`, scale `1.00`
  - week 2: blend `0.75`, scale `1.00`
  - week 3: blend `0.30`, scale `1.05`
  - week 4: blend `0.30`, scale `1.05`
  - week 5: blend `0.35`, scale `0.95`
- This improved diagnostic validation MAE from `0.314839` to `0.277918`, better than both horizon-sharpen (`0.287728`) and horizon-qmap (`0.285052`).
- Prediction mean remains `1.233909`; mean absolute difference from public best is `0.334135`, between `hsharp` and `hqmap`.

Expected role:

```text
Best current validation-backed attempt to improve shape while staying in the public-best ranking family.
```

### Upload 3 - Full Public-Mean Quantile Mapping

```text
submissions/submission_post_publicbest_qmap100.csv
```

Reason:

- Uses the fact that all-zero public MAE `1.2088` implies public target mean is about `1.2088`.
- Keeps the current public-best ranking but fully maps ranks to a tilted integer severity distribution.
- Prediction mean is `1.208808`, matching the public mean implied by the all-zero baseline.
- Validation mean+shape diagnostics support full qmap when target mean is known: global train-tilt qmap to validation truth mean scored `0.279982`, close to the hybrid diagnostic `0.277918`.
- This is the largest ranking-family move in the queue: mean absolute difference from public best is `0.685417`.

Risk:

```text
Very large distribution change: std 1.497772 and hard 0..5 quantile levels.
Could over-discretize and overfit the public mean signal if public/private target shape differs from validation.
```

Expected role:

```text
Maximum-upside third shot. If the smoother shape corrections are not enough for <0.8, this is the bigger distribution correction.
```

### Backup Horizon-Specific QMap

If the first two submissions improve but full qmap looks too aggressive for the final slot, use:

```text
submissions/submission_post_publicbest_hqmap_opt_mkeep.csv
```

Reason:

- Keeps the public-best per-week means at `1.233909`.
- Uses validation-selected qmap blends `0.80, 0.80, 0.70, 0.75, 0.80`.
- Diagnostic validation MAE is `0.285052`.
- It is less extreme than full qmap: MAD vs public best is `0.512569` rather than `0.685417`.

### Backup Diversity Shot - Cal100 Shift20

If the first two post-processing submissions are bad enough that staying in the public-best ranking family looks wrong, replace Upload 3 with:

```text
submissions/submission_blackout91_blend_cal100_fast64_gpu_shift20.csv
```

Reason:

- Prediction mean `1.245536`, still in the public-favorable range.
- More different from public best than refit/ensemble candidates.
- Uses aggressive region calibration, which may correct systematic regional underprediction.

Risk:

```text
Validation is not clean because region residual calibration is fit from validation residuals.
Could overfit local validation and fail public/private.
```

Default queue now keeps all three submissions inside the current public-best ranking family because all submitted public feedback so far favors that family.

## Upload Order

If tomorrow allows exactly three submissions, use this order:

1. `submission_post_publicbest_hsharp_opt_mkeep.csv`
2. `submission_post_publicbest_hybrid_qsharp_opt_mkeep.csv`
3. `submission_post_publicbest_qmap100.csv`

Why this order:

- First submission uses the strongest public evidence and changes only the prediction shape.
- Second uses the strongest current validation-postprocess diagnostic while keeping public-best means.
- Third spends the final slot on the most aggressive public-mean distribution correction.

## Anchor Split Risk Check

To check whether the post-processing is only overfitting one validation slice, the two validation anchors per region were split into latest-anchor rows and older-anchor rows.

Source:

```text
reports/postprocess_anchor_split_diagnostics.json
```

Base validation proxy:

| Split | Base MAE |
| --- | ---: |
| latest anchor | 0.170923 |
| older anchor | 0.458754 |

Fixed full-validation post-processing applied separately to each split:

| Candidate | Overall MAE | Latest MAE | Older MAE | Latest improvement | Older improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hsharp` | 0.287783 | 0.131016 | 0.444551 | 0.039908 | 0.014203 |
| `hqmap` | 0.283356 | 0.127047 | 0.439665 | 0.043876 | 0.019089 |
| `hybrid_qsharp` | 0.277918 | 0.128648 | 0.427188 | 0.042276 | 0.031566 |

Full qmap was also checked with the validation target mean treated as known, matching how the public all-zero score exposes the public mean:

```text
reports/full_qmap_anchor_split_diagnostics.json
```

| Candidate | Overall MAE | Latest MAE | Older MAE | Latest improvement | Older improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| global train-tilt qmap100 to known mean | 0.279982 | 0.146886 | 0.413078 | 0.024037 | 0.045676 |

Interpretation:

- `hsharp` is the most robust when hyperparameters are tuned on one anchor split and evaluated on the other.
- `hybrid_qsharp` is more tuned, but its fixed parameters still improve both latest and older anchor splits, and it has the best overall diagnostic.
- Full qmap is a larger distribution bet, but it also improves both anchor splits when the target mean is known.
- This supports `hsharp` first, `hybrid_qsharp` second, and full public-mean qmap as the final upside shot.

## Quota Timing Risk

The Kaggle CLI currently shows three successful submissions on `2026-05-09`:

```text
2026-05-09 01:23:40
2026-05-09 01:34:02
2026-05-09 01:35:09
```

After those three, another upload attempt returned `400 Bad Request`, consistent with a three-submission quota.

If Kaggle uses a calendar-day quota, the planned **2026-05-10 08:10 Asia/Taipei** upload should be fine.

If Kaggle uses a rolling 24-hour quota and the CLI timestamps are UTC, then `2026-05-10 08:10 Asia/Taipei` is only `2026-05-10 00:10 UTC`, which is before the first `2026-05-09 01:23 UTC` submission ages out. In that case, the three-submit window may not fully reopen until after:

```text
2026-05-10 09:36 Asia/Taipei
```

The upload script now prints a quota preflight warning. If it warns or the first submission receives `400 Bad Request`, wait until after `09:36 Asia/Taipei` before retrying.

The direct upload script also refuses `--execute` before `2026-05-10 08:10:00 Asia/Taipei` unless `--force-before-target` is passed. This is only to prevent accidental early uploads.

If a later submission fails after an earlier one already succeeded, resume from the next queue item instead of resubmitting the successful file:

```bash
bash scripts/submit_three_20260510_0810.sh --execute --start-at 2
bash scripts/submit_three_20260510_0810.sh --execute --start-at 3
```

Dry-run still validates all three queued files; `--start-at` only affects the actual submit loop in `--execute` mode.

Optional wait wrapper:

```bash
bash scripts/wait_until_submit_20260510_0810.sh --execute
```

This waits until `2026-05-10 08:10:00 Asia/Taipei` and then calls the submit script. Its dry-run mode does not wait and does not upload; it only validates the current queue.

Recommended resilient runner:

```bash
bash scripts/resilient_submit_20260510_0810.sh --execute
```

This verifies the manifest, waits for 08:10, attempts the queue, and if the attempt fails, checks which manifest files are visible in Kaggle submissions and resumes after `2026-05-10 09:36 Asia/Taipei` from the first missing queue item.

## If Only One Submission Is Actually Available

Use:

```text
submissions/submission_post_publicbest_hsharp_opt_mkeep.csv
```

It is the most defensible attempt to improve meaningfully from the known public best without changing the public-favorable mean.

## Completion Criteria Before Uploading

Before 2026-05-10 08:10, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_0810_manifest.py
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py submissions/submission_post_publicbest_hsharp_opt_mkeep.csv
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py submissions/submission_post_publicbest_hybrid_qsharp_opt_mkeep.csv
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py submissions/submission_post_publicbest_qmap100.csv
```

Do not upload unless the files validate and Kaggle shows that submissions are available.

## Reality Check

The current evidence does not prove that `<0.8` is reachable with existing models. The jump from `0.8773` to `<0.8` likely requires either:

- a new model family with genuinely different signal, or
- a better way to infer public/test distribution than global shifts.

The three recommended files are the strongest use of existing experiments, but they should be treated as a calculated attempt, not a guaranteed path to `<0.8`.
