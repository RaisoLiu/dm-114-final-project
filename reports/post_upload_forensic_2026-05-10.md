# Post-Upload Forensic Report: 2026-05-10

## Objective Status

Original objective:

- Upload three prediction files for the `2026-05-10 08:10 Asia/Taipei` target.
- Obtain a Kaggle public MAE below `0.8`.

Actual outcome:

- Three uploads were completed after explicit user approval, starting at `2026-05-10 09:49 Asia/Taipei`.
- The `<0.8` public score target was not reached.
- Best visible public score remains `0.8773` from `submission_blackout91_blend_cal02_fast64_gpu_shift35.csv`.

## Kaggle Evidence

| File | Public MAE | Notes |
| --- | ---: | --- |
| `submission_blackout91_blend_cal02_fast64_gpu_shift25.csv` | `0.8804` | Standard blackout91 blend, shift +0.25 |
| `submission_blackout91_blend_cal02_fast64_gpu_shift35.csv` | `0.8773` | Current best visible public score |
| `submission_blackout91_blend_valid1_fast64_gpu_shift35.csv` | `0.8921` | Latest-anchor validation did not transfer |
| `submission_post_publicbest_hsharp_opt_mkeep.csv` | `0.9013` | New upload 1; horizon sharpening worsened public |
| `submission_post_publicbest_hybrid_qsharp_opt_mkeep.csv` | `0.9490` | New upload 2; hybrid qmap/sharpen worsened further |
| `submission_post_publicbest_qmap100.csv` | `1.0352` | New upload 3; public-mean full qmap failed |

The public feedback is monotonic in the wrong direction for the post-processing family:

| Candidate | MAD vs public best | Public delta vs `0.8773` |
| --- | ---: | ---: |
| `hsharp` | `0.173331` | `+0.0240` |
| `hybrid_qsharp` | `0.334135` | `+0.0717` |
| `qmap100` | `0.685417` | `+0.1579` |

Interpretation: keeping the current public-best ranking and making the distribution sharper or more quantized does not help on public. The validation post-processing diagnostics were over-optimistic.

## Validation/Public Mismatch

The all-zero submission scored `1.2088`, and labels are non-negative, so the public target mean is approximately `1.2088`.

The previous main validation setup used two offsets from each region's last usable training anchor:

| Validation offset | Future target mean |
| ---: | ---: |
| `0` days | `0.342260` |
| `365` days | `0.739324` |
| combined old setup | about `0.5408` |

This is far below the public-implied target mean. That explains why post-processing tuned on the old validation predictions preferred sharpening/qmap: it was optimizing a much lower-severity slice.

A scan of older anchors found a more public-like severity slice:

| Validation offset | Future target mean | Future target std |
| ---: | ---: | ---: |
| `735` days | `1.174288` | `1.390920` |

This is still below public `1.2088`, but much closer than the old validation split.

## Code/Experiments Added

Added `--valid-deltas` to `scripts/train_gap_model.py` so validation offsets can be explicitly controlled instead of fixed to `linspace(0,365,n)`.

Also fixed a feature bug in `scripts/train_gap_model.py`: `build_weather_stats()` was parsing months with fixed string positions. Competition dates use wide synthetic years such as `58061-...`, so fixed slicing mis-parsed months and polluted the region-month weather baselines. The script now parses dates with `rsplit("-", 2)`, and `tests/test_pipeline.py` covers wide synthetic years.

Month-fix standard validation run, matching the previous public-best model family:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_gap_model.py \
  --fast \
  --device gpu \
  --gap-mode blackout91 \
  --train-samples-per-region 64 \
  --valid-anchors-per-region 2 \
  --objective blend \
  --blend-step 0.05 \
  --region-calibration-alpha 0.2 \
  --region-calibration-smoothing 4.0 \
  --validation-pred-output reports/blackout91_blend_cal02_fast64_gpu_monthfix_validation_predictions.csv \
  --output submissions/submission_blackout91_blend_cal02_fast64_gpu_monthfix.csv \
  --report-output reports/blackout91_blend_cal02_fast64_gpu_monthfix.json
```

Comparison to the previous same-family run:

| Report | Final validation MAE | Week 5 MAE | Test mean | Selected ordinal weight |
| --- | ---: | ---: | ---: | ---: |
| old `blackout91_blend_cal02_fast64_gpu.json` | `0.318530` | `0.370735` | `0.884284` | `0.80` |
| new `blackout91_blend_cal02_fast64_gpu_monthfix.json` | `0.314379` | `0.368115` | `0.901436` | `1.00` |

Validated local month-fix shifted candidates:

| File | Mean | MAD vs scored public best |
| --- | ---: | ---: |
| `submission_blackout91_blend_cal02_fast64_gpu_monthfix_shift25.csv` | `1.151265` | `0.144572` |
| `submission_blackout91_blend_cal02_fast64_gpu_monthfix_shift30.csv` | `1.201185` | `0.128073` |
| `submission_blackout91_blend_cal02_fast64_gpu_monthfix_shift35.csv` | `1.251103` | `0.125947` |
| `submission_blackout91_blend_cal02_fast64_gpu_monthfix_shift40.csv` | `1.301016` | `0.141238` |

These are not uploaded. If another upload is later approved, `monthfix_shift30` is the cleanest mean-aligned candidate because its mean `1.201185` is closest to the public-implied all-zero mean `1.2088`. This is still not evidence that it can reach `<0.8`.

Month-fix high-severity diagnostic run:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_gap_model.py \
  --fast \
  --device gpu \
  --gap-mode blackout91 \
  --train-samples-per-region 64 \
  --valid-deltas 735 \
  --objective blend \
  --blend-step 0.05 \
  --region-calibration-alpha 0.2 \
  --region-calibration-smoothing 4.0 \
  --validation-pred-output reports/blackout91_blend_cal02_fast64_gpu_monthfix_valid735_validation_predictions.csv \
  --output submissions/submission_blackout91_blend_cal02_fast64_gpu_monthfix_valid735.csv \
  --report-output reports/blackout91_blend_cal02_fast64_gpu_monthfix_valid735.json
```

Result:

| Metric | Value |
| --- | ---: |
| Raw validation MAE | `0.411669` |
| Final calibrated validation MAE | `0.390860` |
| Validation target mean | `1.174288` |
| Validation target std | `1.390920` |
| Selected ordinal blend weight | `1.00` |
| Test prediction mean | `0.878798` |

Week MAE on the `735`-day validation slice:

| Week | MAE |
| ---: | ---: |
| 1 | `0.357620` |
| 2 | `0.390590` |
| 3 | `0.390322` |
| 4 | `0.401267` |
| 5 | `0.414501` |

This high-severity validation slice is substantially harder than the old validation setup and better explains the public leaderboard ceiling.

## Shift Check On High-Severity Validation

On the `735`-day validation predictions, adding a large positive global shift was not supported:

| Prediction column | Base MAE | Best tested shift | Best MAE | MAE at `+0.35` |
| --- | ---: | ---: | ---: | ---: |
| `pred_raw` | `0.411669` | `-0.05` | `0.407149` | `0.512743` |
| `pred_horizon_calibrated` | `0.407146` | `0.00` | `0.407146` | `0.497229` |
| `pred_final_calibrated` | `0.390860` | `0.00` | `0.390860` | `0.483246` |

This means the public benefit of `shift35` is probably not a general validation-backed correction. It is correcting a specific test/public distribution mismatch where the model's test prediction mean is too low, not a universal high-drought validation rule.

## Current Decision

Do not make more uploads without explicit user approval.

The strongest scored artifact remains:

```text
submissions/submission_blackout91_blend_cal02_fast64_gpu_shift35.csv
public MAE: 0.8773
```

There is no current local or public evidence that any existing candidate is likely to reach `<0.8`. The failed uploads specifically rule out the largest public-best ranking/post-processing bets.

The best not-yet-uploaded local candidate is now:

```text
submissions/submission_blackout91_blend_cal02_fast64_gpu_monthfix_shift30.csv
```

Reason:

- It fixes a real feature bug in month-normalized weather baselines.
- It improves comparable local validation from `0.318530` to `0.314379`.
- Its prediction mean `1.201185` is closest to the public-implied mean `1.2088`.
- It is a model-feature change, not another failed qmap/sharpening transform.

Risk:

- Its MAD versus the scored public best is `0.128073`, and the failed `hsharp` candidate had MAD `0.173331`, so it may still lose on public.
- The public score target `<0.8` remains unsupported by evidence.

## Next Productive Work

Before any future upload, the validation design should be changed:

1. Use explicit high-severity validation offsets, starting with `735`, and possibly add `728`, `1460`, and `1825`.
2. Report validation target mean/std for every experiment; do not compare scores across validation slices without those distribution stats.
3. Stop tuning qmap/sharpening on the old low-mean validation predictions.
4. Prefer model-family changes or validation-slice robustness over leaderboard-derived distribution surgery.
5. Treat any future shift-only candidate as a small calibration test, not a credible path to `<0.8` by itself.

Verification after changes:

```text
PYTHONPATH=src .venv/bin/python -m py_compile scripts/train_gap_model.py
PYTHONPATH=src .venv/bin/python -m pytest -q
bash -n scripts/submit_three_20260510_0810.sh scripts/resilient_submit_20260510_0810.sh scripts/wait_until_submit_20260510_0810.sh
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py submissions/submission_blackout91_blend_cal02_fast64_gpu_monthfix.csv
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py submissions/submission_blackout91_blend_cal02_fast64_gpu_monthfix_shift30.csv
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py submissions/submission_blackout91_blend_cal02_fast64_gpu_monthfix_valid735.csv
```

All listed checks passed.
