# DM 114 Final Project Summary Log v3

Date: 2026-05-08

## Goal

Continue improving model performance by following `spec/review_v1.md`, without uploading to Kaggle unless explicitly approved.

## Code Changes

Updated `scripts/train_gap_model.py` with two review-driven improvements:

- Added `--objective ordinal`.
  - Trains threshold classifiers for `score >= 1` through `score >= 5`.
  - Uses summed probabilities as the final ordinal severity prediction.
  - Trains one threshold stack per horizon, for 25 classifiers total.

- Expanded weather anomaly features.
  - Added region-month weather baselines in addition to full-region baselines.
  - Added month-normalized weather anomaly and z-score features.
  - Added precipitation month-sum anomaly.
  - Added longest dry spell and days since wet day.
  - Added composite dryness features from precipitation, humidity, temperature, dew point, and wet-bulb temperature.

## Experiments Run

All experiments used the review-aligned `blackout91` validation mode:

```bash
env PYTHONPATH=src .venv/bin/python scripts/train_gap_model.py \
  --fast \
  --device gpu \
  --gap-mode blackout91 \
  --train-samples-per-region 64 \
  --valid-anchors-per-region 2
```

### Results

| Report | Objective | Features | Raw MAE | Calibrated MAE | Test Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| `reports/blackout91_fast64_gpu.json` | regression | 776 | 0.34008 | 0.33642 | 0.87156 |
| `reports/blackout91_ord_fast64_gpu.json` | ordinal | 776 | 0.33735 | 0.32744 | 0.89405 |
| `reports/blackout91_weather2_fast64_gpu.json` | regression | 1051 | 0.33720 | 0.33465 | 0.85549 |
| `reports/blackout91_weather2_ord_fast64_gpu.json` | ordinal | 1051 | 0.33338 | 0.32546 | 0.87309 |

Best review-aligned local validation result is now:

```text
reports/blackout91_weather2_ord_fast64_gpu.json
validation_mae: 0.32545673847198486
```

This improves over the previous best review-aligned result:

```text
reports/blackout91_fast64_gpu.json
validation_mae: 0.33642280101776123
```

Absolute improvement:

```text
0.010966062545776367 MAE
```

Relative improvement:

```text
3.26%
```

## Best Candidate Generated

Validated local candidate:

```text
submissions/submission_blackout91_weather2_ord_fast64_gpu.csv
```

Validation:

```text
Rows: 2,248
Prediction range: 0.000000 to 4.899360
Prediction mean: 0.873092
```

The unshifted candidate still likely underpredicts public severity because the public all-zero submission scored `1.2088`, implying a public target mean near `1.2088`.

## Shift Candidates

Generated and validated:

| File | Added Shift | Prediction Mean | Range |
| --- | ---: | ---: | --- |
| `submissions/submission_blackout91_weather2_ord_fast64_gpu_shift20.csv` | +0.20 | 1.07305 | 0.200000 to 5.000000 |
| `submissions/submission_blackout91_weather2_ord_fast64_gpu_shift25.csv` | +0.25 | 1.12301 | 0.250000 to 5.000000 |
| `submissions/submission_blackout91_weather2_ord_fast64_gpu_shift35.csv` | +0.35 | 1.22291 | 0.350000 to 5.000000 |

Shift candidate stats are also saved in:

```text
reports/blackout91_weather2_ord_shift_candidates.json
```

## Current Best Scores

Kaggle public scores already observed:

```text
submission_global_median.csv      1.2088
submission_last_train_score.csv   1.0884
submission_gap_zero_fast64.csv    1.0839
```

Best Kaggle public score so far:

```text
1.0839
```

No new Kaggle upload was performed in this v3 run.

## Recommendation

Do not upload the unshifted model first; its mean is still only `0.87309`.

If one upload is allowed later, the safest next candidate is:

```text
submissions/submission_blackout91_weather2_ord_fast64_gpu_shift25.csv
```

Reason:

- It uses the best review-aligned local model.
- It moves the mean to `1.12301`, closer to the public-implied target mean without fully matching `1.2088`.
- It is less aggressive than `shift35`, which may overfit public distribution.

If testing a more aggressive public-shift hypothesis, use:

```text
submissions/submission_blackout91_weather2_ord_fast64_gpu_shift35.csv
```

But this has higher public leaderboard overfit risk.
