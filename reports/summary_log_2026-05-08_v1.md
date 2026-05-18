# DM 114 Final Project Summary Log

Date: 2026-05-08

## Objective

Initial target: follow `spec/v1.md`, run the project pipeline, and try to reach Kaggle score under `0.7`.

Later local target discussed: validation MAE under `0.1`. Current evidence shows only week1 validation is under `0.1`; the 5-week average is not.

## Data And Pipeline Fixes

- Fixed synthetic date handling.
  - The competition data uses years such as `58061`, which pandas `Timestamp` cannot represent.
  - Updated date handling to use string parsing and integer calendar features.

- Fixed validation split logic.
  - Earlier validation by absolute synthetic year was not meaningful across regions.
  - Changed validation to use each region's relative `anchor_index`.

- Confirmed data integrity.
  - Train rows: `12,319,040`
  - Test rows: `204,568`
  - Regions: `2,248`
  - Non-null score labels: `1,757,936`
  - Score range: `[0, 5]`

- Confirmed GPU availability.
  - GPU: `NVIDIA GB10`
  - LightGBM GPU training works.

## Kaggle Submissions

| File | Method | Local Validation | Kaggle Public Score |
|---|---:|---:|---:|
| `submissions/submission_global_median.csv` | all-zero/global median | `0.6051` | `1.2088` |
| `submissions/submission_last_train_score.csv` | repeat last known train score per region | `0.2368` | `1.0884` |
| `submissions/submission_gap_zero_fast64.csv` | GPU LightGBM, score/weather/date features, zero-gap validation | `0.21249` | `1.0839` |

Best Kaggle public score so far: `1.0839`.

## Local Experiments

### Baselines

- Global median/all-zero baseline:
  - Validation MAE: `0.6051002600602244`
  - Public score: `1.2088`

- Last observed score baseline:
  - Validation MAE: `0.2368498494388174`
  - Public score: `1.0884`

### Gap-Aware GPU LightGBM

Created `scripts/train_gap_model.py`.

Features:
- 91-day weather window statistics
- Region encoding
- Date/month/day-of-year features
- Historical score lag/stat features
- Horizon-specific LightGBM models
- Optional GPU training
- Optional calibration by validation residual median

Important reports:

| Report | Mode | Local Validation MAE | Notes |
|---|---:|---:|---|
| `reports/gap_model_fast128_gpu_cal.json` | public-like stale score gap | `0.44252` | Better local model, not uploaded |
| `reports/gap_zero_fast64.json` | zero-gap/current score history | `0.21249` | Uploaded, public `1.0839` |

Best local 5-week validation MAE so far: `0.21249`.

Best local per-horizon result:

```text
week1  0.07648
week2  0.15747
week3  0.23003
week4  0.28342
week5  0.31508
overall 0.21249
```

### Weekly Score-Lag Model

Created `scripts/train_score_lag_model.py`.

Purpose: test whether weekly score history alone can predict the next five labels.

Result from `reports/score_lag_validation_fast.json`:

```text
validation_mae: 0.26112
persistence_mae: 0.23685
week1: 0.10204
week2: 0.18800
week3: 0.26905
week4: 0.33592
week5: 0.41059
```

Conclusion: score-lag model did not beat simple persistence on the 5-week average.

### Weather-To-Current-Score Diagnostic

Tested whether weather features can reconstruct current weekly score directly.

Result:

```text
MAE: 0.66025
rounded MAE: 0.63336
```

Conclusion: weather features alone are not enough to reconstruct labels accurately.

## Key Finding

Local validation was too optimistic for Kaggle public.

Evidence:
- All-zero public score is `1.2088`, so the public target mean is around `1.2088`.
- Several local validation splits had target means around `0.54` to `0.84`.
- This distribution shift explains why local MAE values such as `0.2368` or `0.21249` did not transfer to Kaggle public.

## Current Status

- Best Kaggle public score: `1.0839`
- Best local 5-week validation MAE: `0.21249`
- Best local week1 validation MAE: `0.07648`
- Target `<0.7` on Kaggle: not achieved
- Target `<0.1` on 5-week local validation: not achieved
- Target `<0.1` on week1 local validation: achieved

## Notes

- No further Kaggle uploads should be done without explicit approval.
- Local credential files are ignored by `.gitignore`.
- The best public score came from `submission_gap_zero_fast64.csv`, but it is only a slight improvement over last-score persistence.

## Recommended Next Steps

1. Rebuild validation around the public distribution shift rather than adjacent future labels.
2. Use public feedback to calibrate target mean more carefully, but avoid overfitting the leaderboard.
3. Try region/month/lead-time calibrated ensembles from:
   - gap-aware GPU LightGBM
   - last-score persistence
   - seasonal month mean
4. Do not submit again until the daily limit resets and a new candidate has a clear rationale.
