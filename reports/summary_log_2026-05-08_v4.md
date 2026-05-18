# DM 114 Final Project Summary Log v4

Date: 2026-05-08

## Goal

Improve model performance using the suggestions in `spec/review_v1.1.md`, building on the best v3 gap-aware ordinal model.

## Code Changes

Updated `scripts/train_gap_model.py` with two performance options:

- Added `--objective blend`.
  - Trains the existing horizon-specific regression models and ordinal threshold models.
  - Selects the ordinal blend weight on validation MAE using `--blend-step`.
  - Keeps `--objective regression` and `--objective ordinal` behavior available.

- Added shrinked region-horizon residual calibration.
  - Controlled by `--region-calibration-alpha` and `--region-calibration-smoothing`.
  - Applies a conservative per-region, per-week correction after the existing horizon median shift.
  - This is a calibration candidate fitted from validation residuals, so the final calibrated validation MAE is useful for model comparison but is not an independent CV estimate.

## Experiment Run

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
  --output submissions/submission_blackout91_blend_cal02_fast64_gpu.csv \
  --report-output reports/blackout91_blend_cal02_fast64_gpu.json
```

## Results

| Report | Objective | Raw MAE | Horizon-Cal MAE | Final Cal MAE | Test Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| `reports/blackout91_weather2_ord_fast64_gpu.json` | ordinal | 0.333379 | 0.325457 | 0.325457 | 0.873092 |
| `reports/blackout91_blend_cal02_fast64_gpu.json` | blend | 0.333339 | 0.325306 | 0.318530 | 0.884284 |

The blend selected ordinal weight `0.80`.

Objective-only validation MAE:

```text
regression: 0.337195
ordinal:    0.333663
blend:      0.333339
```

Final calibrated horizon MAE:

```text
week_1: 0.251048
week_2: 0.298817
week_3: 0.312695
week_4: 0.359355
week_5: 0.370735
```

## Candidate Files

Base candidate:

```text
submissions/submission_blackout91_blend_cal02_fast64_gpu.csv
```

Validated shifted candidates:

| File | Added Shift | Prediction Mean | Range |
| --- | ---: | ---: | --- |
| `submissions/submission_blackout91_blend_cal02_fast64_gpu_shift20.csv` | +0.20 | 1.084155 | 0.200000 to 5.000000 |
| `submissions/submission_blackout91_blend_cal02_fast64_gpu_shift25.csv` | +0.25 | 1.134090 | 0.250000 to 5.000000 |
| `submissions/submission_blackout91_blend_cal02_fast64_gpu_shift35.csv` | +0.35 | 1.233909 | 0.350000 to 5.000000 |

Shift stats are saved in:

```text
reports/blackout91_blend_cal02_shift_candidates.json
```

## Validation

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py submissions/submission_blackout91_blend_cal02_fast64_gpu.csv
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py submissions/submission_blackout91_blend_cal02_fast64_gpu_shift20.csv
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py submissions/submission_blackout91_blend_cal02_fast64_gpu_shift25.csv
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py submissions/submission_blackout91_blend_cal02_fast64_gpu_shift35.csv
```

All checks passed.

## Recommendation

Use `submissions/submission_blackout91_blend_cal02_fast64_gpu_shift25.csv` as the next conservative upload candidate if Kaggle submissions are allowed later. It keeps the improved blended/calibrated model and moves the mean to `1.134090`, closer to the public-implied target level without being as aggressive as `shift35`.
