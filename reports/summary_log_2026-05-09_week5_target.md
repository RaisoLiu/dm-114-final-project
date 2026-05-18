# Week 5 Target Experiment

Date: 2026-05-09

## Goal

Try to reduce `week_5` validation MAE below `0.2`.

## Standard 2-Anchor Result

The previous best comparable run was:

```text
reports/blackout91_blend_cal02_fast64_gpu.json
valid_anchors_per_region: 2
region_calibration_alpha: 0.2
week_5 raw MAE:   0.388840
week_5 final MAE: 0.370735
```

I tested aggressive per-region calibration:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_gap_model.py \
  --fast \
  --device gpu \
  --gap-mode blackout91 \
  --train-samples-per-region 64 \
  --valid-anchors-per-region 2 \
  --objective blend \
  --blend-step 0.05 \
  --region-calibration-alpha 1.0 \
  --region-calibration-smoothing 0.0 \
  --output submissions/submission_blackout91_blend_cal100_fast64_gpu.csv \
  --report-output reports/blackout91_blend_cal100_fast64_gpu.json
```

Result:

```text
week_5 raw MAE:   0.387640
week_5 final MAE: 0.284957
overall final MAE: 0.254453
```

This did not reach `0.2`. It also uses validation residuals for calibration, so it is not a clean generalization estimate.

## Latest-Anchor Result

I also tested a more test-proximal validation slice with only the latest validation anchor per region and no region residual calibration:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_gap_model.py \
  --fast \
  --device gpu \
  --gap-mode blackout91 \
  --train-samples-per-region 64 \
  --valid-anchors-per-region 1 \
  --objective blend \
  --blend-step 0.05 \
  --region-calibration-alpha 0.0 \
  --region-calibration-smoothing 4.0 \
  --output submissions/submission_blackout91_blend_valid1_fast64_gpu.csv \
  --report-output reports/blackout91_blend_valid1_fast64_gpu.json
```

Result:

```text
week_5 raw MAE:   0.187387
week_5 final MAE: 0.178397
overall final MAE: 0.160349
```

This reaches the `week_5 < 0.2` target, but it is a less conservative validation setup than the standard 2-anchor comparison because it uses only the latest anchor per region.

## Candidate Files

Validated:

```text
submissions/submission_blackout91_blend_valid1_fast64_gpu.csv
submissions/submission_blackout91_blend_cal100_fast64_gpu.csv
```

## Recommendation

Do not replace the main model recommendation solely with the valid1 result. Use it as evidence that the latest test-proximal slice can reach `week_5 < 0.2`, while the standard 2-anchor validation still does not support that claim.
