# Final Report Outline

## Abstract

Include the group ID, public GitHub repository link, the task, the MAE metric, and the final Kaggle score.

## Project Summary

Describe the goal: predict five future weekly drought severity scores from 91 days of meteorological history for each region. Summarize the leakage-safe validation design and the high-level method.

## Proposed Method

Explain the supervised window construction, where each sample uses the previous 91 days of weather and predicts week 1 through week 5. Describe rolling weather statistics, precipitation dry-spell features, date seasonality, region encoding, horizon-specific models, ensembling, and clipping predictions to `[0, 5]`.

## Experiments

Report dataset statistics from `reports/eda_summary.md`. Compare baselines from `reports/baselines.json` against the model validation MAE from `reports/validation_*.json`. Include per-horizon MAE, model ablation, and Kaggle public leaderboard scores for submitted files.

Suggested experiment table:

| Method | Validation MAE | Public MAE | Notes |
| --- | ---: | ---: | --- |
| Global median | TBD | N/A | From `scripts/baselines.py` |
| Region median | TBD | N/A | From `scripts/baselines.py` |
| Region-month median | TBD | N/A | From `scripts/baselines.py` |
| HGB only | TBD | TBD | `--models hgb` |
| ExtraTrees only | TBD | TBD | `--models extra` |
| HGB + ExtraTrees | TBD | TBD | default sklearn ensemble |
| LightGBM + HGB + ExtraTrees | TBD | TBD | only if LightGBM is installed |

## Reproducibility

State the exact command used to produce the submitted CSV and confirm that the code, report, and Kaggle result refer to the same submission.

## References

Use only real references. Do not include AI-hallucinated citations.
