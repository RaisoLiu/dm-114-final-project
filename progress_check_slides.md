# Progress Check Slides

Use this as the 5-minute May 21 presentation structure.

## Slide 1: Problem And Metric

- Task: predict drought severity for the next five weeks from 91 days of regional meteorological data.
- Input: `test.csv` provides 91 daily rows per `region_id`.
- Output: `pred_week1` through `pred_week5`.
- Metric: MAE, lower is better.
- Competition target for this project work: public score below `0.7`.

## Slide 2: Data Observations

Fill this from:

```bash
PYTHONPATH=src python3 scripts/eda.py
```

Report:

- number of regions in train/test
- train/test date ranges
- rows per region
- score missing rate
- score weekday/cadence
- score distribution
- weather missing values

## Slide 3: Validation Design

- Random split is avoided because nearby dates from the same region leak temporal information.
- Each training sample uses only the previous 91 days of weather.
- Targets are the next five weekly score values.
- Validation uses the most recent anchor dates as a future holdout.
- Rolling CV is available for model selection:

```bash
PYTHONPATH=src python3 scripts/cross_validate.py --models hgb,extra --folds 3 --valid-weeks 52
```

## Slide 4: Baselines

Fill this from:

```bash
PYTHONPATH=src python3 scripts/baselines.py
```

Compare:

- global median
- last observed score
- region median
- target-month median
- region and target-month median

Explain why each baseline is useful and show MAE on the same time split.

## Slide 5: Current Model And Next Steps

Current model:

- rolling weather statistics over 7, 14, 28, 56, and 91 days
- precipitation dry-spell features when precipitation columns exist
- date seasonality
- region encoding
- one model ensemble per forecast horizon
- prediction clipping to `[0, 5]`

Run:

```bash
PYTHONPATH=src python3 scripts/train_predict.py
PYTHONPATH=src python3 scripts/validate_submission.py submissions/submission_YYYYMMDD_HHMMSS.csv
```

Next steps:

- submit the best validation-backed CSV to Kaggle
- tune model list and validation weeks only when rolling CV improves
- record every submission filename, validation MAE, and public MAE
- include ablation and per-horizon MAE in the final report

