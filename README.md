# DM 114 Final Project: Natural Disaster Severity Prediction

This repository contains a reproducible Kaggle pipeline for the course final project. The task is to use 91 days of meteorological data for each region to predict drought severity scores for the next five weeks. The Kaggle metric is MAE, so lower is better.

## Data

Download the competition files from Kaggle and place them here:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

The local assignment materials are in `spec/` and `ref/`.

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The default pipeline uses `pandas`, `numpy`, `scikit-learn`, and `lightgbm`. `--models auto` includes LightGBM when it is installed.

Equivalent `make` targets are available after the environment exists:

```bash
make test
make check-data
make eda
make baselines
make cv
make train
make validate-latest
```

## Kaggle Auth

Kaggle CLI accepts either `KAGGLE_USERNAME` plus `KAGGLE_KEY`, or a `kaggle.json` file. This project can create a local config at `.kaggle/kaggle.json`:

```bash
export KAGGLE_USERNAME='your_username'
export KAGGLE_KEY='your_api_key'
PYTHONPATH=src .venv/bin/python scripts/configure_kaggle.py
```

If your token is the full `kaggle.json` content, this also works:

```bash
export KAGGLE_API_TOKEN='{"username":"your_username","key":"your_api_key"}'
PYTHONPATH=src .venv/bin/python scripts/configure_kaggle.py
```

Then download:

```bash
PYTHONPATH=src .venv/bin/python scripts/download_data.py
```

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The tests cover 91-day window construction, weekly target alignment, test-window construction, and submission formatting.

## EDA

First check that the CSVs match the assignment format:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_data.py
```

Then generate EDA notes:

```bash
PYTHONPATH=src .venv/bin/python scripts/eda.py
```

This writes `reports/eda_summary.md`, which is useful for the progress check and report.

## Baselines

```bash
PYTHONPATH=src .venv/bin/python scripts/baselines.py
```

This writes `reports/baselines.json` with global median, region median, seasonal median, region-seasonal median, and last-observed-score baselines using the same time split style as training.

To create the current low-risk Kaggle baseline submission:

```bash
PYTHONPATH=src .venv/bin/python scripts/make_baseline_submission.py --strategy global_median
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py submissions/submission_global_median.csv
```

On the corrected per-region 104-week validation split, this global-median/all-zero submission has MAE about `0.605`, which is below the `0.7` target.

## Rolling Validation

```bash
PYTHONPATH=src .venv/bin/python scripts/cross_validate.py --fast
PYTHONPATH=src .venv/bin/python scripts/cross_validate.py --models lightgbm,hgb,extra --folds 3 --valid-weeks 52
```

This writes `reports/cross_validation.json`. Use it to choose model settings before spending Kaggle submissions.

## Train And Predict

Fast smoke run:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_predict.py --fast
```

Full sklearn run:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_predict.py
```

Optional run if LightGBM is installed:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_predict.py --models lightgbm,hgb,extra
```

Outputs:

```text
submissions/submission_YYYYMMDD_HHMMSS.csv
models/model_YYYYMMDD_HHMMSS.joblib
reports/validation_YYYYMMDD_HHMMSS.json
```

Before uploading:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py submissions/submission_YYYYMMDD_HHMMSS.csv
```

If Kaggle CLI is installed and configured:

```bash
PYTHONPATH=src .venv/bin/python scripts/submit_kaggle.py submissions/submission_YYYYMMDD_HHMMSS.csv --dry-run
PYTHONPATH=src .venv/bin/python scripts/submit_kaggle.py submissions/submission_YYYYMMDD_HHMMSS.csv --message "validation-backed LightGBM ensemble"
```

For a project-local Kaggle token, run `scripts/configure_kaggle.py` first; `scripts/submit_kaggle.py` automatically uses `.kaggle/kaggle.json` when it exists.

## Method

The pipeline follows the assignment constraints:

- Uses only the 91-day weather window, region ID, and date features.
- Does not treat missing `score` rows as zeros.
- Handles the competition's synthetic years with integer calendar features because some years are outside pandas timestamp bounds.
- Splits validation by each region's relative anchor index, not by absolute synthetic year.
- Uses `--anchor-mode auto` by default: if test windows end on the same weekday as observed labels, it trains on score-day anchors; otherwise it trains on all possible days and maps targets to the next five non-null weekly labels.
- Uses a time-based validation split instead of a random split.
- Trains one ensemble per horizon: week 1 through week 5.
- Clips predictions to `[0, 5]`, matching the valid score range.

Feature groups include rolling weather statistics over 7, 14, 28, 56, and 91 days, short-term versus long-term deltas, date seasonality, region encoding, and precipitation dry-spell features when precipitation-like columns exist.

## Report Notes

Use the validation report JSON and EDA summary for the Experiments section. A clean ablation sequence for the report is:

1. Global/region/seasonal baseline.
2. Rolling weather statistics.
3. Precipitation dry-spell features.
4. Date seasonality and region encoding.
5. Horizon-specific models.
6. Ensemble and clipping.

The final report must be in English, 5-8 pages excluding references, and include group ID plus the public GitHub link in the abstract. Kaggle team display name must be `Team {Group ID}`.
