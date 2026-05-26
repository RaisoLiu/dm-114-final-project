# DM 114 Final Project: Natural Disaster Severity Prediction

This repository contains a reproducible Kaggle pipeline for the course final project. The task is to use 91 days of meteorological data for each region to predict drought severity scores for the next five weeks. The Kaggle metric is MAE, so lower is better.

Public repository URL used in the final report:

```text
https://github.com/RaisoLiu/dm-114-finalproject
```

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

## Final 0.7628 Public-Leaderboard Artifact

The reported final submission is:

```text
submissions/submission_phd_below075_20260522.csv
```

It scored `0.7628` on the Kaggle public leaderboard. The reproducibility path below is a cached-prediction re-blend from retained v18 artefacts and `reports/training_menu_v1.json`; it is not a full retraining of every historical GBDT, lag, CNN, and SSL model from raw data.

The authoritative first step is the data distribution analysis:

```bash
PYTHONPATH=src python scripts/analyze_data_distribution.py --emit-menu
```

This produces `reports/data_characteristics_v1.md`, `data_insights.json`, and the **training menu** `training_menu_v1.json` (the "菜單").

> Note on flags: `--emit-menu` writes the training-menu JSON from cached per-region
> features. The `phd-below075` Make target uses `--force-synthesis --emit-menu`;
> `--force-synthesis` re-emits the menu even if a cached version exists; it does
> **not** synthesize new labels or new submissions.

Then the one-command cached re-blend path to the final submission family is:

```bash
make phd-below075
```

To audit the archived upload exactly, run:

```bash
make verify-submission
```

This regenerates `/tmp/dm114_verify_submission.csv`, validates its schema, and compares it with `submissions/submission_phd_below075_20260522.csv`, reporting maximum absolute and ULP differences (expected max abs diff ≤ `5e-16`).

To reproduce the controlled 9-row OOF ablation table and the cross-leg residual-correlation bootstrap that the iter4 report cites in §Cross-leg residual correlation and Table V:

```bash
make ablation
```

This runs `scripts/regen_lag_2215_oof.py` (deterministic lag-2215 lookup, ~9 s CPU), `scripts/regen_ssl_oof.py` (single forward pass of `checkpoints/track1_finetuned.pt`, ~1 s GPU), then `scripts/build_ablation_9row.py` and `scripts/compute_cross_leg_rho.py`. Outputs land in `reports/ablation_9row.{csv,md}` and `reports/cross_leg_rho_bootstrap.json`.

### Cache checksums (iter4)

A clean clone needs these artefacts in place before `make verify-submission` and `make ablation` will work. Full producer/consumer cross-reference is in `ARTIFACTS.md`; the SHA256 hashes below are inline so a TA can audit cache integrity directly:

```text
2f5ed136f8106dc79222d636b37bd697bdedec5e4df5744a036419645158ed96  reports/oof_tensor.csv
2fe0767025259be39ab6078a59ecb4215f1c413349604a8799ea0890abc0f8ab  reports/_local_eval_gate_report.csv
4885b0c9f003329aa1f561601521d78e0e4bbb42e53c069e0d2e18ba7a8bbd21  reports/training_menu_v1.json
7a08544303e1be5072f5e9d93dc63f20d8d218b6a35fd97829ad9b7763143deb  reports/lag_2215_oof_validation_predictions.csv
352f23dbf6b960f97258a9112e56358790ca3bdeabfeebf60ab2456e04377fe7  reports/track1_ssl_oof_validation_predictions.csv
64aa928b769063fd45ba5b85170434791aa10476d89febedb483065083d5b1c7  reports/ablation_9row.csv
9b0818e97b2238c8c5747ae0efcfb47450a942845dedb687b2b71eb6c556a21f  reports/cross_leg_rho_bootstrap.json
1b7f6ab6484673901eb433c1fa69a4656648c68156253298a712117988a97540  submissions/submission_phd_below075_20260522.csv
```

Several caches under `cache/` (Track 3 CNN and Track 1 SSL prediction caches; ≈ 4.5 GB) and `checkpoints/track1_finetuned.pt` are required for re-running `make ablation`; if missing, restore from the v18 release assets per `ARTIFACTS.md`. `data/train.csv`, `data/test.csv`, and `data/sample_submission.csv` are downloaded directly from the DM 114 Kaggle competition.

### Clean-clone reproduction walkthrough

For a TA verifying that the report, code, and Kaggle result are consistent:

```bash
git clone https://github.com/RaisoLiu/dm-114-finalproject.git
cd dm-114-finalproject

# 1. Python environment
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 2. Data + caches
#    a. Place data/train.csv, data/test.csv, data/sample_submission.csv from
#       the DM 114 (Spring 2026) Kaggle competition under data/
#    b. Restore the cache/ directory and checkpoints/track1_finetuned.pt per
#       ARTIFACTS.md; verify with the SHA256 block above

# 3. Verify the uploaded submission byte-matches the cached re-blend
make verify-submission

# 4. Optional: reproduce the iter4 controlled ablation
make ablation

# 5. Optional: rebuild the report PDF
cd report && make clean && make && make check
```

If `make verify-submission` reports `max abs diff ≤ 5e-16` and `make check` reports
all spec strings present, the cached re-blend pipeline matches the uploaded
submission and the report build is reproducible.

### Layer-2 smoke test (optional)

The walkthrough above (`make verify-submission`, `make ablation`) verifies the
cached re-blend pipeline. To also confirm the *training* pipeline is not just a
black box of cached predictions, two existing lightweight targets exercise the
actual training and validation code paths:

```bash
make test         # pytest: 91-day window, weekly target alignment, submission schema, OOD edge cases
make cv-fast      # 1-fold fast cross-validation on the GBDT pipeline (a few minutes CPU)
```

These do not retrain the v18 models from scratch — that would require the full
GPU pipeline documented in `ARTIFACTS.md` — but they prove the supervised
training, the windowing, and the validation harness all execute against real
`data/train.csv`. Combined with `make verify-submission` (layer 1) and
`make ablation` (layer 3, regenerates lag-2215 OOF + SSL inference + 9-row
ablation), this gives an end-to-end reproducibility ladder.

## Report Notes

Use the validation report JSON and EDA summary for the Experiments section. A clean ablation sequence for the report is:

1. Global/region/seasonal baseline.
2. Rolling weather statistics.
3. Precipitation dry-spell features.
4. Date seasonality and region encoding.
5. Horizon-specific models.
6. Ensemble and clipping.

The final report must be in English, 5-8 pages excluding references, and include group ID plus the public GitHub link in the abstract. Kaggle team display name must be `Team {Group ID}`. Build and verify the report with:

```bash
cd report
make clean
make
make check
```
