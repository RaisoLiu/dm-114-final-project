#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drought.features import REGION_COL, TARGET_COL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create simple validated baseline Kaggle submissions.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="submissions/submission_global_median.csv")
    parser.add_argument("--report-output", default="reports/baseline_submission.json")
    parser.add_argument(
        "--strategy",
        choices=["global_median", "last_train_score", "blend"],
        default="global_median",
    )
    parser.add_argument(
        "--last-score-weight",
        type=float,
        default=0.50,
        help="Weight on each region's last train score for --strategy blend.",
    )
    return parser.parse_args()


def prediction_columns(sample: pd.DataFrame) -> list[str]:
    pred_cols = [col for col in sample.columns if col != REGION_COL]
    if len(pred_cols) != 5:
        raise ValueError(f"Expected 5 prediction columns, found {len(pred_cols)}: {pred_cols}")
    return pred_cols


def last_train_scores(train: pd.DataFrame) -> pd.Series:
    labels = train[[REGION_COL, TARGET_COL]].copy()
    labels[TARGET_COL] = pd.to_numeric(labels[TARGET_COL], errors="coerce")
    labels = labels.dropna(subset=[TARGET_COL])
    if labels.empty:
        raise ValueError("train.csv has no non-null score labels.")
    return labels.groupby(REGION_COL, sort=False)[TARGET_COL].last()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.csv"
    sample_path = data_dir / "sample_submission.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found: {train_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"sample_submission.csv not found: {sample_path}")

    train = pd.read_csv(train_path, usecols=[REGION_COL, TARGET_COL])
    sample = pd.read_csv(sample_path)
    pred_cols = prediction_columns(sample)

    score = pd.to_numeric(train[TARGET_COL], errors="coerce")
    global_median = float(score.median(skipna=True))
    if not np.isfinite(global_median):
        raise ValueError("Could not compute a finite global median score.")

    if args.strategy == "global_median":
        predictions = np.full((len(sample), len(pred_cols)), global_median, dtype=float)
    else:
        region_last = last_train_scores(train)
        last_values = sample[REGION_COL].astype(str).map(region_last).fillna(global_median).to_numpy(dtype=float)
        if args.strategy == "last_train_score":
            predictions = np.repeat(last_values[:, None], len(pred_cols), axis=1)
        else:
            weight = float(np.clip(args.last_score_weight, 0.0, 1.0))
            blended = weight * last_values + (1.0 - weight) * global_median
            predictions = np.repeat(blended[:, None], len(pred_cols), axis=1)

    predictions = np.clip(predictions, 0.0, 5.0)
    submission = sample.copy()
    submission[pred_cols] = predictions

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    report = {
        "strategy": args.strategy,
        "global_median": global_median,
        "last_score_weight": float(args.last_score_weight),
        "rows": int(len(submission)),
        "prediction_columns": pred_cols,
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "prediction_mean": float(predictions.mean()),
        "submission": str(output_path),
    }
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
