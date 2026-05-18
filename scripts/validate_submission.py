#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from drought.features import REGION_COL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Kaggle submission against sample_submission.csv.")
    parser.add_argument("submission", help="Submission CSV to validate.")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=5.0)
    parser.add_argument("--allow-out-of-range", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission_path = Path(args.submission)
    sample_path = Path(args.sample_submission)
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission not found: {submission_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample submission not found: {sample_path}")

    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)
    if submission.shape != sample.shape:
        raise ValueError(f"Shape mismatch: submission {submission.shape}, sample {sample.shape}")
    if submission.columns.tolist() != sample.columns.tolist():
        raise ValueError(
            "Column mismatch:\n"
            f"  submission: {submission.columns.tolist()}\n"
            f"  sample:     {sample.columns.tolist()}"
        )

    pred_cols = [col for col in submission.columns if col != REGION_COL]
    if len(pred_cols) != 5:
        raise ValueError(f"Expected exactly 5 prediction columns, found {len(pred_cols)}: {pred_cols}")

    if REGION_COL in sample.columns:
        sample_ids = sample[REGION_COL].astype(str).tolist()
        sub_ids = submission[REGION_COL].astype(str).tolist()
        if sub_ids != sample_ids:
            raise ValueError("region_id order/content does not match sample_submission.csv")
        if submission[REGION_COL].duplicated().any():
            raise ValueError("Duplicate region_id values found in submission.")

    numeric = submission[pred_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad_cols = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Non-numeric or missing predictions found in columns: {bad_cols}")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Infinite prediction values found.")

    below = values < args.clip_min
    above = values > args.clip_max
    if (below.any() or above.any()) and not args.allow_out_of_range:
        raise ValueError(
            f"Predictions outside [{args.clip_min}, {args.clip_max}]: "
            f"{int(below.sum())} below, {int(above.sum())} above"
        )

    print(f"Validated {submission_path}")
    print(f"Rows: {len(submission):,}")
    print(f"Prediction columns: {pred_cols}")
    print(f"Prediction range: {values.min():.6f} to {values.max():.6f}")
    print(f"Prediction mean: {values.mean():.6f}")


if __name__ == "__main__":
    main()

