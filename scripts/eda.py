#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from drought.features import DATE_COL, REGION_COL, TARGET_COL, infer_weather_columns, load_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create EDA notes for the drought Kaggle project.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="reports/eda_summary.md")
    return parser.parse_args()


def describe_counts(series: pd.Series) -> str:
    desc = series.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    return "\n".join(f"- {idx}: {value:.3f}" for idx, value in desc.items())


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Put train.csv and test.csv under data/ before running EDA.")

    train = load_frame(train_path)
    test = load_frame(test_path)
    sample = pd.read_csv(sample_path) if sample_path.exists() else None
    numeric_cols = infer_weather_columns(train, test)

    lines: list[str] = []
    lines.append("# EDA Summary")
    lines.append("")
    lines.append("## Shapes")
    lines.append(f"- train.csv: {train.shape[0]:,} rows x {train.shape[1]:,} columns")
    lines.append(f"- test.csv: {test.shape[0]:,} rows x {test.shape[1]:,} columns")
    if sample is not None:
        lines.append(f"- sample_submission.csv: {sample.shape[0]:,} rows x {sample.shape[1]:,} columns")
    lines.append(f"- meteorological columns used by pipeline: {len(numeric_cols)}")
    lines.append("")

    lines.append("## Regions And Dates")
    train_counts = train.groupby(REGION_COL).size()
    test_counts = test.groupby(REGION_COL).size()
    lines.append(f"- train regions: {train[REGION_COL].nunique():,}")
    lines.append(f"- test regions: {test[REGION_COL].nunique():,}")
    lines.append(f"- train date range: {train[DATE_COL].min().date()} to {train[DATE_COL].max().date()}")
    lines.append(f"- test date range: {test[DATE_COL].min().date()} to {test[DATE_COL].max().date()}")
    lines.append("- train rows per region:")
    lines.append(describe_counts(train_counts))
    lines.append("- test rows per region:")
    lines.append(describe_counts(test_counts))
    missing_test_regions = sorted(set(test[REGION_COL]) - set(train[REGION_COL]))
    lines.append(f"- test regions not present in train: {len(missing_test_regions)}")
    if missing_test_regions:
        lines.append(f"- first missing regions: {missing_test_regions[:10]}")
    lines.append("")

    if TARGET_COL in train.columns:
        score = pd.to_numeric(train[TARGET_COL], errors="coerce")
        non_null = train.loc[score.notna(), [REGION_COL, DATE_COL, TARGET_COL]].copy()
        non_null[TARGET_COL] = pd.to_numeric(non_null[TARGET_COL], errors="coerce")
        lines.append("## Score Labels")
        lines.append(f"- non-null score rows: {len(non_null):,}")
        lines.append(f"- score missing rate: {score.isna().mean():.4f}")
        lines.append("- score distribution:")
        for value, count in non_null[TARGET_COL].value_counts().sort_index().items():
            lines.append(f"  - {value}: {count:,}")
        weekday_counts = non_null[DATE_COL].dt.day_name().value_counts()
        lines.append("- score weekday counts:")
        for value, count in weekday_counts.items():
            lines.append(f"  - {value}: {count:,}")
        score_counts = non_null.groupby(REGION_COL).size()
        lines.append("- non-null score rows per region:")
        lines.append(describe_counts(score_counts))
        lines.append("")

    lines.append("## Missing Weather Values")
    missing = train[numeric_cols].isna().mean().sort_values(ascending=False)
    for col, rate in missing.head(20).items():
        lines.append(f"- {col}: {rate:.4f}")
    lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

