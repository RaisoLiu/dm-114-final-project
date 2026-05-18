#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from drought.features import DATE_COL, REGION_COL, TARGET_COL, date_day_name, infer_weather_columns, load_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check competition CSV files against the project spec.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="reports/data_check.json")
    parser.add_argument("--strict", action="store_true", help="Fail on spec warnings, not only hard errors.")
    return parser.parse_args()


def add_issue(issues: list[dict[str, str]], severity: str, message: str) -> None:
    issues.append({"severity": severity, "message": message})


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    issues: list[dict[str, str]] = []

    for label, path in [("train.csv", train_path), ("test.csv", test_path), ("sample_submission.csv", sample_path)]:
        if not path.exists():
            add_issue(issues, "error", f"{label} is missing at {path}")

    if any(issue["severity"] == "error" for issue in issues):
        report = {"ok": False, "issues": issues}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise SystemExit(1)

    train = load_frame(train_path)
    test = load_frame(test_path)
    sample = pd.read_csv(sample_path)

    if TARGET_COL not in train.columns:
        add_issue(issues, "error", "train.csv must contain a score column.")
    if TARGET_COL in test.columns:
        add_issue(issues, "error", "test.csv should not contain a score column.")

    try:
        numeric_cols = infer_weather_columns(train, test)
    except ValueError as exc:
        numeric_cols = []
        add_issue(issues, "error", str(exc))

    train_regions = set(train[REGION_COL].astype(str))
    test_regions = set(test[REGION_COL].astype(str))
    missing_from_train = sorted(test_regions - train_regions)
    if missing_from_train:
        add_issue(issues, "warning", f"{len(missing_from_train)} test regions are not present in train.")

    train_counts = train.groupby(REGION_COL).size()
    if not train_counts.empty and train_counts.nunique() != 1:
        add_issue(issues, "warning", "train rows per region are not uniform.")
    if not train_counts.empty and int(train_counts.median()) != 5480:
        add_issue(
            issues,
            "warning",
            f"median train rows per region is {int(train_counts.median())}; reference spec says 5480.",
        )

    test_counts = test.groupby(REGION_COL).size()
    bad_test_regions = test_counts[test_counts != 91]
    if not bad_test_regions.empty:
        add_issue(issues, "error", f"{len(bad_test_regions)} test regions do not have exactly 91 rows.")

    if REGION_COL in sample.columns:
        sample_regions = sample[REGION_COL].astype(str).tolist()
        if len(sample_regions) != len(set(sample_regions)):
            add_issue(issues, "error", "sample_submission.csv contains duplicate region_id values.")
        if set(sample_regions) != test_regions:
            add_issue(issues, "error", "sample_submission.csv region_id set does not match test.csv.")
    elif len(sample) != test[REGION_COL].nunique():
        add_issue(
            issues,
            "error",
            "sample_submission.csv has no region_id column and row count does not match test region count.",
        )

    pred_cols = [col for col in sample.columns if col != REGION_COL]
    if len(pred_cols) != 5:
        add_issue(issues, "error", f"sample_submission.csv should have 5 prediction columns, found {len(pred_cols)}.")

    score_summary: dict[str, object] = {}
    if TARGET_COL in train.columns:
        score = pd.to_numeric(train[TARGET_COL], errors="coerce")
        non_null = train.loc[score.notna(), [REGION_COL, DATE_COL]].copy()
        score_summary = {
            "non_null_rows": int(score.notna().sum()),
            "missing_rate": float(score.isna().mean()),
            "min": float(score.min(skipna=True)),
            "max": float(score.max(skipna=True)),
        }
        if score.notna().sum() == 0:
            add_issue(issues, "error", "score column has no non-null labels.")
        if score.min(skipna=True) < 0 or score.max(skipna=True) > 5:
            add_issue(issues, "warning", "score values are outside the expected [0, 5] range.")
        if not non_null.empty:
            weekday_count = non_null[DATE_COL].map(date_day_name).value_counts().to_dict()
            score_summary["weekday_count"] = {str(k): int(v) for k, v in weekday_count.items()}

    hard_failed = any(issue["severity"] == "error" for issue in issues)
    strict_failed = args.strict and any(issue["severity"] in {"error", "warning"} for issue in issues)
    report = {
        "ok": not hard_failed and not strict_failed,
        "strict": bool(args.strict),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "sample_submission_shape": list(sample.shape),
        "train_regions": int(train[REGION_COL].nunique()),
        "test_regions": int(test[REGION_COL].nunique()),
        "numeric_weather_columns": numeric_cols,
        "score_summary": score_summary,
        "issues": issues,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")
    if hard_failed or strict_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
