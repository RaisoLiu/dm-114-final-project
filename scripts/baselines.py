#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from drought.features import DATE_COL, REGION_COL, TARGET_COL, FeatureConfig, date_feature_parts, load_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simple time-split baselines for the report.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--valid-weeks", type=int, default=104)
    parser.add_argument("--output", default="reports/baselines.json")
    return parser.parse_args()


def make_supervised_targets(train: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    train = train.reset_index(drop=True)
    for region_id, group in train.groupby(REGION_COL, sort=False):
        group = group.reset_index(drop=True)
        score = pd.to_numeric(group[TARGET_COL], errors="coerce").to_numpy(dtype=float)
        anchors = np.flatnonzero(np.isfinite(score))
        anchors = anchors[anchors >= config.input_days - 1]
        for anchor in anchors:
            target_idx = [anchor + (h + 1) * config.horizon_step_days for h in range(config.horizons)]
            if target_idx[-1] >= len(group):
                continue
            targets = score[target_idx]
            if not np.isfinite(targets).all():
                continue
            row: dict[str, object] = {
                REGION_COL: str(region_id),
                "end_date": group.loc[int(anchor), DATE_COL],
                "anchor_index": int(anchor),
                "anchor_score": float(score[anchor]),
            }
            for h, target in enumerate(targets, start=1):
                target_date = group.loc[int(target_idx[h - 1]), DATE_COL]
                row[f"week_{h}"] = float(target)
                row[f"month_{h}"] = int(date_feature_parts(target_date)["month"])
            rows.append(row)
    if not rows:
        raise ValueError("No supervised target rows were produced.")
    return pd.DataFrame(rows)


def time_split(samples: pd.DataFrame, valid_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    anchors = np.array(sorted(pd.to_numeric(samples["anchor_index"], errors="raise").unique()))
    cutoff_idx = max(1, len(anchors) - valid_weeks) if 0 < valid_weeks < len(anchors) else max(1, int(len(anchors) * 0.80))
    cutoff = int(anchors[cutoff_idx])
    anchor_values = pd.to_numeric(samples["anchor_index"], errors="raise")
    train = samples[anchor_values < cutoff].reset_index(drop=True)
    valid = samples[anchor_values >= cutoff].reset_index(drop=True)
    if train.empty or valid.empty:
        raise ValueError("Time split produced an empty train or validation set.")
    return train, valid, f"anchor_index {cutoff}"


def target_cols(config: FeatureConfig) -> list[str]:
    return [f"week_{h}" for h in range(1, config.horizons + 1)]


def evaluate(name: str, truth: np.ndarray, pred: np.ndarray) -> dict[str, float | str]:
    return {"name": name, "mae": float(mean_absolute_error(truth.ravel(), pred.ravel()))}


def main() -> None:
    args = parse_args()
    config = FeatureConfig()
    train_path = Path(args.data_dir) / "train.csv"
    train = load_frame(train_path)
    samples = make_supervised_targets(train, config)
    train_samples, valid_samples, cutoff = time_split(samples, args.valid_weeks)
    y_cols = target_cols(config)
    y_true = valid_samples[y_cols].to_numpy(dtype=float)
    global_median = float(np.median(train_samples[y_cols].to_numpy(dtype=float).ravel()))

    results: list[dict[str, float | str]] = []
    results.append(evaluate("global_median", y_true, np.full_like(y_true, global_median)))

    last_score_pred = np.repeat(valid_samples["anchor_score"].to_numpy(dtype=float)[:, None], config.horizons, axis=1)
    results.append(evaluate("last_observed_score", y_true, last_score_pred))

    melted = train_samples.melt(id_vars=[REGION_COL], value_vars=y_cols, value_name="target")
    region_median = melted.groupby(REGION_COL)["target"].median().to_dict()
    region_pred = np.full_like(y_true, global_median)
    for i, region_id in enumerate(valid_samples[REGION_COL].astype(str)):
        region_pred[i, :] = region_median.get(region_id, global_median)
    results.append(evaluate("region_median", y_true, region_pred))

    month_rows = []
    for h in range(1, config.horizons + 1):
        month_rows.append(
            train_samples[[REGION_COL, f"month_{h}", f"week_{h}"]].rename(
                columns={f"month_{h}": "month", f"week_{h}": "target"}
            )
        )
    month_train = pd.concat(month_rows, ignore_index=True)
    month_median = month_train.groupby("month")["target"].median().to_dict()
    region_month_median = month_train.groupby([REGION_COL, "month"])["target"].median().to_dict()

    month_pred = np.empty_like(y_true)
    region_month_pred = np.empty_like(y_true)
    for h in range(1, config.horizons + 1):
        months = valid_samples[f"month_{h}"].astype(int).to_numpy()
        regions = valid_samples[REGION_COL].astype(str).to_numpy()
        month_pred[:, h - 1] = [month_median.get(month, global_median) for month in months]
        region_month_pred[:, h - 1] = [
            region_month_median.get((region, month), region_median.get(region, month_median.get(month, global_median)))
            for region, month in zip(regions, months, strict=True)
        ]
    results.append(evaluate("target_month_median", y_true, month_pred))
    results.append(evaluate("region_target_month_median", y_true, region_month_pred))

    report = {
        "validation_cutoff": cutoff,
        "training_samples": int(len(train_samples)),
        "validation_samples": int(len(valid_samples)),
        "results": sorted(results, key=lambda item: float(item["mae"])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
