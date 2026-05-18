#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drought.features import (
    FeatureConfig,
    build_training_set,
    coerce_weather_numeric,
    infer_weather_columns,
    load_frame,
)
from drought.modeling import (
    FeaturePreprocessor,
    available_model_names,
    fit_horizon_ensemble,
    recency_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling time-based CV for model selection.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="reports/cross_validation.json")
    parser.add_argument("--models", default="auto", help="auto or comma-separated: hgb,extra,rf,lightgbm")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--valid-weeks", type=int, default=52)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--anchor-mode", choices=["score_days", "all_days"], default="score_days")
    parser.add_argument("--anchor-stride", type=int, default=7)
    parser.add_argument("--max-samples-per-region", type=int, default=None)
    parser.add_argument("--recency-half-life-days", type=float, default=900.0)
    return parser.parse_args()


def make_rolling_folds(meta: pd.DataFrame, requested_folds: int, valid_weeks: int) -> list[dict[str, object]]:
    anchor_values = pd.to_numeric(meta["anchor_index"], errors="raise").to_numpy(dtype=np.int64)
    anchors = np.array(sorted(np.unique(anchor_values)))
    if anchors.size < 4:
        raise ValueError("Not enough unique anchor dates for rolling CV.")
    block = max(1, valid_weeks)
    folds = min(max(1, requested_folds), max(1, (anchors.size - 1) // block))
    result: list[dict[str, object]] = []
    for fold_idx in range(folds):
        valid_end_idx = anchors.size - (folds - 1 - fold_idx) * block
        valid_start_idx = max(1, valid_end_idx - block)
        train_cutoff = int(anchors[valid_start_idx])
        valid_end = int(anchors[valid_end_idx - 1])
        train_mask = anchor_values < train_cutoff
        valid_mask = (anchor_values >= train_cutoff) & (anchor_values <= valid_end)
        if train_mask.sum() == 0 or valid_mask.sum() == 0:
            continue
        result.append(
            {
                "fold": fold_idx + 1,
                "train_cutoff": f"anchor_index {train_cutoff}",
                "valid_end": f"anchor_index {valid_end}",
                "train_mask": train_mask,
                "valid_mask": valid_mask,
            }
        )
    if not result:
        raise ValueError("No usable rolling folds were produced.")
    return result


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found: {train_path}")

    config = FeatureConfig(anchor_mode=args.anchor_mode, anchor_stride=args.anchor_stride)
    print(f"Loading {train_path} ...")
    train_df = load_frame(train_path)
    numeric_cols = infer_weather_columns(train_df)
    train_df = coerce_weather_numeric(train_df, numeric_cols)
    features, targets, meta = build_training_set(
        train_df,
        numeric_cols,
        config,
        max_samples_per_region=args.max_samples_per_region,
    )
    model_names = available_model_names(args.models)
    folds = make_rolling_folds(meta, args.folds, args.valid_weeks)
    print(f"Samples: {len(features):,}; raw features: {len(features.columns):,}; folds: {len(folds)}")
    print(f"Models: {', '.join(model_names)}")

    fold_reports: list[dict[str, object]] = []
    for fold in folds:
        fold_id = int(fold["fold"])
        print(
            f"Fold {fold_id}: valid {fold['train_cutoff']} "
            f"to {fold['valid_end']}"
        )
        train_mask = fold["train_mask"]
        valid_mask = fold["valid_mask"]
        x_train_raw = features.loc[train_mask].reset_index(drop=True)
        y_train = targets.loc[train_mask].reset_index(drop=True)
        meta_train = meta.loc[train_mask].reset_index(drop=True)
        x_valid_raw = features.loc[valid_mask].reset_index(drop=True)
        y_valid = targets.loc[valid_mask].reset_index(drop=True)
        meta_valid = meta.loc[valid_mask].reset_index(drop=True)

        preprocessor = FeaturePreprocessor().fit(x_train_raw, meta_train)
        x_train = preprocessor.transform(x_train_raw, meta_train)
        x_valid = preprocessor.transform(x_valid_raw, meta_valid)
        weights = recency_weights(meta_train, args.recency_half_life_days)
        _, metrics = fit_horizon_ensemble(
            x_train,
            y_train,
            model_names=model_names,
            seed=args.seed + fold_id * 1000,
            fast=args.fast,
            sample_weight=weights,
            validation=(x_valid, y_valid),
        )
        print(f"  MAE: {metrics['mae']:.6f}")
        fold_reports.append(
            {
                "fold": fold_id,
                "train_cutoff": str(fold["train_cutoff"]),
                "valid_end": str(fold["valid_end"]),
                "train_samples": int(train_mask.sum()),
                "valid_samples": int(valid_mask.sum()),
                "metrics": metrics,
            }
        )

    maes = [float(item["metrics"]["mae"]) for item in fold_reports]
    report = {
        "model_names": model_names,
        "numeric_columns": numeric_cols,
        "feature_columns": int(len(features.columns)),
        "samples": int(len(features)),
        "fold_count": len(fold_reports),
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes)),
        "folds": fold_reports,
        "args": vars(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Mean MAE: {report['mean_mae']:.6f} +/- {report['std_mae']:.6f}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
