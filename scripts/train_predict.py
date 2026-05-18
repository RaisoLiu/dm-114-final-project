#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from drought.features import (
    FeatureConfig,
    REGION_COL,
    build_test_set,
    build_training_set,
    coerce_weather_numeric,
    date_dayofweek,
    infer_weather_columns,
    load_frame,
)
from drought.modeling import (
    FeaturePreprocessor,
    available_model_names,
    fit_horizon_ensemble,
    recency_weights,
    save_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train drought severity models and build a Kaggle submission.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--sample-submission", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-output", default=None)
    parser.add_argument("--report-output", default=None)
    parser.add_argument("--models", default="auto", help="auto or comma-separated: hgb,extra,rf,lightgbm")
    parser.add_argument("--fast", action="store_true", help="Use smaller models for quick iteration.")
    parser.add_argument("--valid-weeks", type=int, default=104, help="Most recent anchor dates used for validation.")
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--anchor-mode", choices=["auto", "score_days", "all_days"], default="auto")
    parser.add_argument("--anchor-stride", type=int, default=7)
    parser.add_argument("--max-samples-per-region", type=int, default=None)
    parser.add_argument("--recency-half-life-days", type=float, default=900.0)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=5.0)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def time_split(meta: pd.DataFrame, valid_weeks: int) -> tuple[np.ndarray, np.ndarray, str]:
    anchors = np.array(sorted(pd.to_numeric(meta["anchor_index"], errors="raise").unique()))
    if anchors.size < 4:
        raise ValueError("Not enough unique anchor dates for time validation.")
    if valid_weeks <= 0 or valid_weeks >= anchors.size:
        cutoff_idx = max(1, int(anchors.size * 0.80))
    else:
        cutoff_idx = max(1, anchors.size - valid_weeks)
    cutoff = int(anchors[cutoff_idx])
    anchor_values = pd.to_numeric(meta["anchor_index"], errors="raise").to_numpy(dtype=np.int64)
    train_mask = anchor_values < cutoff
    valid_mask = ~train_mask
    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError("Time split produced an empty train or validation set.")
    cutoff_label = f"anchor_index {cutoff}"
    return train_mask, valid_mask, cutoff_label


def choose_anchor_mode(train_df: pd.DataFrame, test_df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    score_rows = train_df[pd.to_numeric(train_df["score"], errors="coerce").notna()]
    if score_rows.empty:
        return "all_days"

    score_weekdays = set(score_rows["date"].map(date_dayofweek).astype(int).tolist())
    test_end_weekdays = set(test_df.groupby(REGION_COL, sort=False)["date"].last().map(date_dayofweek).astype(int).tolist())
    return "score_days" if test_end_weekdays <= score_weekdays else "all_days"


def format_submission(
    sample_path: Path,
    test_meta: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path,
    clip_min: float,
    clip_max: float,
) -> None:
    sample = pd.read_csv(sample_path)
    pred_cols = [col for col in sample.columns if col != REGION_COL]
    if len(pred_cols) != predictions.shape[1]:
        if predictions.shape[1] == 5 and len(sample.columns) >= 6:
            pred_cols = sample.columns[-5:].tolist()
        else:
            raise ValueError(
                f"Cannot identify 5 prediction columns in sample submission. Columns: {sample.columns.tolist()}"
            )

    pred_df = pd.DataFrame(np.clip(predictions, clip_min, clip_max), columns=pred_cols)
    pred_df[REGION_COL] = test_meta[REGION_COL].astype(str).to_numpy()

    if REGION_COL in sample.columns:
        sample[REGION_COL] = sample[REGION_COL].astype(str)
        merged = sample[[REGION_COL]].merge(pred_df, on=REGION_COL, how="left")
        if merged[pred_cols].isna().any().any():
            missing = merged.loc[merged[pred_cols].isna().any(axis=1), REGION_COL].tolist()
            raise ValueError(f"Missing predictions for sample regions: {missing[:10]}")
        for col in pred_cols:
            sample[col] = merged[col].to_numpy()
        submission = sample
    else:
        if len(sample) != len(pred_df):
            raise ValueError("sample_submission row count does not match test region count and no region_id column exists.")
        submission = sample.copy()
        for col in pred_cols:
            submission[col] = pred_df[col].to_numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_path = require_file(Path(args.train) if args.train else data_dir / "train.csv", "train.csv")
    test_path = require_file(Path(args.test) if args.test else data_dir / "test.csv", "test.csv")
    sample_path = require_file(
        Path(args.sample_submission) if args.sample_submission else data_dir / "sample_submission.csv",
        "sample_submission.csv",
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else Path("submissions") / f"submission_{stamp}.csv"
    model_path = Path(args.model_output) if args.model_output else Path("models") / f"model_{stamp}.joblib"
    report_path = Path(args.report_output) if args.report_output else Path("reports") / f"validation_{stamp}.json"

    print(f"Loading data from {data_dir} ...")
    train_df = load_frame(train_path)
    test_df = load_frame(test_path)
    anchor_mode = choose_anchor_mode(train_df, test_df, args.anchor_mode)
    if args.anchor_mode == "auto":
        print(f"Auto-selected anchor mode: {anchor_mode}")
    config = FeatureConfig(anchor_mode=anchor_mode, anchor_stride=args.anchor_stride)
    numeric_cols = infer_weather_columns(train_df, test_df)
    train_df = coerce_weather_numeric(train_df, numeric_cols)
    test_df = coerce_weather_numeric(test_df, numeric_cols)
    print(f"Using {len(numeric_cols)} meteorological columns.")

    print("Building 91-day training windows ...")
    features, targets, meta = build_training_set(
        train_df,
        numeric_cols,
        config,
        max_samples_per_region=args.max_samples_per_region,
    )
    test_features, test_meta = build_test_set(test_df, numeric_cols, config)
    print(f"Training samples: {len(features):,}; raw feature columns: {len(features.columns):,}.")

    train_mask, valid_mask, cutoff = time_split(meta, args.valid_weeks)
    x_train_raw = features.loc[train_mask].reset_index(drop=True)
    y_train = targets.loc[train_mask].reset_index(drop=True)
    meta_train = meta.loc[train_mask].reset_index(drop=True)
    x_valid_raw = features.loc[valid_mask].reset_index(drop=True)
    y_valid = targets.loc[valid_mask].reset_index(drop=True)
    meta_valid = meta.loc[valid_mask].reset_index(drop=True)

    preprocessor = FeaturePreprocessor().fit(x_train_raw, meta_train)
    x_train = preprocessor.transform(x_train_raw, meta_train)
    x_valid = preprocessor.transform(x_valid_raw, meta_valid)
    model_names = available_model_names(args.models)
    weights = recency_weights(meta_train, args.recency_half_life_days)
    print(f"Validation cutoff: {cutoff} ({len(x_train):,} train / {len(x_valid):,} valid).")
    print(f"Training validation ensemble: {', '.join(model_names)}")

    validation_ensemble, validation_metrics = fit_horizon_ensemble(
        x_train,
        y_train,
        model_names=model_names,
        seed=args.seed,
        fast=args.fast,
        sample_weight=weights,
        validation=(x_valid, y_valid),
    )
    valid_pred = np.clip(validation_ensemble.predict(x_valid), args.clip_min, args.clip_max)
    validation_metrics["mae"] = float(mean_absolute_error(y_valid.to_numpy().ravel(), valid_pred.ravel()))
    validation_metrics["horizon_mae"] = {
        col: float(mean_absolute_error(y_valid[col], valid_pred[:, i])) for i, col in enumerate(y_valid.columns)
    }
    print(f"Validation MAE: {validation_metrics['mae']:.6f}")
    for col, value in validation_metrics["horizon_mae"].items():
        print(f"  {col}: {value:.6f}")

    print("Refitting on all training windows for final submission ...")
    final_preprocessor = FeaturePreprocessor().fit(features, meta)
    x_all = final_preprocessor.transform(features, meta)
    x_test = final_preprocessor.transform(test_features, test_meta)
    final_weights = recency_weights(meta, args.recency_half_life_days)
    final_ensemble, _ = fit_horizon_ensemble(
        x_all,
        targets,
        model_names=model_names,
        seed=args.seed,
        fast=args.fast,
        sample_weight=final_weights,
        fixed_weights=validation_ensemble.weights_by_horizon,
    )
    test_pred = np.clip(final_ensemble.predict(x_test), args.clip_min, args.clip_max)

    format_submission(sample_path, test_meta, test_pred, output_path, args.clip_min, args.clip_max)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_bundle(
        str(model_path),
        {
            "config": config,
            "numeric_cols": numeric_cols,
            "preprocessor": final_preprocessor,
            "ensemble": final_ensemble,
            "validation_metrics": validation_metrics,
            "args": vars(args),
        },
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "validation_cutoff": cutoff,
        "selected_anchor_mode": anchor_mode,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "training_samples": int(len(features)),
        "validation_samples": int(len(x_valid)),
        "feature_count": int(x_all.shape[1]),
        "numeric_columns": numeric_cols,
        "model_names": model_names,
        "validation_metrics": validation_metrics,
        "submission": str(output_path),
        "model": str(model_path),
        "args": vars(args),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved submission: {output_path}")
    print(f"Saved model: {model_path}")
    print(f"Saved validation report: {report_path}")


if __name__ == "__main__":
    main()
