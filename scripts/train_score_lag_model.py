#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from drought.features import DATE_COL, REGION_COL, TARGET_COL, date_dayofweek, date_dayofyear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train weekly score-lag models for local validation.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="reports/score_lag_validation.json")
    parser.add_argument("--valid-weeks", type=int, default=104)
    parser.add_argument("--lags", type=int, default=52)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def make_model(seed: int, fast: bool, device: str) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="mae",
        n_estimators=500 if fast else 1600,
        learning_rate=0.045 if fast else 0.025,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.82,
        reg_alpha=0.03,
        reg_lambda=0.35,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        device_type=device,
    )


def slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float32)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    y = values.astype(np.float32)
    return float(np.dot(x, y - y.mean()) / denom)


def date_parts(date_value: str) -> list[float]:
    year, month, day = [int(part) for part in date_value.split("-")]
    doy = date_dayofyear(date_value)
    dow = date_dayofweek(date_value)
    return [
        float(month),
        float(day),
        float(doy),
        float(dow),
        float(np.sin(2.0 * np.pi * month / 12.0)),
        float(np.cos(2.0 * np.pi * month / 12.0)),
        float(np.sin(2.0 * np.pi * doy / 366.0)),
        float(np.cos(2.0 * np.pi * doy / 366.0)),
        float(np.sin(2.0 * np.pi * dow / 7.0)),
        float(np.cos(2.0 * np.pi * dow / 7.0)),
        float(year % 4),
    ]


def row_features(
    scores: np.ndarray,
    dates: list[str],
    anchor: int,
    lags: int,
    region_code: int,
    region_train_mean: float,
    region_train_std: float,
) -> list[float]:
    hist = scores[: anchor + 1].astype(np.float32)
    recent = hist[-lags:]
    if recent.size < lags:
        recent = np.pad(recent, (lags - recent.size, 0), constant_values=float(hist[0]))

    feats: list[float] = [
        float(region_code),
        float(anchor),
        region_train_mean,
        region_train_std,
        float(hist[-1]),
    ]
    feats.extend(date_parts(dates[anchor]))
    feats.extend(recent[::-1].astype(float).tolist())

    diffs = np.diff(recent)
    feats.extend(diffs[-16:][::-1].astype(float).tolist())
    if diffs.size < 16:
        feats.extend([0.0] * (16 - diffs.size))

    for window in (2, 3, 4, 8, 13, 26, 52):
        values = hist[-window:]
        feats.extend(
            [
                float(values.mean()),
                float(np.median(values)),
                float(values.std()),
                float(values.min()),
                float(values.max()),
                float(values[-1] - values[0]),
                slope(values),
                float((values > 0).mean()),
                float((values >= 2).mean()),
            ]
        )

    last = hist[-1]
    run = 0
    for value in hist[::-1]:
        if value == last:
            run += 1
        else:
            break
    feats.extend(
        [
            float(run),
            float(hist.size),
            float(hist.mean()),
            float(np.median(hist)),
            float(hist.std()),
            float(hist.max()),
            float((hist > 0).mean()),
            float((hist >= 2).mean()),
        ]
    )
    return feats


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    train_path = Path(args.data_dir) / "train.csv"
    labels = pd.read_csv(train_path, usecols=[REGION_COL, DATE_COL, TARGET_COL])
    labels = labels[pd.to_numeric(labels[TARGET_COL], errors="coerce").notna()].copy()
    labels["score"] = pd.to_numeric(labels[TARGET_COL], errors="raise").astype(np.float32)

    x_train_rows: list[list[float]] = []
    y_train_rows: list[list[float]] = []
    x_valid_rows: list[list[float]] = []
    y_valid_rows: list[list[float]] = []
    persistence_valid_rows: list[list[float]] = []

    print("Building weekly lag rows ...")
    for region_code, (region, group) in enumerate(labels.groupby(REGION_COL, sort=False)):
        group = group.reset_index(drop=True)
        scores = group["score"].to_numpy(dtype=np.float32)
        dates = group[DATE_COL].astype(str).tolist()
        train_end = len(scores) - args.valid_weeks - 5
        if train_end <= args.lags:
            continue
        train_scores = scores[: train_end + 1]
        region_train_mean = float(train_scores.mean())
        region_train_std = float(train_scores.std())
        for anchor in range(args.lags, len(scores) - 5):
            row = row_features(scores, dates, anchor, args.lags, region_code, region_train_mean, region_train_std)
            target = [float(scores[anchor + h]) for h in range(1, 6)]
            if anchor < train_end:
                x_train_rows.append(row)
                y_train_rows.append(target)
            else:
                x_valid_rows.append(row)
                y_valid_rows.append(target)
                persistence_valid_rows.append([float(scores[anchor])] * 5)

    x_train = np.asarray(x_train_rows, dtype=np.float32)
    y_train = np.asarray(y_train_rows, dtype=np.float32)
    x_valid = np.asarray(x_valid_rows, dtype=np.float32)
    y_valid = np.asarray(y_valid_rows, dtype=np.float32)
    persistence_valid = np.asarray(persistence_valid_rows, dtype=np.float32)

    if args.max_train_rows is not None and x_train.shape[0] > args.max_train_rows:
        indices = rng.choice(x_train.shape[0], size=args.max_train_rows, replace=False)
        x_train = x_train[indices]
        y_train = y_train[indices]

    print(f"Train rows: {x_train.shape}; valid rows: {x_valid.shape}")
    pred = np.zeros_like(y_valid)
    horizon_reports: dict[str, float] = {}
    for horizon in range(5):
        model = make_model(args.seed + horizon * 101, args.fast, args.device)
        model.fit(x_train, y_train[:, horizon])
        pred[:, horizon] = np.clip(model.predict(x_valid), 0.0, 5.0)
        horizon_reports[f"week_{horizon + 1}"] = float(mean_absolute_error(y_valid[:, horizon], pred[:, horizon]))
        print(f"  week_{horizon + 1}: {horizon_reports[f'week_{horizon + 1}']:.6f}")

    report = {
        "validation_mae": float(mean_absolute_error(y_valid.ravel(), pred.ravel())),
        "persistence_mae": float(mean_absolute_error(y_valid.ravel(), persistence_valid.ravel())),
        "horizon_mae": horizon_reports,
        "train_rows": int(x_train.shape[0]),
        "valid_rows": int(x_valid.shape[0]),
        "feature_count": int(x_train.shape[1]),
        "valid_weeks": int(args.valid_weeks),
        "lags": int(args.lags),
        "device": args.device,
        "fast": bool(args.fast),
        "args": vars(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
