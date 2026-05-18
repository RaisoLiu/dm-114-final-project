#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from drought.features import DATE_COL, REGION_COL, TARGET_COL, date_ordinal
from train_gap_model import (
    WEATHER_COLS,
    build_region_stats,
    build_weather_stats,
    make_model as make_gap_model,
    make_row,
    output_distribution,
)
from train_predict import format_submission
from train_score_lag_model import make_model as make_lag_model
from train_score_lag_model import row_features


warnings.filterwarnings("ignore", message="X does not have valid feature names")


@dataclass
class NowcastCase:
    region: str
    region_code: int
    anchor: int
    cutoff: int
    anchor_date: str
    current_target: float
    future_target: np.ndarray
    cutoff_score_pos: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent current-score nowcast plus score-lag forecast candidate.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="submissions/submission_latent_nowcast.csv")
    parser.add_argument("--report-output", default="reports/latent_nowcast.json")
    parser.add_argument("--train-samples-per-region", type=int, default=96)
    parser.add_argument("--final-train-samples-per-region", type=int, default=160)
    parser.add_argument("--valid-anchors-per-region", type=int, default=2)
    parser.add_argument("--valid-weeks", type=int, default=104)
    parser.add_argument("--lags", type=int, default=52)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def score_targets(score_idx: np.ndarray, score_values: np.ndarray, anchor: int, horizons: int = 5) -> np.ndarray | None:
    start = int(np.searchsorted(score_idx, anchor, side="right"))
    end = start + horizons
    if end > score_idx.size:
        return None
    return score_values[start:end].astype(np.float32)


def current_score_target(score_idx: np.ndarray, score_values: np.ndarray, anchor: int) -> tuple[int, float] | None:
    pos = int(np.searchsorted(score_idx, anchor, side="right")) - 1
    if pos < 0:
        return None
    return pos, float(score_values[pos])


def blackout_cutoff(anchor: int) -> int:
    return max(0, anchor - 91)


def build_nowcast_rows(
    train: pd.DataFrame,
    region_to_code: dict[str, int],
    weather_stats_by_region: dict[str, dict[str, np.ndarray]],
    samples_per_region: int,
    valid_anchors_per_region: int,
    seed: int,
    include_recent_for_train: bool,
) -> tuple[np.ndarray, np.ndarray, list[NowcastCase], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train_rows: list[list[float]] = []
    y_train_rows: list[float] = []
    valid_cases: list[NowcastCase] = []
    x_valid_rows: list[list[float]] = []
    y_valid_rows: list[float] = []

    for region_id, group in train.groupby(REGION_COL, sort=False):
        region = str(region_id)
        group = group.reset_index(drop=True)
        weather_values = group[WEATHER_COLS].to_numpy(dtype=np.float32)
        weather_stats = weather_stats_by_region[region]
        score = pd.to_numeric(group[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)
        score_idx = np.flatnonzero(np.isfinite(score)).astype(np.int32)
        score_values = score[score_idx].astype(np.float32)
        region_code = region_to_code.get(region, len(region_to_code))
        last_usable_anchor = len(group) - 36
        if last_usable_anchor < 91 or score_values.size < 8:
            continue

        valid_deltas = np.linspace(0, 365, max(1, valid_anchors_per_region), dtype=int)
        valid_anchors = [int(last_usable_anchor - delta) for delta in valid_deltas if last_usable_anchor - delta >= 91]
        valid_anchor_set = set(valid_anchors)

        if not include_recent_for_train:
            candidate_anchors = np.arange(365, max(366, last_usable_anchor - 420), 7, dtype=np.int32)
            if candidate_anchors.size == 0:
                candidate_anchors = np.arange(91, last_usable_anchor + 1, 7, dtype=np.int32)
        else:
            candidate_anchors = np.arange(365, last_usable_anchor + 1, 7, dtype=np.int32)

        sample_size = min(samples_per_region, candidate_anchors.size)
        sampled_anchors = rng.choice(candidate_anchors, size=sample_size, replace=False)
        for anchor_value in sampled_anchors:
            anchor = int(anchor_value)
            if anchor in valid_anchor_set:
                continue
            target = current_score_target(score_idx, score_values, anchor)
            future = score_targets(score_idx, score_values, anchor)
            if target is None or future is None:
                continue
            cutoff = blackout_cutoff(anchor)
            region_stats = build_region_stats(score_values[: int(np.searchsorted(score_idx, cutoff, side="right"))])
            x_train_rows.append(
                make_row(
                    weather_values[anchor - 90 : anchor + 1],
                    str(group.loc[anchor, DATE_COL]),
                    anchor,
                    cutoff,
                    score_idx,
                    score_values,
                    region_code,
                    region_stats,
                    weather_stats,
                )
            )
            y_train_rows.append(target[1])

        for anchor in valid_anchors:
            target = current_score_target(score_idx, score_values, anchor)
            future = score_targets(score_idx, score_values, anchor)
            if target is None or future is None:
                continue
            cutoff = blackout_cutoff(anchor)
            cutoff_pos = int(np.searchsorted(score_idx, cutoff, side="right")) - 1
            if cutoff_pos < 0:
                continue
            region_stats = build_region_stats(score_values[: cutoff_pos + 1])
            row = make_row(
                weather_values[anchor - 90 : anchor + 1],
                str(group.loc[anchor, DATE_COL]),
                anchor,
                cutoff,
                score_idx,
                score_values,
                region_code,
                region_stats,
                weather_stats,
            )
            x_valid_rows.append(row)
            y_valid_rows.append(target[1])
            valid_cases.append(
                NowcastCase(
                    region=region,
                    region_code=region_code,
                    anchor=anchor,
                    cutoff=cutoff,
                    anchor_date=str(group.loc[anchor, DATE_COL]),
                    current_target=target[1],
                    future_target=future,
                    cutoff_score_pos=cutoff_pos,
                )
            )

    return (
        np.asarray(x_train_rows, dtype=np.float32),
        np.asarray(y_train_rows, dtype=np.float32),
        valid_cases,
        np.asarray(x_valid_rows, dtype=np.float32),
        np.asarray(y_valid_rows, dtype=np.float32),
    )


def build_forecast_rows(
    labels: pd.DataFrame,
    lags: int,
    valid_weeks: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    for region_code, (_region, group) in enumerate(labels.groupby(REGION_COL, sort=False)):
        group = group.reset_index(drop=True)
        scores = group["score"].to_numpy(dtype=np.float32)
        dates = group[DATE_COL].astype(str).tolist()
        if scores.size <= lags + 5:
            continue
        if valid_weeks is None:
            max_anchor = len(scores) - 5
        else:
            max_anchor = max(lags, len(scores) - valid_weeks - 5)
        region_train_scores = scores[:max_anchor]
        region_train_mean = float(region_train_scores.mean())
        region_train_std = float(region_train_scores.std())
        for anchor in range(lags, max_anchor):
            x_rows.append(row_features(scores, dates, anchor, lags, region_code, region_train_mean, region_train_std))
            y_rows.append([float(scores[anchor + h]) for h in range(1, 6)])
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def fit_forecast_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    fast: bool,
    device: str,
) -> list[Any]:
    models: list[Any] = []
    for horizon in range(5):
        model = make_lag_model(seed + horizon * 101, fast, device)
        model.fit(x_train, y_train[:, horizon])
        models.append(model)
    return models


def forecast_from_history(
    models: list[Any],
    score_values: np.ndarray,
    score_dates: list[str],
    cutoff_score_pos: int,
    predicted_current: float,
    anchor_date: str,
    lags: int,
    region_code: int,
) -> np.ndarray:
    history_scores = np.concatenate(
        [score_values[: cutoff_score_pos + 1].astype(np.float32), np.asarray([predicted_current], dtype=np.float32)]
    )
    history_dates = score_dates[: cutoff_score_pos + 1] + [anchor_date]
    region_mean = float(history_scores.mean())
    region_std = float(history_scores.std())
    row = np.asarray(
        row_features(
            history_scores,
            history_dates,
            len(history_scores) - 1,
            lags,
            region_code,
            region_mean,
            region_std,
        ),
        dtype=np.float32,
    ).reshape(1, -1)
    return np.asarray([np.clip(model.predict(row)[0], 0.0, 5.0) for model in models], dtype=np.float32)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"

    print("Loading data ...")
    train = pd.read_csv(train_path, usecols=[REGION_COL, DATE_COL, TARGET_COL, *WEATHER_COLS])
    test = pd.read_csv(test_path, usecols=[REGION_COL, DATE_COL, *WEATHER_COLS])
    sample = pd.read_csv(sample_path)
    region_order = sample[REGION_COL].astype(str).tolist()
    region_to_code = {region_id: idx for idx, region_id in enumerate(region_order)}

    weather_stats_by_region = {
        str(region): build_weather_stats(group) for region, group in train.groupby(REGION_COL, sort=False)
    }

    print("Building validation nowcast rows ...")
    x_now_train, y_now_train, valid_cases, x_now_valid, y_now_valid = build_nowcast_rows(
        train,
        region_to_code,
        weather_stats_by_region,
        args.train_samples_per_region,
        args.valid_anchors_per_region,
        args.seed,
        include_recent_for_train=False,
    )
    print(f"Nowcast train rows: {x_now_train.shape}; valid rows: {x_now_valid.shape}")

    nowcast_model = make_gap_model("lightgbm", args.seed, args.fast, args.device)
    nowcast_model.fit(x_now_train, y_now_train)
    current_valid_pred = np.clip(nowcast_model.predict(x_now_valid), 0.0, 5.0).astype(np.float32)
    current_valid_mae = float(mean_absolute_error(y_now_valid, current_valid_pred))
    current_shift = float(np.median(y_now_valid - current_valid_pred))
    current_valid_pred_cal = np.clip(current_valid_pred + current_shift, 0.0, 5.0).astype(np.float32)
    current_valid_mae_cal = float(mean_absolute_error(y_now_valid, current_valid_pred_cal))

    print("Training validation score-lag forecast models ...")
    labels = train[pd.to_numeric(train[TARGET_COL], errors="coerce").notna()].copy()
    labels["score"] = pd.to_numeric(labels[TARGET_COL], errors="raise").astype(np.float32)
    x_forecast_train, y_forecast_train = build_forecast_rows(labels, args.lags, args.valid_weeks)
    forecast_models = fit_forecast_models(x_forecast_train, y_forecast_train, args.seed, args.fast, args.device)

    train_label_groups = {
        str(region): group.reset_index(drop=True) for region, group in labels.groupby(REGION_COL, sort=False)
    }
    combined_pred_rows: list[np.ndarray] = []
    combined_true_rows: list[np.ndarray] = []
    for case, predicted_current in zip(valid_cases, current_valid_pred_cal, strict=True):
        label_group = train_label_groups[case.region]
        score_values = label_group["score"].to_numpy(dtype=np.float32)
        score_dates = label_group[DATE_COL].astype(str).tolist()
        combined_pred_rows.append(
            forecast_from_history(
                forecast_models,
                score_values,
                score_dates,
                case.cutoff_score_pos,
                float(predicted_current),
                case.anchor_date,
                args.lags,
                case.region_code,
            )
        )
        combined_true_rows.append(case.future_target)
    combined_pred = np.vstack(combined_pred_rows).astype(np.float32)
    combined_true = np.vstack(combined_true_rows).astype(np.float32)
    combined_mae_raw = float(mean_absolute_error(combined_true.ravel(), combined_pred.ravel()))
    horizon_shift = np.median(combined_true - combined_pred, axis=0).astype(np.float32)
    combined_pred_cal = np.clip(combined_pred + horizon_shift[None, :], 0.0, 5.0)
    combined_mae = float(mean_absolute_error(combined_true.ravel(), combined_pred_cal.ravel()))

    print("Training final nowcast and forecast models ...")
    x_now_final, y_now_final, _unused_cases, _unused_x, _unused_y = build_nowcast_rows(
        train,
        region_to_code,
        weather_stats_by_region,
        args.final_train_samples_per_region,
        0,
        args.seed + 17,
        include_recent_for_train=True,
    )
    final_nowcast_model = make_gap_model("lightgbm", args.seed + 707, args.fast, args.device)
    final_nowcast_model.fit(x_now_final, y_now_final)
    x_forecast_final, y_forecast_final = build_forecast_rows(labels, args.lags, valid_weeks=None)
    final_forecast_models = fit_forecast_models(x_forecast_final, y_forecast_final, args.seed + 909, args.fast, args.device)

    print("Building test rows ...")
    train_daily_groups = {str(region): group.reset_index(drop=True) for region, group in train.groupby(REGION_COL, sort=False)}
    test_groups = {str(region): group.reset_index(drop=True) for region, group in test.groupby(REGION_COL, sort=False)}
    train_meta = train.groupby(REGION_COL, sort=False).agg(train_end=(DATE_COL, "last"))
    test_meta = test.groupby(REGION_COL, sort=False).agg(test_end=(DATE_COL, "last"))
    train_end_ord = train_meta["train_end"].map(date_ordinal)
    test_end_ord = test_meta["test_end"].map(date_ordinal)

    test_nowcast_rows: list[list[float]] = []
    test_context: list[tuple[str, int, str, int]] = []
    for region in region_order:
        train_group = train_daily_groups[region]
        test_group = test_groups[region]
        score = pd.to_numeric(train_group[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)
        score_idx = np.flatnonzero(np.isfinite(score)).astype(np.int32)
        score_values = score[score_idx].astype(np.float32)
        cutoff_score_pos = len(score_values) - 1
        weather_values = test_group[WEATHER_COLS].to_numpy(dtype=np.float32)[-91:]
        end_date = str(test_group[DATE_COL].iloc[-1])
        anchor_index = len(train_group) - 1 + int(test_end_ord[region] - train_end_ord[region])
        score_cutoff = int(score_idx[-1])
        region_stats = build_region_stats(score_values)
        test_nowcast_rows.append(
            make_row(
                weather_values,
                end_date,
                anchor_index,
                score_cutoff,
                score_idx,
                score_values,
                region_to_code[region],
                region_stats,
                weather_stats_by_region[region],
            )
        )
        test_context.append((region, cutoff_score_pos, end_date, region_to_code[region]))

    x_test_nowcast = np.asarray(test_nowcast_rows, dtype=np.float32)
    test_current_raw = np.clip(final_nowcast_model.predict(x_test_nowcast), 0.0, 5.0).astype(np.float32)
    test_current = np.clip(test_current_raw + current_shift, 0.0, 5.0).astype(np.float32)

    test_pred_rows: list[np.ndarray] = []
    for predicted_current, (region, cutoff_score_pos, end_date, region_code) in zip(test_current, test_context, strict=True):
        label_group = train_label_groups[region]
        score_values = label_group["score"].to_numpy(dtype=np.float32)
        score_dates = label_group[DATE_COL].astype(str).tolist()
        test_pred_rows.append(
            forecast_from_history(
                final_forecast_models,
                score_values,
                score_dates,
                cutoff_score_pos,
                float(predicted_current),
                end_date,
                args.lags,
                region_code,
            )
        )
    test_pred_raw = np.vstack(test_pred_rows).astype(np.float32)
    test_pred = np.clip(test_pred_raw + horizon_shift[None, :], 0.0, 5.0)
    format_submission(sample_path, pd.DataFrame({REGION_COL: region_order}), test_pred, Path(args.output), 0.0, 5.0)

    report = {
        "current_nowcast_mae_raw": current_valid_mae,
        "current_nowcast_mae": current_valid_mae_cal,
        "current_shift": current_shift,
        "validation_mae_raw": combined_mae_raw,
        "validation_mae": combined_mae,
        "horizon_mae": {
            f"week_{idx + 1}": float(mean_absolute_error(combined_true[:, idx], combined_pred_cal[:, idx]))
            for idx in range(5)
        },
        "horizon_shift": {f"week_{idx + 1}": float(horizon_shift[idx]) for idx in range(5)},
        "nowcast_train_rows": int(x_now_train.shape[0]),
        "nowcast_final_rows": int(x_now_final.shape[0]),
        "forecast_train_rows": int(x_forecast_train.shape[0]),
        "forecast_final_rows": int(x_forecast_final.shape[0]),
        "features_nowcast": int(x_now_train.shape[1]),
        "features_forecast": int(x_forecast_train.shape[1]),
        "test_current_mean": float(test_current.mean()),
        "test_current_raw_mean": float(test_current_raw.mean()),
        "test_prediction_mean": float(test_pred.mean()),
        "test_prediction_raw_mean": float(test_pred_raw.mean()),
        "test_prediction_min": float(test_pred.min()),
        "test_prediction_max": float(test_pred.max()),
        "test_prediction_distribution": output_distribution(test_pred),
        "submission": args.output,
        "args": vars(args),
    }
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
