#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from drought.features import (
    DATE_COL,
    REGION_COL,
    TARGET_COL,
    cycle_phase_parts,
    date_dayofweek,
    date_dayofyear,
    date_ordinal,
)
from train_predict import format_submission

# Cycle period from ACF analysis (reports/data_insights.json["periodicity"]["best_P"])
CYCLE_PERIOD = 2184


warnings.filterwarnings("ignore", message="X does not have valid feature names")

WEATHER_COLS = [
    "prec",
    "surf_pre",
    "humidity",
    "tmp",
    "dp_tmp",
    "wb_tmp",
    "tmp_max",
    "tmp_min",
    "tmp_range",
    "surf_tmp",
    "wind",
    "wind_max",
    "wind_min",
    "wind_range",
]
WINDOWS = (7, 14, 28, 56, 91)
LABEL_WINDOWS = (4, 13, 26, 52)

# Track C (Plan v4) — extended-feature toggle. Set EXTRA_SIGNAL_FEATURES=1 to enable the
# ~20 new features (pressure anomaly, precip intensity, ET interaction, neighbor ensemble,
# DOY seasonality). Default OFF for backward compatibility with prior reports.
EXTRA_SIGNAL_FEATURES = bool(int(os.environ.get("EXTRA_SIGNAL_FEATURES", "0")))
PREC_INTENSITY_THRESHOLD = 5.0  # mm/day for "significant rainfall"
DOY_BUCKET_DAYS = 14  # ~26 buckets per 366-day year


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a gap-aware local model and build a candidate submission.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="submissions/submission_gap_model.csv")
    parser.add_argument("--report-output", default="reports/gap_model.json")
    parser.add_argument(
        "--validation-pred-output",
        default=None,
        help="Optional CSV path for validation truth and predictions, useful for post-processing studies.",
    )
    parser.add_argument("--train-samples-per-region", type=int, default=96)
    parser.add_argument("--valid-anchors-per-region", type=int, default=2)
    parser.add_argument(
        "--valid-deltas",
        default=None,
        help=(
            "Optional comma-separated day offsets from each region's last usable anchor "
            "for validation, e.g. '0,365,735'. Defaults to linspace(0,365,valid_anchors_per_region)."
        ),
    )
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--models", default="lightgbm", choices=["lightgbm", "hgb"])
    parser.add_argument("--objective", default="regression", choices=["regression", "ordinal", "blend"])
    parser.add_argument(
        "--blend-step",
        type=float,
        default=0.05,
        help="Grid step for validation-selected ordinal weight when --objective blend.",
    )
    parser.add_argument(
        "--region-calibration-alpha",
        type=float,
        default=0.0,
        help="Shrinked region-horizon residual correction strength. Set 0 to disable.",
    )
    parser.add_argument(
        "--region-calibration-smoothing",
        type=float,
        default=4.0,
        help="Larger values shrink region-horizon residual corrections more strongly.",
    )
    parser.add_argument(
        "--refit-all-for-test",
        action="store_true",
        help="After validation, refit the selected objective on train+validation rows before predicting test.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument(
        "--gap-mode",
        choices=["public_like", "zero", "blackout91"],
        default="public_like",
        help="Use test-like stale score gaps, current-anchor score history, or a 91-day score blackout.",
    )
    parser.add_argument("--fast", action="store_true")
    # Plan v6 adversarial-validation selection
    parser.add_argument(
        "--adversarial-scores",
        default=None,
        help=(
            "Path to adversarial scores CSV/parquet with (region_id, anchor_index, p_test). "
            "When set, training and validation anchors are selected by p_test instead of "
            "valid_deltas + uniform random sampling."
        ),
    )
    parser.add_argument(
        "--adversarial-val-frac",
        type=float,
        default=0.05,
        help="Top fraction (per region) of adversarial-scored anchors used for validation.",
    )
    parser.add_argument(
        "--adversarial-train-frac",
        type=float,
        default=0.40,
        help="Next top fraction (per region) used for training (after validation slice).",
    )
    parser.add_argument(
        "--adversarial-train-cap",
        type=int,
        default=48,
        help="Max training anchors per region selected from adversarial pool.",
    )
    return parser.parse_args()


def parse_valid_deltas(value: str | None, valid_anchors_per_region: int) -> list[int]:
    if value is None or not value.strip():
        return [int(v) for v in np.linspace(0, 365, max(1, valid_anchors_per_region), dtype=int)]
    deltas: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        delta = int(part)
        if delta < 0:
            raise ValueError("--valid-deltas values must be non-negative")
        deltas.append(delta)
    if not deltas:
        raise ValueError("--valid-deltas did not contain any valid integer offsets")
    return deltas


def score_targets(score_idx: np.ndarray, score_values: np.ndarray, anchor: int, horizons: int = 5) -> np.ndarray | None:
    start = int(np.searchsorted(score_idx, anchor, side="right"))
    end = start + horizons
    if end > score_idx.size:
        return None
    return score_values[start:end].astype(np.float32)


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


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for value in mask:
        if bool(value):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def days_since_true(mask: np.ndarray) -> int:
    if not mask.any():
        return int(mask.size)
    return int(mask.size - 1 - np.flatnonzero(mask)[-1])


def date_features(date_value: str) -> list[float]:
    year, month, day = date_parts(date_value)
    doy = date_dayofyear(date_value)
    dow = date_dayofweek(date_value)
    cp = cycle_phase_parts(date_value, period=CYCLE_PERIOD)
    return [
        float(month),
        float((month - 1) // 3 + 1),
        float(day),
        float(doy),
        float(dow),
        float(np.sin(2 * np.pi * month / 12.0)),
        float(np.cos(2 * np.pi * month / 12.0)),
        float(np.sin(2 * np.pi * doy / 366.0)),
        float(np.cos(2 * np.pi * doy / 366.0)),
        float(np.sin(2 * np.pi * dow / 7.0)),
        float(np.cos(2 * np.pi * dow / 7.0)),
        float(year % 4),
        # Cycle-phase features (P=2184d from ACF; see reports/data_insights.json)
        cp["cycle_phase"],
        cp["cycle_phase_sin"],
        cp["cycle_phase_cos"],
        cp["cycle_phase_ordinal"],
    ]


def date_parts(date_value: object) -> tuple[int, int, int]:
    if hasattr(date_value, "year") and hasattr(date_value, "month") and hasattr(date_value, "day"):
        return int(date_value.year), int(date_value.month), int(date_value.day)
    year_text, month_text, day_text = str(date_value).rsplit("-", 2)
    return int(year_text), int(month_text), int(day_text)


def date_month(date_value: object) -> int:
    return date_parts(date_value)[1]


def label_features(
    score_idx: np.ndarray,
    score_values: np.ndarray,
    cutoff: int,
    region_stats: dict[str, float],
) -> list[float]:
    pos = int(np.searchsorted(score_idx, cutoff, side="right"))
    hist = score_values[:pos].astype(np.float32)
    feats: list[float] = [float(pos)]
    if hist.size:
        last_idx = int(score_idx[pos - 1])
        feats.extend(
            [
                float(hist[-1]),
                float(cutoff - last_idx),
                float(hist.mean()),
                float(np.median(hist)),
                float(hist.std()),
                float(hist.max()),
                float((hist > 0).mean()),
                float((hist >= 2).mean()),
            ]
        )
    else:
        feats.extend(
            [
                region_stats["median"],
                9999.0,
                region_stats["mean"],
                region_stats["median"],
                region_stats["std"],
                region_stats["max"],
                region_stats["positive_frac"],
                region_stats["ge2_frac"],
            ]
        )

    for window in LABEL_WINDOWS:
        recent = hist[-window:]
        if recent.size:
            feats.extend(
                [
                    float(recent[-1]),
                    float(recent.mean()),
                    float(np.median(recent)),
                    float(recent.std()),
                    float(recent.max()),
                    slope(recent),
                ]
            )
        else:
            feats.extend(
                [
                    region_stats["median"],
                    region_stats["mean"],
                    region_stats["median"],
                    region_stats["std"],
                    region_stats["max"],
                    0.0,
                ]
            )
    return feats


def weather_features(
    weather: np.ndarray,
    weather_stats: dict[str, np.ndarray] | None = None,
    end_date: str | None = None,
) -> list[float]:
    feats: list[float] = []
    mean_baseline = weather_stats["mean"] if weather_stats is not None else None
    std_baseline = weather_stats["std"] if weather_stats is not None else None
    month_mean_baseline = None
    month_std_baseline = None
    if weather_stats is not None and end_date is not None:
        month = int(end_date.split("-")[1])
        month_mean_baseline = weather_stats["month_mean"][month]
        month_std_baseline = weather_stats["month_std"][month]
    for col_idx in range(weather.shape[1]):
        values_91 = weather[:, col_idx].astype(np.float32)
        full_mean = float(values_91.mean())
        baseline_mean = float(mean_baseline[col_idx]) if mean_baseline is not None else 0.0
        baseline_std = float(std_baseline[col_idx]) if std_baseline is not None else 1.0
        baseline_std = max(baseline_std, 1e-6)
        month_baseline_mean = (
            float(month_mean_baseline[col_idx]) if month_mean_baseline is not None else baseline_mean
        )
        month_baseline_std = float(month_std_baseline[col_idx]) if month_std_baseline is not None else baseline_std
        month_baseline_std = max(month_baseline_std, 1e-6)
        for window in WINDOWS:
            values = values_91[-window:]
            window_mean = float(values.mean())
            window_last = float(values[-1])
            feats.extend(
                [
                    window_mean,
                    float(values.std()),
                    float(values.min()),
                    float(values.max()),
                    window_last,
                    float(values[-1] - values[0]),
                    window_mean - full_mean,
                    window_mean - baseline_mean,
                    (window_mean - baseline_mean) / baseline_std,
                    (window_last - baseline_mean) / baseline_std,
                    window_mean - month_baseline_mean,
                    (window_mean - month_baseline_mean) / month_baseline_std,
                    (window_last - month_baseline_mean) / month_baseline_std,
                ]
            )
            if col_idx == 0:
                expected_sum = baseline_mean * window
                expected_month_sum = month_baseline_mean * window
                dry_mask = values <= 1e-6
                wet_mask = values > 1e-6
                feats.extend(
                    [
                        float(values.sum()),
                        float(values.sum() - expected_sum),
                        float(values.sum() - expected_month_sum),
                        float(dry_mask.sum()),
                        float(wet_mask.sum()),
                        float(longest_true_run(dry_mask)),
                        float(days_since_true(wet_mask)),
                    ]
                )
    col_to_idx = {name: idx for idx, name in enumerate(WEATHER_COLS)}
    for window in WINDOWS:
        values = weather[-window:].astype(np.float32)
        prec_sum = float(values[:, col_to_idx["prec"]].sum())
        humidity_mean = float(values[:, col_to_idx["humidity"]].mean())
        tmp_mean = float(values[:, col_to_idx["tmp"]].mean())
        tmp_max_mean = float(values[:, col_to_idx["tmp_max"]].mean())
        dp_gap_mean = float((values[:, col_to_idx["tmp"]] - values[:, col_to_idx["dp_tmp"]]).mean())
        wb_gap_mean = float((values[:, col_to_idx["tmp"]] - values[:, col_to_idx["wb_tmp"]]).mean())
        feats.extend(
            [
                prec_sum,
                humidity_mean,
                tmp_mean,
                tmp_max_mean,
                dp_gap_mean,
                wb_gap_mean,
                tmp_max_mean - humidity_mean,
                tmp_max_mean - humidity_mean - prec_sum / max(1, window),
                float(values[:, col_to_idx["tmp"]].mean() - weather[:, col_to_idx["tmp"]].mean()),
                float(values[:, col_to_idx["prec"]].sum() - weather[:, col_to_idx["prec"]].mean() * window),
            ]
        )
    return feats


def build_region_neighbor_index(all_regions: list[str]) -> dict[str, list[str]]:
    """For each region, list up to 4 numeric neighbors (R±1, R±2 by numeric id parse).

    92% of regions have at least one numeric neighbor present (per data audit).
    Used by `neighbor_features` to inject cross-region drought correlation signal.
    """
    region_to_num: dict[str, int] = {}
    num_to_region: dict[int, str] = {}
    for r in all_regions:
        m = re.match(r"R(\d+)", str(r))
        if m:
            n = int(m.group(1))
            region_to_num[r] = n
            num_to_region[n] = r
    neighbors: dict[str, list[str]] = {}
    for r, n in region_to_num.items():
        nb: list[str] = []
        for offset in (-2, -1, 1, 2):
            target = num_to_region.get(n + offset)
            if target is not None:
                nb.append(target)
        neighbors[r] = nb
    return neighbors


def build_region_score_summary(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-region precomputed score statistics for neighbor features.

    Uses ALL training history per region. Slight leak risk (anchors at different cutoffs
    see the same summary), but the leak is into neighbor scores, not own scores, so
    cross-region correlation signal still flows correctly.
    """
    summary: dict[str, dict[str, float]] = {}
    for region, group in train.groupby(REGION_COL, sort=False):
        scores = pd.to_numeric(group[TARGET_COL], errors="coerce").dropna().to_numpy(dtype=np.float32)
        if scores.size == 0:
            summary[str(region)] = {
                "recent_13": 0.0,
                "recent_52": 0.0,
                "last": 0.0,
                "available": 0.0,
            }
        else:
            summary[str(region)] = {
                "recent_13": float(scores[-13:].mean()),
                "recent_52": float(scores[-52:].mean()),
                "last": float(scores[-1]),
                "available": 1.0,
            }
    return summary


def build_region_doy_baseline(train: pd.DataFrame) -> dict[str, dict[int, dict[str, float]]]:
    """Per-region per-DOY-bucket baseline (mean, std, count) of observed scores."""
    baselines: dict[str, dict[int, dict[str, float]]] = {}
    for region, group in train.groupby(REGION_COL, sort=False):
        g = group.dropna(subset=[TARGET_COL])
        if g.empty:
            baselines[str(region)] = {}
            continue
        doys = g[DATE_COL].apply(date_dayofyear).to_numpy(dtype=np.int32)
        scores = g[TARGET_COL].to_numpy(dtype=np.float32)
        buckets = (doys // DOY_BUCKET_DAYS).astype(np.int32)
        per_bucket: dict[int, dict[str, float]] = {}
        for b in np.unique(buckets):
            mask = buckets == b
            cnt = int(mask.sum())
            per_bucket[int(b)] = {
                "mean": float(scores[mask].mean()),
                "std": float(scores[mask].std()) if cnt > 1 else 0.5,
                "count": float(cnt),
            }
        baselines[str(region)] = per_bucket
    return baselines


def neighbor_features(
    region: str,
    neighbors_idx: dict[str, list[str]],
    score_summary: dict[str, dict[str, float]],
    region_stats: dict[str, float],
) -> list[float]:
    """4 features: neighbor mean recent-13, recent-52, last, available count."""
    nb_list = neighbors_idx.get(region, [])
    if not nb_list:
        return [region_stats["mean"], region_stats["mean"], region_stats["last"], 0.0]
    means_13: list[float] = []
    means_52: list[float] = []
    last_scores: list[float] = []
    for nb in nb_list:
        s = score_summary.get(nb)
        if s is None or s.get("available", 0.0) < 0.5:
            continue
        means_13.append(s["recent_13"])
        means_52.append(s["recent_52"])
        last_scores.append(s["last"])
    if not means_13:
        return [region_stats["mean"], region_stats["mean"], region_stats["last"], 0.0]
    return [
        float(np.mean(means_13)),
        float(np.mean(means_52)),
        float(np.mean(last_scores)),
        float(len(means_13)),
    ]


def doy_seasonality_features(
    region: str,
    end_date: str,
    doy_baseline: dict[str, dict[int, dict[str, float]]],
    region_stats: dict[str, float],
) -> list[float]:
    """4 features: doy bucket mean, std, count, and deviation of region_stats.last from baseline."""
    rb = doy_baseline.get(region)
    if not rb:
        return [region_stats["mean"], 0.5, 0.0, 0.0]
    doy = date_dayofyear(end_date)
    bucket = int(doy // DOY_BUCKET_DAYS)
    info = rb.get(bucket)
    if info is None:
        keys = sorted(rb.keys())
        if not keys:
            return [region_stats["mean"], 0.5, 0.0, 0.0]
        info = rb[min(keys, key=lambda k: abs(k - bucket))]
    return [
        info["mean"],
        info["std"],
        info["count"],
        float(region_stats["last"] - info["mean"]),
    ]


def extra_signal_features(
    weather: np.ndarray,
    end_date: str,
    region: str,
    region_stats: dict[str, float],
    neighbors_idx: dict[str, list[str]],
    score_summary: dict[str, dict[str, float]],
    doy_baseline: dict[str, dict[int, dict[str, float]]],
) -> list[float]:
    """20-feature Track-C signal pack: pressure anomaly (4), precip intensity (5),
    ET interaction (3), neighbor ensemble (4), DOY seasonality (4)."""
    feats: list[float] = []
    col_to_idx = {name: idx for idx, name in enumerate(WEATHER_COLS)}

    # Pressure anomaly (4)
    surf_pre = weather[:, col_to_idx["surf_pre"]].astype(np.float32)
    pre_full_mean = float(surf_pre.mean())
    pre_full_std = max(float(surf_pre.std()), 1e-6)
    for window in (7, 28, 91):
        win = surf_pre[-window:]
        feats.append((float(win.mean()) - pre_full_mean) / pre_full_std)
    feats.append(slope(surf_pre))

    # Precipitation intensity (5)
    prec = weather[:, col_to_idx["prec"]].astype(np.float32)
    p91 = prec
    p28 = prec[-28:]
    feats.append(float(p91.std()) / max(float(p91.mean()), 1e-6))
    p90_91 = float(np.percentile(p91, 90)) if p91.size > 0 else 0.0
    feats.append(float((p91 > p90_91).sum()))
    p90_28 = float(np.percentile(p28, 90)) if p28.size > 0 else 0.0
    feats.append(float((p28 > p90_28).sum()))
    sig_mask = p91 > PREC_INTENSITY_THRESHOLD
    feats.append(float(days_since_true(sig_mask)))
    sorted_p = np.sort(p91)
    n = sorted_p.size
    if n > 0 and sorted_p.sum() > 0:
        gini = float((np.arange(1, n + 1) * sorted_p).sum() / (n * sorted_p.sum()) - (n + 1) / (2 * n) * 2)
    else:
        gini = 0.0
    feats.append(gini)

    # ET interaction (3)
    tmp = weather[:, col_to_idx["tmp"]].astype(np.float32)
    dp_tmp = weather[:, col_to_idx["dp_tmp"]].astype(np.float32)
    wind = weather[:, col_to_idx["wind"]].astype(np.float32)
    humidity = weather[:, col_to_idx["humidity"]].astype(np.float32)
    et_full = (tmp - dp_tmp) * wind * (1.0 - humidity / 100.0)
    for window in (7, 28, 91):
        feats.append(float(et_full[-window:].mean()))

    # Neighbor ensemble (4)
    feats.extend(neighbor_features(region, neighbors_idx, score_summary, region_stats))

    # DOY seasonality (4)
    feats.extend(doy_seasonality_features(region, end_date, doy_baseline, region_stats))

    return feats


def make_row(
    weather_window: np.ndarray,
    end_date: str,
    anchor_index: int,
    score_cutoff: int,
    score_idx: np.ndarray,
    score_values: np.ndarray,
    region_code: int,
    region_stats: dict[str, float],
    weather_stats: dict[str, np.ndarray] | None = None,
    extra_context: dict[str, Any] | None = None,
) -> list[float]:
    feats: list[float] = [
        float(region_code),
        float(anchor_index),
        float(score_cutoff),
        float(anchor_index - score_cutoff),
    ]
    feats.extend(date_features(end_date))
    feats.extend(
        [
            region_stats["mean"],
            region_stats["median"],
            region_stats["std"],
            region_stats["max"],
            region_stats["positive_frac"],
            region_stats["ge2_frac"],
            region_stats["last"],
        ]
    )
    feats.extend(label_features(score_idx, score_values, score_cutoff, region_stats))
    feats.extend(weather_features(weather_window, weather_stats, end_date))
    if extra_context is not None and EXTRA_SIGNAL_FEATURES:
        feats.extend(
            extra_signal_features(
                weather_window,
                end_date,
                extra_context["region"],
                region_stats,
                extra_context["neighbors_idx"],
                extra_context["score_summary"],
                extra_context["doy_baseline"],
            )
        )
    return feats


def score_cutoff_for_mode(anchor: int, gap: int, mode: str) -> int:
    if mode == "zero":
        return anchor
    if mode == "blackout91":
        return max(0, anchor - 91)
    return max(0, anchor - gap)


def output_distribution(values: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for horizon in range(values.shape[1]):
        col = values[:, horizon]
        result[f"week_{horizon + 1}"] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "pct_gt_0_5": float((col > 0.5).mean()),
            "pct_gt_1_5": float((col > 1.5).mean()),
            "pct_gt_2_5": float((col > 2.5).mean()),
        }
    return result


def build_weather_stats(group: pd.DataFrame) -> dict[str, np.ndarray]:
    values = group[WEATHER_COLS].to_numpy(dtype=np.float32)
    global_mean = values.mean(axis=0)
    global_std = np.maximum(values.std(axis=0), 1e-6)
    months = group[DATE_COL].map(date_month).to_numpy(dtype=np.int16)
    month_mean = np.zeros((13, len(WEATHER_COLS)), dtype=np.float32)
    month_std = np.zeros((13, len(WEATHER_COLS)), dtype=np.float32)
    for month in range(1, 13):
        month_values = values[months == month]
        if month_values.size == 0:
            month_mean[month] = global_mean
            month_std[month] = global_std
        else:
            month_mean[month] = month_values.mean(axis=0)
            month_std[month] = np.maximum(month_values.std(axis=0), 1e-6)
    return {
        "mean": global_mean,
        "std": global_std,
        "month_mean": month_mean,
        "month_std": month_std,
    }


def make_model(name: str, seed: int, fast: bool, device: str) -> Any:
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="mae",
            n_estimators=550 if fast else 1200,
            learning_rate=0.045 if fast else 0.028,
            num_leaves=63,
            min_child_samples=45,
            subsample=0.86,
            subsample_freq=1,
            colsample_bytree=0.78,
            reg_alpha=0.05,
            reg_lambda=0.45,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            device_type=device,
        )
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=180 if fast else 700,
            learning_rate=0.07 if fast else 0.04,
            max_leaf_nodes=31,
            min_samples_leaf=35,
            l2_regularization=0.05,
            random_state=seed,
        )
    raise ValueError(name)


def make_classifier(seed: int, fast: bool, device: str) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary",
        n_estimators=320 if fast else 900,
        learning_rate=0.055 if fast else 0.03,
        num_leaves=63,
        min_child_samples=45,
        subsample=0.86,
        subsample_freq=1,
        colsample_bytree=0.78,
        reg_alpha=0.05,
        reg_lambda=0.45,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        device_type=device,
    )


def fit_ordinal_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    seed: int,
    fast: bool,
    device: str,
) -> tuple[list[list[Any]], np.ndarray]:
    models_by_horizon: list[list[Any]] = []
    valid_pred = np.zeros((x_valid.shape[0], y_train.shape[1]), dtype=np.float32)
    for horizon in range(y_train.shape[1]):
        threshold_models: list[Any] = []
        pred = np.zeros(x_valid.shape[0], dtype=np.float32)
        for threshold in range(1, 6):
            binary_target = (y_train[:, horizon] >= threshold).astype(np.int8)
            if binary_target.min() == binary_target.max():
                constant = float(binary_target[0])
                threshold_models.append(constant)
                pred += constant
                continue
            model = make_classifier(seed + horizon * 101 + threshold * 17, fast, device)
            model.fit(x_train, binary_target)
            pred += model.predict_proba(x_valid)[:, 1].astype(np.float32)
            threshold_models.append(model)
        valid_pred[:, horizon] = np.clip(pred, 0.0, 5.0)
        models_by_horizon.append(threshold_models)
    return models_by_horizon, valid_pred


def fit_regression_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    model_name: str,
    seed: int,
    fast: bool,
    device: str,
) -> tuple[list[Any], np.ndarray]:
    models: list[Any] = []
    valid_pred = np.zeros((x_valid.shape[0], y_train.shape[1]), dtype=np.float32)
    for horizon in range(y_train.shape[1]):
        model = make_model(model_name, seed + horizon * 101, fast, device)
        model.fit(x_train, y_train[:, horizon])
        valid_pred[:, horizon] = np.clip(model.predict(x_valid), 0.0, 5.0).astype(np.float32)
        models.append(model)
    return models, valid_pred


def predict_regression_models(models: list[Any], x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.clip(model.predict(x), 0.0, 5.0) for model in models]).astype(np.float32)


def predict_ordinal_models(models_by_horizon: list[list[Any]], x: np.ndarray) -> np.ndarray:
    predictions = np.zeros((x.shape[0], len(models_by_horizon)), dtype=np.float32)
    for horizon, threshold_models in enumerate(models_by_horizon):
        pred = np.zeros(x.shape[0], dtype=np.float32)
        for model in threshold_models:
            if isinstance(model, float):
                pred += model
            else:
                pred += model.predict_proba(x)[:, 1].astype(np.float32)
        predictions[:, horizon] = np.clip(pred, 0.0, 5.0)
    return predictions


def choose_blend(
    y_valid: np.ndarray,
    regression_pred: np.ndarray,
    ordinal_pred: np.ndarray,
    step: float,
) -> tuple[float, np.ndarray, list[dict[str, float]]]:
    step = float(np.clip(step, 0.01, 1.0))
    weights = np.arange(0.0, 1.0 + step / 2.0, step, dtype=np.float32)
    if weights[-1] < 1.0:
        weights = np.append(weights, np.float32(1.0))

    candidates: list[dict[str, float]] = []
    best_weight = 0.0
    best_mae = np.inf
    best_pred = regression_pred
    for ordinal_weight in weights:
        pred = (1.0 - float(ordinal_weight)) * regression_pred + float(ordinal_weight) * ordinal_pred
        pred = np.clip(pred, 0.0, 5.0).astype(np.float32)
        mae = float(mean_absolute_error(y_valid.ravel(), pred.ravel()))
        candidates.append({"ordinal_weight": float(ordinal_weight), "mae": mae})
        if mae < best_mae:
            best_mae = mae
            best_weight = float(ordinal_weight)
            best_pred = pred
    return best_weight, best_pred, candidates


def fit_region_horizon_corrections(
    regions: np.ndarray,
    residuals: np.ndarray,
    alpha: float,
    smoothing: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float | int | bool]]:
    if alpha <= 0:
        return {}, np.zeros_like(residuals, dtype=np.float32), {"enabled": False}

    corrections: dict[str, np.ndarray] = {}
    row_corrections = np.zeros_like(residuals, dtype=np.float32)
    smoothing = max(0.0, float(smoothing))
    alpha = max(0.0, float(alpha))

    for region in np.unique(regions.astype(str)):
        mask = regions == region
        count = int(mask.sum())
        if count == 0:
            continue
        shrink = alpha * count / (count + smoothing) if smoothing > 0 else alpha
        correction = (np.median(residuals[mask], axis=0) * shrink).astype(np.float32)
        corrections[str(region)] = correction
        row_corrections[mask] = correction

    if corrections:
        values = np.vstack(list(corrections.values()))
        summary: dict[str, float | int | bool] = {
            "enabled": True,
            "alpha": float(alpha),
            "smoothing": float(smoothing),
            "regions": int(len(corrections)),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
            "mean_abs": float(np.mean(np.abs(values))),
        }
    else:
        summary = {"enabled": False}
    return corrections, row_corrections, summary


def corrections_for_regions(regions: list[str], corrections: dict[str, np.ndarray], horizons: int) -> np.ndarray:
    if not corrections:
        return np.zeros((len(regions), horizons), dtype=np.float32)
    zero = np.zeros(horizons, dtype=np.float32)
    return np.vstack([corrections.get(str(region), zero) for region in regions]).astype(np.float32)


def build_region_stats(score_values: np.ndarray) -> dict[str, float]:
    if score_values.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "max": 0.0,
            "positive_frac": 0.0,
            "ge2_frac": 0.0,
            "last": 0.0,
        }
    return {
        "mean": float(score_values.mean()),
        "median": float(np.median(score_values)),
        "std": float(score_values.std()),
        "max": float(score_values.max()),
        "positive_frac": float((score_values > 0).mean()),
        "ge2_frac": float((score_values >= 2).mean()),
        "last": float(score_values[-1]),
    }


def stats_before_cutoff(score_idx: np.ndarray, score_values: np.ndarray, cutoff: int) -> dict[str, float]:
    pos = int(np.searchsorted(score_idx, cutoff, side="right"))
    return build_region_stats(score_values[:pos].astype(np.float32))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
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

    train_meta = train.groupby(REGION_COL, sort=False).agg(train_end=(DATE_COL, "last"))
    test_meta = test.groupby(REGION_COL, sort=False).agg(test_end=(DATE_COL, "last"))
    train_end_ord = train_meta["train_end"].map(date_ordinal)
    test_end_ord = test_meta["test_end"].map(date_ordinal)
    public_like_gap_by_region = (test_end_ord - train_end_ord + 6).astype(int).to_dict()
    public_like_gaps = np.array(list(public_like_gap_by_region.values()), dtype=np.int32)
    weather_stats_by_region = {
        str(region): build_weather_stats(group) for region, group in train.groupby(REGION_COL, sort=False)
    }

    # Track C (Plan v4) — precompute indexes for extra signal features.
    if EXTRA_SIGNAL_FEATURES:
        print("Building extra-signal indexes (neighbors, score summary, DOY baselines) ...")
        neighbors_idx = build_region_neighbor_index(region_order)
        score_summary = build_region_score_summary(train)
        doy_baseline = build_region_doy_baseline(train)
        nb_avg = np.mean([len(v) for v in neighbors_idx.values()]) if neighbors_idx else 0.0
        print(f"Neighbors: {len(neighbors_idx)} regions, avg {nb_avg:.2f} neighbors each")
        print(f"DOY baselines: {len(doy_baseline)} regions, ~{DOY_BUCKET_DAYS}-day buckets")
    else:
        neighbors_idx = {}
        score_summary = {}
        doy_baseline = {}

    x_train_rows: list[list[float]] = []
    y_train_rows: list[np.ndarray] = []
    x_valid_rows: list[list[float]] = []
    y_valid_rows: list[np.ndarray] = []
    valid_region_rows: list[str] = []
    validation_deltas = parse_valid_deltas(args.valid_deltas, args.valid_anchors_per_region)
    print(f"Validation deltas from each region's last usable anchor: {validation_deltas}")

    # Plan v6: optional adversarial-validation-based anchor selection.
    adv_train_by_region: dict[str, list[int]] = {}
    adv_val_by_region: dict[str, list[int]] = {}
    if args.adversarial_scores is not None:
        adv_path = Path(args.adversarial_scores)
        if not adv_path.exists():
            raise SystemExit(f"--adversarial-scores file not found: {adv_path}")
        print(f"Loading adversarial scores from {adv_path} ...")
        if adv_path.suffix == ".parquet":
            adv_df = pd.read_parquet(adv_path)
        else:
            adv_df = pd.read_csv(adv_path)
        adv_df["region_id"] = adv_df["region_id"].astype(str)
        # Per region, rank by p_test descending; top val_frac -> val, next train_frac (capped) -> train.
        for region, sub in adv_df.groupby("region_id", sort=False):
            sub = sub.sort_values("p_test", ascending=False)
            n = len(sub)
            n_val = max(1, int(round(args.adversarial_val_frac * n)))
            n_train_cap = min(args.adversarial_train_cap,
                              int(round(args.adversarial_train_frac * n)))
            val_anchors = sub.iloc[:n_val]["anchor_index"].astype(int).tolist()
            train_anchors = sub.iloc[n_val:n_val + n_train_cap]["anchor_index"].astype(int).tolist()
            adv_val_by_region[region] = val_anchors
            adv_train_by_region[region] = train_anchors
        print(f"  loaded scores for {len(adv_val_by_region)} regions")
        sample_region = next(iter(adv_val_by_region))
        print(f"  sample region {sample_region}: {len(adv_val_by_region[sample_region])} val + {len(adv_train_by_region[sample_region])} train anchors")

    print("Building train/validation rows ...")
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
        if last_usable_anchor < 91:
            continue

        gap = int(public_like_gap_by_region[region])
        extra_context = (
            {
                "region": region,
                "neighbors_idx": neighbors_idx,
                "score_summary": score_summary,
                "doy_baseline": doy_baseline,
            }
            if EXTRA_SIGNAL_FEATURES
            else None
        )
        if adv_val_by_region:
            # Plan v6: use adversarial-selected anchors
            valid_anchors = [int(a) for a in adv_val_by_region.get(region, [])
                             if 91 <= int(a) <= last_usable_anchor]
        else:
            valid_anchors = [int(last_usable_anchor - delta) for delta in validation_deltas if last_usable_anchor - delta >= 91]
        valid_anchor_set = set(valid_anchors)
        for anchor in valid_anchors:
            targets = score_targets(score_idx, score_values, anchor)
            if targets is None:
                continue
            cutoff = score_cutoff_for_mode(anchor, gap, args.gap_mode)
            region_stats = stats_before_cutoff(score_idx, score_values, cutoff)
            x_valid_rows.append(
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
                    extra_context,
                )
            )
            y_valid_rows.append(targets)
            valid_region_rows.append(region)

        if adv_train_by_region:
            # Plan v6: use adversarial-selected anchors directly (no random subsampling)
            anchors = np.asarray(
                [a for a in adv_train_by_region.get(region, [])
                 if 91 <= a <= last_usable_anchor and a not in valid_anchor_set],
                dtype=np.int32,
            )
        else:
            candidate_anchors = np.arange(365, max(366, last_usable_anchor - 420), 7, dtype=np.int32)
            if candidate_anchors.size == 0:
                candidate_anchors = np.arange(91, last_usable_anchor + 1, 7, dtype=np.int32)
            sample_size = min(args.train_samples_per_region, candidate_anchors.size)
            anchors = rng.choice(candidate_anchors, size=sample_size, replace=False)
        for anchor in anchors:
            anchor = int(anchor)
            if anchor in valid_anchor_set:
                continue
            targets = score_targets(score_idx, score_values, anchor)
            if targets is None:
                continue
            sampled_gap = int(rng.choice(public_like_gaps))
            cutoff = score_cutoff_for_mode(anchor, sampled_gap, args.gap_mode)
            region_stats = stats_before_cutoff(score_idx, score_values, cutoff)
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
                    extra_context,
                )
            )
            y_train_rows.append(targets)

    x_train = np.asarray(x_train_rows, dtype=np.float32)
    y_train = np.asarray(y_train_rows, dtype=np.float32)
    x_valid = np.asarray(x_valid_rows, dtype=np.float32)
    y_valid = np.asarray(y_valid_rows, dtype=np.float32)
    valid_regions = np.asarray(valid_region_rows, dtype=str)
    print(f"Train rows: {x_train.shape}; valid rows: {x_valid.shape}")

    print("Training horizon models ...")
    blend_weight_ordinal = None
    blend_candidates: list[dict[str, float]] = []
    objective_metrics: dict[str, float] = {}
    if args.objective == "ordinal":
        ordinal_models, valid_pred = fit_ordinal_models(x_train, y_train, x_valid, args.seed, args.fast, args.device)
        models: Any = {"ordinal": ordinal_models}
        objective_metrics["ordinal"] = float(mean_absolute_error(y_valid.ravel(), valid_pred.ravel()))
    elif args.objective == "blend":
        regression_models, regression_pred = fit_regression_models(
            x_train, y_train, x_valid, args.models, args.seed, args.fast, args.device
        )
        ordinal_models, ordinal_pred = fit_ordinal_models(x_train, y_train, x_valid, args.seed + 777, args.fast, args.device)
        blend_weight_ordinal, valid_pred, blend_candidates = choose_blend(
            y_valid, regression_pred, ordinal_pred, args.blend_step
        )
        models = {
            "regression": regression_models,
            "ordinal": ordinal_models,
            "blend_weight_ordinal": blend_weight_ordinal,
        }
        objective_metrics["regression"] = float(mean_absolute_error(y_valid.ravel(), regression_pred.ravel()))
        objective_metrics["ordinal"] = float(mean_absolute_error(y_valid.ravel(), ordinal_pred.ravel()))
        objective_metrics["blend"] = float(mean_absolute_error(y_valid.ravel(), valid_pred.ravel()))
        print(f"  selected ordinal blend weight: {blend_weight_ordinal:.2f}")
    else:
        regression_models, valid_pred = fit_regression_models(
            x_train, y_train, x_valid, args.models, args.seed, args.fast, args.device
        )
        models = {"regression": regression_models}
        objective_metrics["regression"] = float(mean_absolute_error(y_valid.ravel(), valid_pred.ravel()))
    for horizon in range(5):
        print(f"  week_{horizon + 1}: {mean_absolute_error(y_valid[:, horizon], valid_pred[:, horizon]):.6f}")

    validation_mae = float(mean_absolute_error(y_valid.ravel(), valid_pred.ravel()))
    horizon_shift = np.median(y_valid - valid_pred, axis=0).astype(np.float32)
    valid_pred_horizon_calibrated = np.clip(valid_pred + horizon_shift[None, :], 0.0, 5.0)
    validation_mae_horizon_calibrated = float(
        mean_absolute_error(y_valid.ravel(), valid_pred_horizon_calibrated.ravel())
    )
    region_corrections, valid_region_correction, region_calibration_summary = fit_region_horizon_corrections(
        valid_regions,
        y_valid - valid_pred_horizon_calibrated,
        args.region_calibration_alpha,
        args.region_calibration_smoothing,
    )
    valid_pred_calibrated = np.clip(valid_pred_horizon_calibrated + valid_region_correction, 0.0, 5.0)
    validation_mae_calibrated = float(mean_absolute_error(y_valid.ravel(), valid_pred_calibrated.ravel()))
    print(f"Validation MAE: {validation_mae:.6f}")
    print(f"Horizon-calibrated validation MAE: {validation_mae_horizon_calibrated:.6f}")
    print(f"Final calibrated validation MAE: {validation_mae_calibrated:.6f}")

    if args.validation_pred_output:
        validation_pred_path = Path(args.validation_pred_output)
        validation_pred_path.parent.mkdir(parents=True, exist_ok=True)
        validation_rows: list[dict[str, float | str | int]] = []
        for row_idx, region in enumerate(valid_regions):
            for horizon in range(y_valid.shape[1]):
                validation_rows.append(
                    {
                        "row_index": int(row_idx),
                        "region_id": str(region),
                        "horizon": int(horizon + 1),
                        "y_true": float(y_valid[row_idx, horizon]),
                        "pred_raw": float(valid_pred[row_idx, horizon]),
                        "pred_horizon_calibrated": float(valid_pred_horizon_calibrated[row_idx, horizon]),
                        "pred_final_calibrated": float(valid_pred_calibrated[row_idx, horizon]),
                    }
                )
        pd.DataFrame(validation_rows).to_csv(validation_pred_path, index=False)
        print(f"Wrote validation predictions: {validation_pred_path}")

    print("Building test rows ...")
    test_rows: list[list[float]] = []
    train_groups = {str(region): group.reset_index(drop=True) for region, group in train.groupby(REGION_COL, sort=False)}
    test_groups = {str(region): group.reset_index(drop=True) for region, group in test.groupby(REGION_COL, sort=False)}
    for region in region_order:
        train_group = train_groups[region]
        test_group = test_groups[region]
        score = pd.to_numeric(train_group[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)
        score_idx = np.flatnonzero(np.isfinite(score)).astype(np.int32)
        score_values = score[score_idx].astype(np.float32)
        region_stats = build_region_stats(score_values)
        weather_stats = weather_stats_by_region[region]
        weather_values = test_group[WEATHER_COLS].to_numpy(dtype=np.float32)[-91:]
        end_date = str(test_group[DATE_COL].iloc[-1])
        anchor_index = len(train_group) - 1 + int(test_end_ord[region] - train_end_ord[region])
        score_cutoff = int(score_idx[-1])
        test_extra_context = (
            {
                "region": region,
                "neighbors_idx": neighbors_idx,
                "score_summary": score_summary,
                "doy_baseline": doy_baseline,
            }
            if EXTRA_SIGNAL_FEATURES
            else None
        )
        test_rows.append(
            make_row(
                weather_values,
                end_date,
                anchor_index,
                score_cutoff,
                score_idx,
                score_values,
                region_to_code[region],
                region_stats,
                weather_stats,
                test_extra_context,
            )
        )

    x_test = np.asarray(test_rows, dtype=np.float32)
    if args.refit_all_for_test:
        print("Refitting selected objective on train+validation rows for test prediction ...")
        x_final_train = np.vstack([x_train, x_valid]).astype(np.float32)
        y_final_train = np.vstack([y_train, y_valid]).astype(np.float32)
        if args.objective == "ordinal":
            _, test_pred_raw = fit_ordinal_models(
                x_final_train, y_final_train, x_test, args.seed, args.fast, args.device
            )
        elif args.objective == "blend":
            _, test_pred_regression = fit_regression_models(
                x_final_train, y_final_train, x_test, args.models, args.seed, args.fast, args.device
            )
            _, test_pred_ordinal = fit_ordinal_models(
                x_final_train, y_final_train, x_test, args.seed + 777, args.fast, args.device
            )
            weight = float(models["blend_weight_ordinal"])
            test_pred_raw = ((1.0 - weight) * test_pred_regression + weight * test_pred_ordinal).astype(np.float32)
        else:
            _, test_pred_raw = fit_regression_models(
                x_final_train, y_final_train, x_test, args.models, args.seed, args.fast, args.device
            )
    elif args.objective == "ordinal":
        test_pred_raw = predict_ordinal_models(models["ordinal"], x_test)
    elif args.objective == "blend":
        test_pred_regression = predict_regression_models(models["regression"], x_test)
        test_pred_ordinal = predict_ordinal_models(models["ordinal"], x_test)
        weight = float(models["blend_weight_ordinal"])
        test_pred_raw = ((1.0 - weight) * test_pred_regression + weight * test_pred_ordinal).astype(np.float32)
    else:
        test_pred_raw = predict_regression_models(models["regression"], x_test)
    test_region_correction = corrections_for_regions(region_order, region_corrections, y_train.shape[1])
    test_pred = np.clip(test_pred_raw + horizon_shift[None, :] + test_region_correction, 0.0, 5.0)
    format_submission(sample_path, pd.DataFrame({REGION_COL: region_order}), test_pred, Path(args.output), 0.0, 5.0)

    report = {
        "validation_mae_raw": validation_mae,
        "validation_mae_horizon_calibrated": validation_mae_horizon_calibrated,
        "validation_mae": validation_mae_calibrated,
        "horizon_mae": {
            f"week_{h + 1}": float(mean_absolute_error(y_valid[:, h], valid_pred_calibrated[:, h])) for h in range(5)
        },
        "horizon_mae_raw": {
            f"week_{h + 1}": float(mean_absolute_error(y_valid[:, h], valid_pred[:, h])) for h in range(5)
        },
        "horizon_shift": {f"week_{h + 1}": float(horizon_shift[h]) for h in range(5)},
        "region_calibration": region_calibration_summary,
        "objective_metrics": objective_metrics,
        "blend_weight_ordinal": blend_weight_ordinal,
        "blend_candidates": blend_candidates,
        "train_rows": int(x_train.shape[0]),
        "valid_rows": int(x_valid.shape[0]),
        "valid_deltas": validation_deltas,
        "validation_target_mean": float(y_valid.mean()),
        "validation_target_std": float(y_valid.std()),
        "features": int(x_train.shape[1]),
        "model": args.models,
        "objective": args.objective,
        "device": args.device,
        "fast": bool(args.fast),
        "refit_all_for_test": bool(args.refit_all_for_test),
        "submission": args.output,
        "test_prediction_min": float(test_pred.min()),
        "test_prediction_max": float(test_pred.max()),
        "test_prediction_mean": float(test_pred.mean()),
        "test_prediction_raw_mean": float(test_pred_raw.mean()),
        "test_prediction_distribution": output_distribution(test_pred),
        "gap_mode": args.gap_mode,
        "public_like_gap_days": {
            "min": int(public_like_gaps.min()),
            "median": float(np.median(public_like_gaps)),
            "max": int(public_like_gaps.max()),
        },
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
