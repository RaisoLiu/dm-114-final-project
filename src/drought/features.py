from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd


REGION_COL = "region_id"
DATE_COL = "date"
TARGET_COL = "score"
DATE_ORDINAL_COL = "end_date_ordinal"


_DAYS_BEFORE_MONTH_COMMON = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
_DAYS_BEFORE_MONTH_LEAP = (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)
_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass(frozen=True)
class FeatureConfig:
    input_days: int = 91
    horizons: int = 5
    horizon_step_days: int = 7
    windows: tuple[int, ...] = (7, 14, 28, 56, 91)
    dry_threshold: float = 1e-6
    anchor_mode: str = "score_days"
    anchor_stride: int = 7


def safe_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "value"


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _parse_date_parts(value: object) -> tuple[int, int, int]:
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return int(value.year), int(value.month), int(value.day)
    text = str(value)
    year_text, month_text, day_text = text.split("-", 2)
    return int(year_text), int(month_text), int(day_text)


def date_ordinal(value: object) -> int:
    """Return a sortable proleptic-Gregorian day number for synthetic dates.

    The competition dates use years far beyond pandas' timestamp bounds, so
    date arithmetic is intentionally implemented with integer calendar math.
    """

    year, month, day = _parse_date_parts(value)
    year -= month <= 2
    era = (year if year >= 0 else year - 399) // 400
    year_of_era = year - era * 400
    month_adj = month - 3 if month > 2 else month + 9
    day_of_year = (153 * month_adj + 2) // 5 + day - 1
    day_of_era = year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year
    return era * 146097 + day_of_era


def date_dayofyear(value: object) -> int:
    year, month, day = _parse_date_parts(value)
    days_before = _DAYS_BEFORE_MONTH_LEAP if _is_leap_year(year) else _DAYS_BEFORE_MONTH_COMMON
    return days_before[month - 1] + day


def date_dayofweek(value: object) -> int:
    if hasattr(value, "dayofweek"):
        return int(value.dayofweek)
    # 1970-01-01 is Thursday, i.e. 3 with Monday=0.
    return int((date_ordinal(value) + 3) % 7)


def date_day_name(value: object) -> str:
    return _DAY_NAMES[date_dayofweek(value)]


def date_feature_parts(value: object) -> dict[str, float]:
    year, month, _day = _parse_date_parts(value)
    dayofyear = date_dayofyear(value)
    return {
        "year": float(year),
        "month": float(month),
        "quarter": float((month - 1) // 3 + 1),
        "dayofweek": float(date_dayofweek(value)),
        "dayofyear": float(dayofyear),
        "weekofyear": float((dayofyear - 1) // 7 + 1),
    }


def date_ordinals(values: pd.Series) -> pd.Series:
    return values.map(date_ordinal).astype("int64")


def load_frame(path: str | bytes | "os.PathLike[str]") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {REGION_COL, DATE_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df[DATE_COL] = df[DATE_COL].astype(str)
    df[REGION_COL] = df[REGION_COL].astype(str)
    return df


def infer_weather_columns(train_df: pd.DataFrame, test_df: pd.DataFrame | None = None) -> list[str]:
    excluded = {REGION_COL, DATE_COL, TARGET_COL}
    candidates = [col for col in train_df.columns if col not in excluded]
    if test_df is not None:
        candidates = [col for col in candidates if col in test_df.columns]
    numeric_cols: list[str] = []
    for col in candidates:
        converted = pd.to_numeric(train_df[col], errors="coerce")
        if converted.notna().any():
            numeric_cols.append(col)
    if not numeric_cols:
        raise ValueError("No numeric meteorological feature columns were found.")
    return numeric_cols


def coerce_weather_numeric(df: pd.DataFrame, numeric_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def detect_precipitation_columns(cols: Iterable[str]) -> list[str]:
    keys = ("precip", "prectot", "prec", "prcp", "rain", "ppt")
    return [col for col in cols if any(key in col.lower() for key in keys)]


def _last_valid(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    return float(valid[-1]) if valid.size else np.nan


def _first_valid(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    return float(valid[0]) if valid.size else np.nan


def _slope(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return 0.0
    y = values[mask].astype(float)
    x = np.arange(values.size, dtype=float)[mask]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denom)


def _nan_stat(values: np.ndarray, func: str) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    if func == "mean":
        return float(np.mean(finite))
    if func == "std":
        return float(np.std(finite))
    if func == "min":
        return float(np.min(finite))
    if func == "max":
        return float(np.max(finite))
    if func == "median":
        return float(np.median(finite))
    if func == "q10":
        return float(np.quantile(finite, 0.10))
    if func == "q90":
        return float(np.quantile(finite, 0.90))
    raise ValueError(f"Unknown stat: {func}")


def _max_consecutive(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for value in mask:
        if bool(value):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def _days_since_wet(values: np.ndarray, threshold: float) -> float:
    finite = np.isfinite(values)
    wet = finite & (values > threshold)
    if not wet.any():
        return float(values.size)
    return float(values.size - 1 - np.flatnonzero(wet)[-1])


def make_window_features(
    window_df: pd.DataFrame,
    numeric_cols: list[str],
    region_id: str,
    end_date: pd.Timestamp,
    config: FeatureConfig,
) -> dict[str, float | str]:
    if len(window_df) < config.input_days:
        raise ValueError(f"Expected at least {config.input_days} input days, got {len(window_df)}")

    date_parts = date_feature_parts(end_date)
    features: dict[str, float | str] = {
        "region_id_raw": str(region_id),
        "end_year": date_parts["year"],
        "end_month": date_parts["month"],
        "end_quarter": date_parts["quarter"],
        "end_dayofweek": date_parts["dayofweek"],
        "end_dayofyear": date_parts["dayofyear"],
        "end_weekofyear": date_parts["weekofyear"],
    }
    features["end_month_sin"] = math.sin(2.0 * math.pi * date_parts["month"] / 12.0)
    features["end_month_cos"] = math.cos(2.0 * math.pi * date_parts["month"] / 12.0)
    features["end_doy_sin"] = math.sin(2.0 * math.pi * date_parts["dayofyear"] / 366.0)
    features["end_doy_cos"] = math.cos(2.0 * math.pi * date_parts["dayofyear"] / 366.0)

    precip_cols = set(detect_precipitation_columns(numeric_cols))
    for col in numeric_cols:
        safe_col = safe_name(col)
        full_values = window_df[col].to_numpy(dtype=float)
        full_mean = _nan_stat(full_values, "mean")
        full_last = _last_valid(full_values)

        for days in config.windows:
            values = full_values[-days:]
            prefix = f"{safe_col}_{days}d"
            mean_value = _nan_stat(values, "mean")
            features[f"{prefix}_mean"] = mean_value
            features[f"{prefix}_std"] = _nan_stat(values, "std")
            features[f"{prefix}_min"] = _nan_stat(values, "min")
            features[f"{prefix}_max"] = _nan_stat(values, "max")
            features[f"{prefix}_median"] = _nan_stat(values, "median")
            features[f"{prefix}_q10"] = _nan_stat(values, "q10")
            features[f"{prefix}_q90"] = _nan_stat(values, "q90")
            features[f"{prefix}_first"] = _first_valid(values)
            features[f"{prefix}_last"] = _last_valid(values)
            features[f"{prefix}_delta"] = features[f"{prefix}_last"] - features[f"{prefix}_first"]
            features[f"{prefix}_slope"] = _slope(values)
            features[f"{prefix}_missing_frac"] = float(np.mean(~np.isfinite(values)))
            features[f"{prefix}_last_minus_mean"] = full_last - mean_value

            if days != config.input_days:
                features[f"{prefix}_mean_minus_91d"] = mean_value - full_mean

            if col in precip_cols:
                finite = values[np.isfinite(values)]
                dry = np.isfinite(values) & (values <= config.dry_threshold)
                wet = np.isfinite(values) & (values > config.dry_threshold)
                features[f"{prefix}_sum"] = float(np.sum(finite)) if finite.size else np.nan
                features[f"{prefix}_dry_days"] = float(np.sum(dry))
                features[f"{prefix}_wet_days"] = float(np.sum(wet))
                features[f"{prefix}_max_dry_run"] = float(_max_consecutive(dry))
                features[f"{prefix}_days_since_wet"] = _days_since_wet(values, config.dry_threshold)

    return features


def _score_day_targets(score: np.ndarray, anchor: int, config: FeatureConfig) -> np.ndarray:
    offsets = [(h + 1) * config.horizon_step_days for h in range(config.horizons)]
    indices = [anchor + offset for offset in offsets]
    if indices[-1] >= score.size:
        return np.full(config.horizons, np.nan)
    return score[indices].astype(float)


def _next_non_null_targets(score: np.ndarray, anchor: int, config: FeatureConfig) -> np.ndarray:
    future = np.flatnonzero(np.isfinite(score) & (np.arange(score.size) > anchor))
    future = future[: config.horizons]
    if future.size < config.horizons:
        return np.full(config.horizons, np.nan)
    return score[future].astype(float)


def build_training_set(
    train_df: pd.DataFrame,
    numeric_cols: list[str],
    config: FeatureConfig,
    max_samples_per_region: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if TARGET_COL not in train_df.columns:
        raise ValueError("train.csv must contain a score column.")

    feature_rows: list[dict[str, float | str]] = []
    target_rows: list[dict[str, float]] = []
    meta_rows: list[dict[str, object]] = []

    sorted_df = train_df.reset_index(drop=True)
    for region_id, group in sorted_df.groupby(REGION_COL, sort=False):
        group = group.reset_index(drop=True)
        score = pd.to_numeric(group[TARGET_COL], errors="coerce").to_numpy(dtype=float)

        if config.anchor_mode == "score_days":
            anchors = np.flatnonzero(np.isfinite(score))
        elif config.anchor_mode == "all_days":
            start = config.input_days - 1
            stop = max(start, len(group) - config.horizon_step_days * config.horizons)
            anchors = np.arange(start, stop, max(1, config.anchor_stride), dtype=int)
        else:
            raise ValueError("anchor_mode must be either 'score_days' or 'all_days'.")

        anchors = anchors[anchors >= config.input_days - 1]
        if max_samples_per_region is not None and anchors.size > max_samples_per_region:
            anchors = anchors[-max_samples_per_region:]

        for anchor in anchors:
            if config.anchor_mode == "score_days":
                targets = _score_day_targets(score, int(anchor), config)
            else:
                targets = _next_non_null_targets(score, int(anchor), config)
            if not np.isfinite(targets).all():
                continue

            start = int(anchor) - config.input_days + 1
            end = int(anchor) + 1
            window = group.iloc[start:end]
            end_date = group.loc[int(anchor), DATE_COL]
            feature_rows.append(make_window_features(window, numeric_cols, region_id, end_date, config))
            target_rows.append({f"week_{i + 1}": float(targets[i]) for i in range(config.horizons)})
            meta_rows.append(
                {
                    REGION_COL: str(region_id),
                    "end_date": end_date,
                    DATE_ORDINAL_COL: int(date_ordinal(end_date)),
                    "anchor_index": int(anchor),
                }
            )

    if not feature_rows:
        raise ValueError("No training samples were produced. Check score cadence and input window settings.")

    return pd.DataFrame(feature_rows), pd.DataFrame(target_rows), pd.DataFrame(meta_rows)


def build_test_set(
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: list[dict[str, float | str]] = []
    meta_rows: list[dict[str, object]] = []

    sorted_df = test_df.reset_index(drop=True)
    for region_id, group in sorted_df.groupby(REGION_COL, sort=False):
        group = group.reset_index(drop=True)
        if len(group) < config.input_days:
            raise ValueError(f"Region {region_id} has only {len(group)} test rows; expected {config.input_days}.")
        window = group.iloc[-config.input_days :]
        end_date = window[DATE_COL].iloc[-1]
        feature_rows.append(make_window_features(window, numeric_cols, region_id, end_date, config))
        meta_rows.append(
            {
                REGION_COL: str(region_id),
                "end_date": end_date,
                DATE_ORDINAL_COL: int(date_ordinal(end_date)),
                "test_rows": int(len(group)),
            }
        )

    return pd.DataFrame(feature_rows), pd.DataFrame(meta_rows)
