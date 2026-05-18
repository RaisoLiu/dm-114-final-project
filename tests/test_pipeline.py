from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drought.features import FeatureConfig, build_test_set, build_training_set, coerce_weather_numeric
from train_gap_model import WEATHER_COLS, build_weather_stats, date_features
from train_predict import choose_anchor_mode, format_submission


def make_single_region_frame(days: int = 150) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    rows = []
    for idx, date in enumerate(dates):
        score = float(idx) if idx % 7 == 0 else np.nan
        rows.append(
            {
                "region_id": "R1",
                "date": date,
                "rainfall": float(idx),
                "temp_mean": float(1000 + idx),
                "score": score,
            }
        )
    return pd.DataFrame(rows)


def test_training_windows_use_only_91_past_days_and_align_weekly_targets() -> None:
    config = FeatureConfig()
    train = coerce_weather_numeric(make_single_region_frame(), ["rainfall", "temp_mean"])

    features, targets, meta = build_training_set(train, ["rainfall", "temp_mean"], config)

    first_anchor = int(meta.loc[0, "anchor_index"])
    assert first_anchor == 91
    assert meta.loc[0, "end_date"] == pd.Timestamp("2020-04-01")

    # The 91-day window for anchor index 91 is rows 1..91 inclusive.
    assert features.loc[0, "rainfall_91d_first"] == 1.0
    assert features.loc[0, "rainfall_91d_last"] == 91.0
    assert features.loc[0, "rainfall_91d_max"] == 91.0

    # Targets are exactly the next five weekly score rows, not NaN rows and not weather future leakage.
    assert targets.loc[0].tolist() == [98.0, 105.0, 112.0, 119.0, 126.0]
    assert not any("score" in column.lower() for column in features.columns)


def test_test_windows_take_last_91_rows_per_region() -> None:
    config = FeatureConfig()
    test = make_single_region_frame().drop(columns=["score"])
    test = coerce_weather_numeric(test, ["rainfall", "temp_mean"])

    features, meta = build_test_set(test, ["rainfall", "temp_mean"], config)

    assert len(features) == 1
    assert meta.loc[0, "end_date"] == pd.Timestamp("2020-05-29")
    assert features.loc[0, "rainfall_91d_first"] == 59.0
    assert features.loc[0, "rainfall_91d_last"] == 149.0


def test_submission_format_preserves_sample_columns_and_region_order(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    output_path = tmp_path / "submission.csv"
    sample = pd.DataFrame(
        {
            "region_id": ["R2", "R1"],
            "pred_week1": [0, 0],
            "pred_week2": [0, 0],
            "pred_week3": [0, 0],
            "pred_week4": [0, 0],
            "pred_week5": [0, 0],
        }
    )
    sample.to_csv(sample_path, index=False)
    test_meta = pd.DataFrame({"region_id": ["R1", "R2"]})
    predictions = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=float)

    format_submission(sample_path, test_meta, predictions, output_path, 0.0, 5.0)

    submission = pd.read_csv(output_path)
    assert submission.columns.tolist() == sample.columns.tolist()
    assert submission["region_id"].tolist() == ["R2", "R1"]
    assert submission.loc[0, "pred_week1"] == 5.0
    assert submission.loc[1, "pred_week5"] == 5.0


def test_auto_anchor_mode_uses_all_days_when_test_end_weekday_differs() -> None:
    train = make_single_region_frame()
    aligned_test = train.iloc[8:99].drop(columns=["score"]).copy()
    shifted_test = train.iloc[9:100].drop(columns=["score"]).copy()

    assert choose_anchor_mode(train, aligned_test, "auto") == "score_days"
    assert choose_anchor_mode(train, shifted_test, "auto") == "all_days"
    assert choose_anchor_mode(train, shifted_test, "score_days") == "score_days"


def test_gap_model_month_features_handle_synthetic_year_width() -> None:
    rows = []
    for month in range(1, 13):
        row = {"region_id": "R1", "date": f"58061-{month:02d}-15"}
        for col in WEATHER_COLS:
            row[col] = float(month)
        rows.append(row)

    stats = build_weather_stats(pd.DataFrame(rows))

    assert stats["month_mean"][1, 0] == 1.0
    assert stats["month_mean"][7, 0] == 7.0
    assert stats["month_mean"][12, 0] == 12.0
    assert date_features("58061-12-15")[0] == 12.0
