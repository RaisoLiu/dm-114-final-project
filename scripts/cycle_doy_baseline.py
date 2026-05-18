#!/usr/bin/env python3
"""Per-region cycle-aware DOY baseline.

For each test anchor, for each horizon h ∈ {1..5}:
  target_date = test_end_date + 7*h days
  target_DOY = day_of_year(target_date) mod 366
  prediction = mean of observed train scores at this region with DOY ≈ target_DOY (±tolerance)

No model training. Pure historical averaging.

Hypothesis: synthetic multi-year data is approximately cyclic on DOY → matching DOY across
years is the strongest persistence signal we have. ext150's pb30 uses DOY-bucketed features
but mixes them with weather; this baseline isolates the pure cycle signal.

Output: submissions/submission_cycle_doy_baseline.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drought.features import DATE_COL, REGION_COL, TARGET_COL, date_dayofyear, date_ordinal  # noqa: E402

DOY_TOLERANCE = 7  # ±days
GLOBAL_MEAN_FALLBACK = 1.2334


def main() -> int:
    data_dir = PROJECT_ROOT / "data"
    print("Loading data ...")
    train = pd.read_csv(data_dir / "train.csv", usecols=[REGION_COL, DATE_COL, TARGET_COL])
    test = pd.read_csv(data_dir / "test.csv", usecols=[REGION_COL, DATE_COL])
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    region_order = sample[REGION_COL].astype(str).tolist()

    # For each region, precompute (DOY, score) for observed scores
    print("Precomputing per-region (DOY, score) for observed scores ...")
    region_to_doy_scores: dict[str, list[tuple[int, float]]] = {}
    region_to_global_mean: dict[str, float] = {}
    for region_id, group in train.groupby(REGION_COL, sort=False):
        region = str(region_id)
        scores_with_dates = []
        score_arr = pd.to_numeric(group[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
        dates = group[DATE_COL].tolist()
        for d, s in zip(dates, score_arr):
            if np.isfinite(s):
                doy = date_dayofyear(d)
                scores_with_dates.append((doy, float(s)))
        region_to_doy_scores[region] = scores_with_dates
        if scores_with_dates:
            region_to_global_mean[region] = float(np.mean([s for _, s in scores_with_dates]))
        else:
            region_to_global_mean[region] = GLOBAL_MEAN_FALLBACK

    print(f"  {len(region_to_doy_scores)} regions; sample region first 3 (DOY, score): "
          f"{region_to_doy_scores[region_order[0]][:3]}")

    print("Building predictions for each test anchor (region) × 5 horizons ...")
    rows = []
    test_end_dates = test.groupby(REGION_COL, sort=False)[DATE_COL].agg("last").to_dict()
    for region in region_order:
        test_end = str(test_end_dates[region])
        test_end_doy = date_dayofyear(test_end)
        doy_scores = region_to_doy_scores.get(region, [])
        fallback = region_to_global_mean.get(region, GLOBAL_MEAN_FALLBACK)
        preds = []
        for h in range(1, 6):
            # Forecast 7*h days ahead
            target_doy = ((test_end_doy + 7 * h - 1) % 366) + 1
            # Match historical DOYs ± tolerance
            matches = [
                s for doy, s in doy_scores
                if abs((doy - target_doy + 366) % 366) <= DOY_TOLERANCE
                or abs((target_doy - doy + 366) % 366) <= DOY_TOLERANCE
            ]
            if len(matches) >= 5:
                pred = float(np.mean(matches))
            else:
                pred = fallback
            preds.append(pred)
        rows.append([region] + preds)

    out = pd.DataFrame(rows, columns=[REGION_COL, "pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"])
    out[REGION_COL] = out[REGION_COL].astype(str)
    out_path = PROJECT_ROOT / "submissions" / "submission_cycle_doy_baseline.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[info] wrote {out_path}")
    print(f"[info] overall mean: {out.iloc[:, 1:].values.mean():.4f}")
    for c in ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]:
        print(f"  {c}: mean={out[c].mean():.4f}  std={out[c].std():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
