#!/usr/bin/env python3
"""Plan v6 E3 calibration: per-DOY quantile mapping.

Maps a candidate submission's per-horizon distribution to match the
adversarial-val-slice target CDF, bucketed by DOY (28-day buckets, ~13 buckets).

Inputs:
  --candidate PATH                : candidate submission CSV
  --val-predictions PATH          : training-time val predictions CSV (with y_true and pred_final_calibrated)
  --output PATH                   : output CSV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]
DOY_BUCKET_DAYS = 28


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--val-predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sample = pd.read_csv(DATA / "sample_submission.csv")
    sample["region_id"] = sample["region_id"].astype(str)
    region_order = sample["region_id"].tolist()

    # Get per-region test-anchor DOY (from test.csv last row)
    test = pd.read_csv(DATA / "test.csv", usecols=["region_id", "date"])
    test["region_id"] = test["region_id"].astype(str)
    test_end_dates = test.groupby("region_id", sort=False)["date"].last()

    def parse_doy(date_str: str) -> int:
        y, m, d = [int(x) for x in date_str.split("-", 2)]
        days_before = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        return days_before[m - 1] + d

    test_doy_by_region = {r: parse_doy(test_end_dates[r]) for r in region_order}

    # Bucket test anchors by DOY
    def doy_bucket(doy: int) -> int:
        return doy // DOY_BUCKET_DAYS

    test_bucket = np.array([doy_bucket(test_doy_by_region[r]) for r in region_order])
    print(f"[info] test DOY buckets: {np.unique(test_bucket)}")

    # Load val predictions
    vp = pd.read_csv(args.val_predictions)
    print(f"[info] val predictions: {len(vp):,} rows, columns: {list(vp.columns)}")
    pred_col = "pred_final_calibrated" if "pred_final_calibrated" in vp.columns else (
        "pred_raw" if "pred_raw" in vp.columns else "prediction"
    )
    if pred_col not in vp.columns or "y_true" not in vp.columns:
        print(f"[error] val predictions must have y_true and one of: pred_final_calibrated/pred_raw/prediction")
        return 1

    # Need per-row DOY for val anchors. Look up from train.csv via row_index
    # But row_index in val_predictions is local index. Need region+anchor mapping.
    # Use train.csv: index within region's group corresponds to anchor_index.
    if "anchor_index" in vp.columns and "region_id" in vp.columns:
        train = pd.read_csv(DATA / "train.csv", usecols=["region_id", "date"])
        train["region_id"] = train["region_id"].astype(str)
        date_lookup: dict[tuple[str, int], str] = {}
        for r, g in train.groupby("region_id", sort=False):
            r_str = str(r)
            for i, date_val in enumerate(g["date"].tolist()):
                date_lookup[(r_str, i)] = date_val
        vp["doy"] = vp.apply(
            lambda row: parse_doy(date_lookup.get((str(row["region_id"]), int(row["anchor_index"])), "0000-01-01")),
            axis=1,
        )
    else:
        # Fall back: compute DOY from anchor_index + region last_usable
        print("[warn] val_predictions has no anchor_index column; skipping per-DOY calibration")
        return 1

    vp["bucket"] = (vp["doy"] // DOY_BUCKET_DAYS).astype(int)
    print(f"[info] val DOY buckets: {sorted(vp['bucket'].unique())}")

    # Per (bucket, horizon): empirical CDF of y_true and of pred
    cand = pd.read_csv(args.candidate)
    cand["region_id"] = cand["region_id"].astype(str)
    cand = cand.set_index("region_id").reindex(region_order).reset_index()
    cand_arr = cand[PRED_COLS].to_numpy(dtype=np.float64)

    out = cand_arr.copy()
    n_buckets = 13
    for h_idx, h_col in enumerate(PRED_COLS):
        horizon_num = h_idx + 1
        vp_h = vp[vp["horizon"] == horizon_num] if "horizon" in vp.columns else vp
        for b in range(n_buckets):
            test_mask = test_bucket == b
            val_mask = vp_h["bucket"] == b
            if test_mask.sum() == 0 or val_mask.sum() < 30:
                continue
            val_pred_vals = vp_h.loc[val_mask, pred_col].to_numpy()
            val_true_vals = vp_h.loc[val_mask, "y_true"].to_numpy()
            # Per-bucket mapping: sort val_pred ↔ val_true to build percentile map
            sort_idx = np.argsort(val_pred_vals)
            sorted_pred = val_pred_vals[sort_idx]
            sorted_true = val_true_vals[sort_idx]
            # Map each candidate test prediction to its percentile in val_pred CDF,
            # then look up val_true at same percentile.
            test_preds = cand_arr[test_mask, h_idx]
            # Use np.interp on sorted arrays
            cdf = np.searchsorted(sorted_pred, test_preds, side="right") / len(sorted_pred)
            mapped = np.interp(cdf, np.linspace(0, 1, len(sorted_true)), np.sort(sorted_true))
            out[test_mask, h_idx] = mapped

    out = np.clip(out, 0.0, 5.0)
    df_out = pd.DataFrame(out, columns=PRED_COLS)
    df_out.insert(0, "region_id", region_order)
    df_out.to_csv(args.output, index=False)
    print(f"[info] wrote {args.output}  mean={out.mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
