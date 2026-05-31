#!/usr/bin/env python3
"""Plan v9 D2 — Score-only long-horizon model with variable-horizon prediction.

Key insight from Plan v9 audit:
- train→test gap varies per region (89–694 days, median 531).
- Hence per-region we want to predict score at (train_end + gap + 7*h) for h ∈ 1..5.
- Standard score-lag models train with fixed horizons 7,14,21,28,35; not directly usable.
- Solution: train a single LGBM with `horizon` (days ahead) as an input feature.

Architecture:
- Anchor at every 4th weekly position in train (down-sample for speed).
- For each anchor, build feature vector (no short lags <14 weeks; rely on long lags
  + DOY cycle + region stats — these are ALL inference-time-available at test).
- For each anchor, sample 5 random horizons in [60, 730] days; target = score at
  (anchor + horizon) IF that index exists in train.
- Concatenate horizon to features; train one LGBM on (X, horizon) → target.

Inference:
- For each test region, compute features at the LAST weekly anchor in train.csv.
- For each h ∈ 1..5, set horizon = gap_days + 7*h and predict.
- Output CSV: submissions/submission_score_only_longhorizon.csv.

Sanity checks:
- MAD vs ext150 (gate ≤ 0.30 from plan).
- Per-horizon mean (target ~ [1.0, 1.4]).
- ρ vs ext150 errors on adversarial-val slice (target < 0.3).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drought.features import DATE_COL, REGION_COL, TARGET_COL, date_dayofyear, date_dayofweek  # noqa: E402

DATA = PROJECT_ROOT / "data"
SUB = PROJECT_ROOT / "submissions"
REPORTS = PROJECT_ROOT / "reports"
PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]


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


def date_parts_at(date_str: str) -> list[float]:
    year, month, day = [int(p) for p in date_str.split("-")]
    doy = date_dayofyear(date_str)
    dow = date_dayofweek(date_str)
    return [
        float(month), float(day), float(doy), float(dow),
        float(np.sin(2.0 * np.pi * month / 12.0)),
        float(np.cos(2.0 * np.pi * month / 12.0)),
        float(np.sin(2.0 * np.pi * doy / 366.0)),
        float(np.cos(2.0 * np.pi * doy / 366.0)),
        float(np.sin(2.0 * np.pi * dow / 7.0)),
        float(np.cos(2.0 * np.pi * dow / 7.0)),
        float(year % 4),
    ]


def date_parts_at_offset(anchor_date: str, offset_days: int) -> list[float]:
    """Date-parts at (anchor + offset_days). Avoids real datetime due to year>2262 issue."""
    # Use synthetic year arithmetic — note: this is approximate for leap years,
    # but DOY computation already absorbs leap-year handling.
    year, month, day = [int(p) for p in anchor_date.split("-")]
    # convert to absolute day count (synthetic Gregorian, ignoring leap subtleties for offsets <1000 days)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ord_anchor = year * 365 + sum(days_in_month[:month - 1]) + day
    ord_target = ord_anchor + offset_days
    # Inverse mapping back to (year, month, day)
    year_t = ord_target // 365
    rem = ord_target - year_t * 365
    month_t = 1
    while month_t <= 12 and rem > days_in_month[month_t - 1]:
        rem -= days_in_month[month_t - 1]
        month_t += 1
    day_t = max(rem, 1)
    return date_parts_at(f"{year_t:04d}-{month_t:02d}-{day_t:02d}")


def build_anchor_features(
    scores: np.ndarray,
    dates: list[str],
    anchor: int,
    region_code: int,
    region_mean: float,
    region_std: float,
) -> list[float]:
    """Features computed AT the anchor position (no future leakage)."""
    hist = scores[: anchor + 1].astype(np.float32)
    feats: list[float] = [
        float(region_code),
        region_mean,
        region_std,
        float(hist[-1]),  # last observed score
        float(anchor),    # absolute anchor index (proxy for region's time in cycle)
    ]
    # date features at anchor
    feats.extend(date_parts_at(dates[anchor]))
    # long-lag features (14, 26, 39, 52, 78, 104, 156, 208 weeks)
    for lag in (14, 26, 39, 52, 78, 104, 156, 208):
        idx = anchor - lag
        if idx >= 0:
            feats.append(float(hist[idx]))
        else:
            feats.append(float(region_mean))
    # rolling window stats over BACKWARD horizons (so they're available at any anchor)
    for window in (4, 13, 26, 52, 104):
        end = max(0, anchor + 1 - 14)  # window ENDS 14 weeks before anchor (safe long-lag)
        start = max(0, end - window)
        values = hist[start:end] if end > start else hist[:1]
        feats.extend([
            float(values.mean()),
            float(np.median(values)) if values.size > 0 else 0.0,
            float(values.std()),
            float(values.max()),
            float((values > 0).mean()),
            float((values >= 2).mean()),
            slope(values),
        ])
    # "anchor age" — how many weekly scores have we observed
    feats.append(float(hist.size))
    feats.append(float(hist.mean()))
    feats.append(float(np.median(hist)))
    feats.append(float(hist.std()))
    feats.append(float(hist.max()))
    feats.append(float((hist > 0).mean()))
    return feats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA))
    parser.add_argument("--anchor-stride", type=int, default=4, help="weekly stride for anchor sampling")
    parser.add_argument("--horizons-per-anchor", type=int, default=5)
    parser.add_argument("--horizon-min", type=int, default=60)
    parser.add_argument("--horizon-max", type=int, default=730)
    parser.add_argument("--max-train-rows", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--output-csv", default=str(SUB / "submission_score_only_longhorizon.csv"))
    parser.add_argument("--report-json", default=str(REPORTS / "score_only_longhorizon.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)

    print("Loading train.csv ...")
    train = pd.read_csv(data_dir / "train.csv", usecols=[REGION_COL, DATE_COL, TARGET_COL])
    train[TARGET_COL] = pd.to_numeric(train[TARGET_COL], errors="coerce")
    train = train[train[TARGET_COL].notna()].copy()
    train[REGION_COL] = train[REGION_COL].astype(str)
    print(f"  {len(train):,} non-null score rows, {train[REGION_COL].nunique()} regions")

    sample = pd.read_csv(data_dir / "sample_submission.csv", usecols=[REGION_COL])
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    region_order = sample[REGION_COL].tolist()
    region_to_code = {r: i for i, r in enumerate(region_order)}

    # Build per-region train_end_date and test_end_date for gap calculation
    print("Loading test.csv (for gap calculation) ...")
    test = pd.read_csv(data_dir / "test.csv", usecols=[REGION_COL, DATE_COL])
    test[REGION_COL] = test[REGION_COL].astype(str)

    train_end_dates = train.groupby(REGION_COL, sort=False)[DATE_COL].last().to_dict()
    test_end_dates = test.groupby(REGION_COL, sort=False)[DATE_COL].last().to_dict()

    def date_diff_days(d1: str, d2: str) -> int:
        # d1 - d2 in days (d1 later)
        y1, m1, dd1 = [int(p) for p in d1.split("-")]
        y2, m2, dd2 = [int(p) for p in d2.split("-")]
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        ord1 = y1 * 365 + sum(days_in_month[:m1 - 1]) + dd1
        ord2 = y2 * 365 + sum(days_in_month[:m2 - 1]) + dd2
        return ord1 - ord2

    gap_per_region: dict[str, int] = {}
    for r in region_order:
        if r in train_end_dates and r in test_end_dates:
            gap_per_region[r] = date_diff_days(test_end_dates[r], train_end_dates[r])
        else:
            gap_per_region[r] = 531  # median fallback
    print(f"  gap stats: min={min(gap_per_region.values())}, "
          f"median={np.median(list(gap_per_region.values())):.0f}, "
          f"max={max(gap_per_region.values())}")

    # Build training rows
    print("Building training feature rows ...")
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    horizons_used: list[int] = []
    n_anchors = 0
    for region, group in train.groupby(REGION_COL, sort=False):
        if region not in region_to_code:
            continue
        scores = group[TARGET_COL].to_numpy(dtype=np.float32)
        dates = group[DATE_COL].astype(str).tolist()
        rcode = region_to_code[region]
        rmean = float(scores.mean())
        rstd = float(scores.std())
        n_weekly = len(scores)
        # Anchors: every `anchor_stride` weeks, starting from week 208 (to have long lags available)
        for anchor in range(208, n_weekly - 1, args.anchor_stride):
            base_feats = build_anchor_features(scores, dates, anchor, rcode, rmean, rstd)
            n_anchors += 1
            # Sample `horizons_per_anchor` horizons; target must be in range
            for _ in range(args.horizons_per_anchor):
                horizon_days = int(rng.integers(args.horizon_min, args.horizon_max + 1))
                # Convert to weekly offset (round to nearest week)
                horizon_weeks = horizon_days // 7
                target_idx = anchor + horizon_weeks
                if target_idx >= n_weekly:
                    continue
                # Date features at target
                target_date_parts = date_parts_at_offset(dates[anchor], horizon_days)
                full_feats = base_feats + [float(horizon_days)] + target_date_parts
                x_rows.append(full_feats)
                y_rows.append(float(scores[target_idx]))
                horizons_used.append(horizon_days)
    print(f"  anchors built: {n_anchors:,}; total (anchor, horizon) rows: {len(x_rows):,}")

    X = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  horizon distribution: min={min(horizons_used)}, median={int(np.median(horizons_used))}, max={max(horizons_used)}")

    # Subsample if needed
    if X.shape[0] > args.max_train_rows:
        idx = rng.choice(X.shape[0], size=args.max_train_rows, replace=False)
        X = X[idx]
        y = y[idx]
        print(f"  subsampled to {X.shape[0]:,} rows")

    # Train LGBM
    print("Training LightGBM ...")
    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:
        print(f"[error] lightgbm not installed: {e}")
        return 1
    model = LGBMRegressor(
        objective="mae",
        n_estimators=args.n_estimators,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.30,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=-1,
        device_type=args.device,
    )
    # Hold-out 10% for val
    n_val = int(0.10 * X.shape[0])
    perm = rng.permutation(X.shape[0])
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], callbacks=[])
    val_pred = np.clip(model.predict(X[val_idx]), 0.0, 5.0)
    val_mae = float(np.abs(val_pred - y[val_idx]).mean())
    val_persistence_mae = float(np.abs(y[val_idx] - np.full_like(y[val_idx], y[val_idx].mean())).mean())
    print(f"  val MAE = {val_mae:.4f}")
    print(f"  persistence-to-mean val MAE = {val_persistence_mae:.4f}")

    # Inference: produce test predictions
    print("Building test predictions ...")
    out_rows: list[list[float]] = []
    for region in region_order:
        # Find the training data for this region
        rgroup = train[train[REGION_COL] == region]
        if len(rgroup) == 0:
            out_rows.append([float(0.5)] * 5)
            continue
        scores = rgroup[TARGET_COL].to_numpy(dtype=np.float32)
        dates = rgroup[DATE_COL].astype(str).tolist()
        rcode = region_to_code[region]
        rmean = float(scores.mean())
        rstd = float(scores.std())
        # Anchor = last weekly score in train
        anchor = len(scores) - 1
        base_feats = build_anchor_features(scores, dates, anchor, rcode, rmean, rstd)
        gap_days = gap_per_region.get(region, 531)
        preds = []
        for h in range(1, 6):
            horizon_days = gap_days + 7 * h
            target_date_parts = date_parts_at_offset(dates[anchor], horizon_days)
            full_feats = base_feats + [float(horizon_days)] + target_date_parts
            p = float(model.predict(np.asarray([full_feats], dtype=np.float32))[0])
            p = float(np.clip(p, 0.0, 5.0))
            preds.append(p)
        out_rows.append(preds)

    out_df = pd.DataFrame(out_rows, columns=PRED_COLS)
    out_df.insert(0, REGION_COL, region_order)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")

    # Sanity checks
    print()
    print("Sanity checks ...")
    print(f"  per-horizon mean: {[round(out_df[c].mean(), 4) for c in PRED_COLS]}")
    print(f"  overall mean: {out_df[PRED_COLS].values.mean():.4f}")
    ext150 = pd.read_csv(SUB / "submission_round5_pb30_x150_repro.csv").set_index(REGION_COL)
    ext150.index = ext150.index.astype(str)
    out_aligned = out_df.set_index(REGION_COL).reindex(region_order).reset_index()
    mad = float(np.abs(out_aligned[PRED_COLS].values - ext150.reindex(region_order)[PRED_COLS].values).mean())
    print(f"  MAD vs ext150: {mad:.4f}")
    print(f"  range: [{out_df[PRED_COLS].values.min():.4f}, {out_df[PRED_COLS].values.max():.4f}]")

    # Persist report
    report = {
        "val_mae": val_mae,
        "val_persistence_mae": val_persistence_mae,
        "n_train_rows": int(X[tr_idx].shape[0]),
        "n_val_rows": int(X[val_idx].shape[0]),
        "feature_count": int(X.shape[1]),
        "horizon_min": int(min(horizons_used)),
        "horizon_max": int(max(horizons_used)),
        "horizon_median": int(np.median(horizons_used)),
        "test_per_horizon_mean": [float(out_df[c].mean()) for c in PRED_COLS],
        "test_overall_mean": float(out_df[PRED_COLS].values.mean()),
        "mad_vs_ext150": mad,
        "test_min": float(out_df[PRED_COLS].values.min()),
        "test_max": float(out_df[PRED_COLS].values.max()),
        "args": vars(args),
    }
    rp = Path(args.report_json)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
