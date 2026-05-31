#!/usr/bin/env python3
"""Track 2.5 — LGBM with multi-year phase features + 5-yr score lags.

Key insight from Track 2 FFT EDA: synth scores have dominant 5-year (1825 d) period
in 2,245 / 2,248 regions. ext150 only uses up to 208-week lags (4 years), missing the
dominant 5-year cycle. This script adds:
  - score lag features at multi-year periods (260, 365, 520 weeks)
  - sin/cos phase features for dominant periods (1825d, 1368d, 2737d, 365d)
  - Standard weather window summary stats

The model is trained globally on all 2,248 regions (data is NOT zero-shot —
test regions are the same as train regions, just extended in time).

Output: submissions/_v18_track25_lgbm_multiyear.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent

WEATHER_COLS = ['prec', 'surf_pre', 'humidity', 'tmp', 'dp_tmp', 'wb_tmp',
                'tmp_max', 'tmp_min', 'tmp_range', 'surf_tmp',
                'wind', 'wind_max', 'wind_min', 'wind_range']
PRED_COLS = [f'pred_week{i+1}' for i in range(5)]
WINDOW = 91

# Window stats: (mean, std, min, max, sum) per channel over various windows
ROLL_WINDOWS = [7, 14, 28, 56, 91]
SCORE_LAGS_DAYS = [7, 14, 21, 28, 35, 42, 56, 84, 168, 364, 728, 1092, 1456, 1820, 2548, 3640]
# i.e. weeks: 1, 2, 3, 4, 5, 6, 8, 12, 24, 52, 104, 156, 208, 260 (5yr), 364 (7yr), 520 (10yr)
MULTI_YEAR_PERIODS = [1825, 1368, 2737, 365, 730]  # days


def build_anchor_features(arr_w: np.ndarray, arr_s: np.ndarray,
                           anchor_t: int) -> dict:
    """For one anchor time index t in one region, compute all features."""
    feats = {}
    # Weather window stats
    for win in ROLL_WINDOWS:
        if anchor_t + 1 - win < 0:
            continue
        chunk = arr_w[anchor_t + 1 - win : anchor_t + 1]  # (win, 14)
        for ci, col in enumerate(WEATHER_COLS):
            x = chunk[:, ci]
            feats[f'{col}_w{win}_mean'] = float(x.mean())
            feats[f'{col}_w{win}_std'] = float(x.std())
            feats[f'{col}_w{win}_max'] = float(x.max())
            feats[f'{col}_w{win}_min'] = float(x.min())
            feats[f'{col}_w{win}_sum'] = float(x.sum())

    # Score lags (use most recent non-null score at offset lag)
    for lag in SCORE_LAGS_DAYS:
        t = anchor_t - lag
        if t < 0 or np.isnan(arr_s[t]):
            # search backward to nearest non-null within ±3 days
            found = None
            for delta in range(0, 8):
                if t - delta >= 0 and not np.isnan(arr_s[t - delta]):
                    found = arr_s[t - delta]; break
                if t + delta < len(arr_s) and t + delta <= anchor_t and not np.isnan(arr_s[t + delta]):
                    found = arr_s[t + delta]; break
            feats[f'score_lag_{lag}d'] = float(found) if found is not None else -1.0
        else:
            feats[f'score_lag_{lag}d'] = float(arr_s[t])

    # Multi-year phase features (using anchor_t as day index)
    for period in MULTI_YEAR_PERIODS:
        feats[f'phase_sin_{period}d'] = float(np.sin(2 * np.pi * anchor_t / period))
        feats[f'phase_cos_{period}d'] = float(np.cos(2 * np.pi * anchor_t / period))

    # Anchor-position metadata (within region)
    feats['anchor_t'] = float(anchor_t)

    return feats


def build_train_features(train_df: pd.DataFrame, max_anchors_per_region: int | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Vectorized feature build for all train anchors."""
    print(f"Building train features from {len(train_df)} rows...")
    train_df = train_df.sort_values(['region_id', 'date']).reset_index(drop=True)
    regions = train_df['region_id'].unique().tolist()
    n_per_region = train_df.groupby('region_id').size().iloc[0]
    n_regions = len(regions)
    w_all = train_df[WEATHER_COLS].values.astype(np.float32).reshape(n_regions, n_per_region, len(WEATHER_COLS))
    s_all = train_df['score'].values.astype(np.float32).reshape(n_regions, n_per_region)

    rows = []
    targets = []
    region_idx = []
    region_to_idx = {r: i for i, r in enumerate(regions)}
    last_t_per_region = []  # for test anchor reference

    t0 = time.time()
    for r_i, rid in enumerate(regions):
        s = s_all[r_i]
        w = w_all[r_i]
        anchor_mask = ~np.isnan(s)
        valid_anchors = np.where(anchor_mask)[0]
        valid_anchors = valid_anchors[(valid_anchors >= WINDOW - 1) & (valid_anchors <= n_per_region - 1 - 35)]
        if max_anchors_per_region and len(valid_anchors) > max_anchors_per_region:
            step = len(valid_anchors) // max_anchors_per_region
            valid_anchors = valid_anchors[::step][:max_anchors_per_region]
        for t in valid_anchors:
            future = s[t + 7 * np.arange(1, 6)]
            if np.isnan(future).any():
                continue
            feats = build_anchor_features(w, s, t)
            feats['region_idx'] = r_i
            rows.append(feats)
            targets.append(future)
            region_idx.append(r_i)
        if (r_i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (n_regions - r_i - 1) / (r_i + 1)
            print(f"  region {r_i+1}/{n_regions}  rows={len(rows)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    print(f"  built {len(rows)} feature rows in {time.time()-t0:.0f}s")
    X = pd.DataFrame(rows)
    y = np.stack(targets).astype(np.float32)
    region_idx = np.array(region_idx, dtype=np.int64)
    return X, y, region_idx, regions, w_all, s_all


def build_test_features(test_df: pd.DataFrame, w_all_train: np.ndarray,
                         s_all_train: np.ndarray, regions: list) -> tuple[pd.DataFrame, list]:
    """For each test region, build features at anchor = (last day of train) + offset to test_end-1.

    Actually test gives us 91 days of weather AFTER train ends. We need to predict
    scores at 7,14,21,28,35 days after test's last day. The "anchor" feature window
    is the LAST 91 days observable, which is test's full window. But score lags must
    come from train (test has no scores).
    """
    print(f"Building test features...")
    test_df = test_df.sort_values(['region_id', 'date']).reset_index(drop=True)
    test_regions = test_df['region_id'].unique().tolist()
    n_test_regions = len(test_regions)
    test_days_per_region = test_df.groupby('region_id').size().iloc[0]
    assert test_days_per_region == WINDOW, f"got {test_days_per_region} days, expected {WINDOW}"
    region_to_idx = {r: i for i, r in enumerate(regions)}
    w_test = test_df[WEATHER_COLS].values.astype(np.float32).reshape(n_test_regions, test_days_per_region, len(WEATHER_COLS))

    rows = []
    for r_i, rid in enumerate(test_regions):
        train_r_idx = region_to_idx[rid]
        # Train weather + scores ended at last index of train (day n_per_region - 1)
        train_w = w_all_train[train_r_idx]  # (5480, 14)
        train_s = s_all_train[train_r_idx]  # (5480,)
        # Concatenate train + test weather: anchor at end of test
        full_w = np.concatenate([train_w, w_test[r_i]], axis=0)  # (5480+91, 14)
        # Score history: train scores only (test has none); pad NaN for test days
        full_s = np.concatenate([train_s, np.full(test_days_per_region, np.nan, dtype=np.float32)])
        anchor_t = len(full_w) - 1  # last test day; predictions are at +7,+14,...,+35
        feats = build_anchor_features(full_w, full_s, anchor_t)
        feats['region_idx'] = train_r_idx
        rows.append(feats)
    print(f"  built {len(rows)} test feature rows")
    X = pd.DataFrame(rows)
    return X, test_regions


def train_lgbm_per_horizon(X_tr, y_tr, region_idx, X_te, n_estimators=2000, seeds=(42, 1234, 7777)):
    """Train one LGBM per horizon, multi-seed avg. Returns test predictions (n_test, 5)."""
    print(f"Training LGBM: {X_tr.shape} train, {len(seeds)} seeds × 5 horizons")
    # Validation split: random 10% within each region
    np.random.seed(0)
    mask = np.random.rand(len(X_tr)) < 0.9
    X_tr_split = X_tr.iloc[mask].reset_index(drop=True)
    y_tr_split = y_tr[mask]
    X_va_split = X_tr.iloc[~mask].reset_index(drop=True)
    y_va_split = y_tr[~mask]
    print(f"  train {len(X_tr_split)} val {len(X_va_split)}")

    n_test = len(X_te)
    test_preds = np.zeros((n_test, 5), dtype=np.float32)
    val_maes = []
    for h in range(5):
        h_preds = np.zeros(n_test, dtype=np.float32)
        for seed in seeds:
            params = dict(
                objective='regression_l1',
                metric='mae',
                num_leaves=255,
                min_data_in_leaf=200,
                learning_rate=0.03,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbosity=-1,
                seed=seed,
            )
            d_tr = lgb.Dataset(X_tr_split, label=y_tr_split[:, h])
            d_va = lgb.Dataset(X_va_split, label=y_va_split[:, h], reference=d_tr)
            model = lgb.train(params, d_tr, num_boost_round=n_estimators,
                              valid_sets=[d_va], callbacks=[lgb.early_stopping(100)])
            h_preds += model.predict(X_te, num_iteration=model.best_iteration) / len(seeds)
            val_pred = model.predict(X_va_split, num_iteration=model.best_iteration)
            val_mae = float(np.abs(val_pred - y_va_split[:, h]).mean())
        val_maes.append(val_mae)
        test_preds[:, h] = h_preds
        print(f"  h{h+1} val_mae (last seed): {val_mae:.4f}")
    print(f"  per-horizon val MAE: {val_maes}, mean: {np.mean(val_maes):.4f}")
    return test_preds, val_maes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-anchors", type=int, default=400, help="max anchors per region")
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs='*', default=[42, 1234, 7777])
    ap.add_argument("--output", default="submissions/_v18_track25_lgbm_multiyear.csv")
    args = ap.parse_args()

    print("Loading train.csv...")
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    print("Loading test.csv...")
    test_df = pd.read_csv(ROOT / "data" / "test.csv")

    X_tr, y_tr, region_idx, regions, w_all_train, s_all_train = \
        build_train_features(train_df, max_anchors_per_region=args.max_anchors)
    X_te, test_regions = build_test_features(test_df, w_all_train, s_all_train, regions)
    print(f"Feature count: {X_tr.shape[1]}")

    test_preds, val_maes = train_lgbm_per_horizon(X_tr, y_tr, region_idx, X_te,
                                                    n_estimators=args.n_estimators,
                                                    seeds=args.seeds)
    test_preds = np.clip(test_preds, 0.0, 5.0)
    out_df = pd.DataFrame({'region_id': test_regions})
    for i, col in enumerate(PRED_COLS):
        out_df[col] = test_preds[:, i]
    out_path = ROOT / args.output
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved candidate: {out_path}")
    print(f"  preds mean={test_preds.mean():.4f} std={test_preds.std():.4f}")


if __name__ == "__main__":
    main()
