#!/usr/bin/env python3
"""Quick hypothesis test: does score(t) ≈ score(t - 1825 days)?

If yes, pure 5-yr lag lookup beats ext150 immediately.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import time

ROOT = Path(__file__).resolve().parent.parent
WEATHER_COLS = ['prec', 'surf_pre', 'humidity', 'tmp', 'dp_tmp', 'wb_tmp',
                'tmp_max', 'tmp_min', 'tmp_range', 'surf_tmp',
                'wind', 'wind_max', 'wind_min', 'wind_range']
PRED_COLS = [f'pred_week{i+1}' for i in range(5)]
WINDOW = 91

print("Loading train.csv...")
t0 = time.time()
train_df = pd.read_csv(ROOT / "data" / "train.csv", usecols=['region_id', 'date', 'score'])
print(f"  load {time.time()-t0:.1f}s")

train_df = train_df.sort_values(['region_id', 'date']).reset_index(drop=True)
regions = train_df['region_id'].unique().tolist()
n_per_region = train_df.groupby('region_id').size().iloc[0]
print(f"  {len(regions)} regions, {n_per_region} days each")

# Reshape into (R, T)
s_all = train_df['score'].values.astype(np.float32).reshape(len(regions), n_per_region)

# Test correlation at various lags
print("\nCorrelation of score(t) with score(t - lag) across regions:")
print(f"{'lag_days':>10s}  {'mean_corr':>10s}  {'med_corr':>10s}  {'frac_>0.7':>10s}  {'frac_>0.9':>10s}")
for lag in [7, 28, 91, 365, 728, 1095, 1368, 1825, 2737]:
    if lag >= n_per_region:
        continue
    corrs = []
    for r in range(len(regions)):
        s = s_all[r]
        s_t = s[lag:]
        s_lag = s[:-lag]
        mask = ~np.isnan(s_t) & ~np.isnan(s_lag)
        if mask.sum() < 50:
            continue
        a = s_t[mask]
        b = s_lag[mask]
        if a.std() < 1e-6 or b.std() < 1e-6:
            continue
        c = np.corrcoef(a, b)[0, 1]
        corrs.append(c)
    corrs = np.array(corrs)
    print(f"  {lag:>10d}  {corrs.mean():>10.4f}  {np.median(corrs):>10.4f}"
          f"  {(corrs > 0.7).mean():>10.4f}  {(corrs > 0.9).mean():>10.4f}")

# Direct MAE test: predict score(t) as score(t - 1825) for various lags
print(f"\n=== Direct MAE: score(t) vs score(t - lag) ===")
for lag in [365, 1825]:
    abs_errs = []
    for r in range(len(regions)):
        s = s_all[r]
        s_t = s[lag:]
        s_lag = s[:-lag]
        mask = ~np.isnan(s_t) & ~np.isnan(s_lag)
        abs_errs.extend(np.abs(s_t[mask] - s_lag[mask]))
    abs_errs = np.array(abs_errs)
    print(f"  lag={lag}d:  n={len(abs_errs)}  MAE={abs_errs.mean():.4f}  median_err={np.median(abs_errs):.4f}")

# Now try predicting test horizons as score(test_date - 1825 + h*7)
# Load test
test_df = pd.read_csv(ROOT / "data" / "test.csv", usecols=['region_id', 'date'])
test_df = test_df.sort_values(['region_id', 'date']).reset_index(drop=True)
test_dates = test_df.groupby('region_id')['date'].agg(['min', 'max']).reset_index()
print(f"\nTest: {len(test_dates)} regions, dates range example: {test_dates['min'].iloc[0]} to {test_dates['max'].iloc[0]}")

# For each test region, find the date 1825 days before each prediction target
# Predictions are at test_max + 7, 14, 21, 28, 35 days
# We look up scores at (test_max + h*7) - 1825 days

def date_to_day_ordinal(date_str):
    """Convert synth date YYYY-MM-DD to day ordinal (treating each year as 365 days)."""
    y, m, d = date_str.split('-')
    y, m, d = int(y), int(m), int(d)
    # Approximate Gregorian arithmetic
    days_before_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    return y * 365 + days_before_month[m-1] + (d - 1)

# Build per-region: date -> score lookup
train_df['day_ord'] = train_df['date'].apply(date_to_day_ordinal)
test_df['day_ord'] = test_df['date'].apply(date_to_day_ordinal)
train_date_score = {}
for rid, g in train_df.dropna(subset=['score']).groupby('region_id'):
    train_date_score[rid] = dict(zip(g['day_ord'], g['score']))

# Predict each test horizon for each test region using 1825-day lookup
preds_5yr = []
n_match = 0
for rid in test_df['region_id'].unique():
    test_max_day = test_df[test_df['region_id'] == rid]['day_ord'].max()
    region_lookup = train_date_score.get(rid, {})
    row = {'region_id': rid}
    for h in range(1, 6):
        target_day = test_max_day + 7 * h
        lookup_day = target_day - 1825
        # Allow ±3 day search
        score = None
        for delta in range(0, 8):
            for sign in [0, -1, 1]:
                d = lookup_day + sign * delta
                if d in region_lookup:
                    score = region_lookup[d]; break
            if score is not None: break
        row[f'pred_week{h}'] = score if score is not None else 0.84  # fallback to global mean
        if score is not None and h == 1:
            n_match += 1
    preds_5yr.append(row)
print(f"  5-yr lookup matched {n_match}/{len(test_df['region_id'].unique())} regions (h1)")
pred_df = pd.DataFrame(preds_5yr)
out_path = ROOT / "submissions" / "_v18_pure_5yr_lag.csv"
pred_df.to_csv(out_path, index=False)
print(f"\nSaved candidate: {out_path}")
print(f"  preds mean={pred_df[PRED_COLS].values.mean():.4f} std={pred_df[PRED_COLS].values.std():.4f}")
print(f"\nNow evaluate against oracle to estimate public score...")
