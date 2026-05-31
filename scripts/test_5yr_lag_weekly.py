#!/usr/bin/env python3
"""Test 5-year (1820 days = 260 weeks) lag prediction WITH proper weekly alignment.

For each test region, predict score(test_date + h*7) as score(test_date + h*7 - 260*7).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import time

ROOT = Path(__file__).resolve().parent.parent
PRED = [f'pred_week{i+1}' for i in range(5)]

print("Loading...")
train_df = pd.read_csv(ROOT / "data" / "train.csv", usecols=['region_id', 'date', 'score'])
test_df = pd.read_csv(ROOT / "data" / "test.csv", usecols=['region_id', 'date'])

# Use row index within each region as day_ordinal (since rows are sorted by date)
train_df = train_df.sort_values(['region_id', 'date']).reset_index(drop=True)
train_df['day_idx'] = train_df.groupby('region_id').cumcount()  # 0..5479 per region
test_df = test_df.sort_values(['region_id', 'date']).reset_index(drop=True)
test_df['day_idx'] = test_df.groupby('region_id').cumcount() + 5480  # 5480..5570

# Per region, find weekly anchors in train (where score is not null)
# Also: train_end is at day_idx=5479 of train; test runs day_idx 5480..5570
# Predictions are at day_idx = test_max + 7*h = 5570 + 7*h for h=1..5
# That's day_idx 5577, 5584, 5591, 5598, 5605
# For 5-year lag (1820 days), lookup day_idx = predict_day - 1820

# Compute train score lookup: for each (region, day_idx) where score not null
train_scored = train_df.dropna(subset=['score'])
score_lookup = {(r, d): s for r, d, s in zip(train_scored['region_id'], train_scored['day_idx'], train_scored['score'])}
print(f"  {len(score_lookup)} (region, day_idx) score entries")
print(f"  train day_idx range: 0..{train_df['day_idx'].max()}")
print(f"  test day_idx range: {test_df['day_idx'].min()}..{test_df['day_idx'].max()}")

# Now for each test region, look up at various lags
test_regions = test_df['region_id'].unique()
print(f"\nTrying multiple lags (in days):")
for lag_days in [1820, 1825, 1827, 1813, 1834, 365, 730, 1095, 1456, 2548]:
    n_match = 0
    n_total = 0
    for rid in test_regions:
        test_max = test_df[test_df['region_id'] == rid]['day_idx'].max()
        for h in range(1, 6):
            pred_day = test_max + 7 * h
            lookup_day = pred_day - lag_days
            # Try within ±3 days window
            for delta in [0, -1, 1, -2, 2, -3, 3]:
                if (rid, lookup_day + delta) in score_lookup:
                    n_match += 1
                    break
            n_total += 1
    rate = n_match / n_total if n_total > 0 else 0
    print(f"  lag={lag_days}d:  {n_match}/{n_total} ({rate*100:.1f}%) matched within ±3 d")

# OK let's use lag=1820 (=260 weeks) and save the prediction candidate
LAG = 1820
print(f"\n=== Build candidate with lag={LAG}d ===")
preds_rows = []
n_no_match = 0
for rid in test_regions:
    test_max = test_df[test_df['region_id'] == rid]['day_idx'].max()
    row = {'region_id': rid}
    for h in range(1, 6):
        pred_day = test_max + 7 * h
        lookup_day = pred_day - LAG
        score = None
        for delta in [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7]:
            if (rid, lookup_day + delta) in score_lookup:
                score = score_lookup[(rid, lookup_day + delta)]; break
        if score is None:
            n_no_match += 1
            score = 0.84  # fallback to global mean
        row[f'pred_week{h}'] = score
    preds_rows.append(row)
print(f"  {n_no_match} no-match (using fallback) out of {len(test_regions) * 5} predictions")
df_out = pd.DataFrame(preds_rows)
out_path = ROOT / "submissions" / "_v18_pure_5yr_lag_weekly.csv"
df_out.to_csv(out_path, index=False)
print(f"  saved: {out_path}")
print(f"  preds mean={df_out[PRED].values.mean():.4f}  std={df_out[PRED].values.std():.4f}")

# Also try blending with ext150
ext150 = pd.read_csv('submissions/submission_round5_pb30_x150_repro.csv')
common = sorted(set(ext150['region_id']) & set(df_out['region_id']))
e = ext150.set_index('region_id').loc[common][PRED].values
lag5 = df_out.set_index('region_id').loc[common][PRED].values
import sys; sys.path.insert(0, 'scripts')
from local_eval_gate import candidate_stats, predict_public, fit_calibration
truth = pd.read_csv('reports/_local_eval_oracle.csv')
df_report = pd.read_csv('reports/_local_eval_gate_report.csv')
known = df_report.dropna(subset=['public']).copy()
coef, info = fit_calibration(known)

# Raw lag5
df_pure = pd.DataFrame(lag5, columns=PRED); df_pure.insert(0, 'region_id', common)
s = candidate_stats(df_pure, ext150, truth)
pp = predict_public(coef, s)
print(f"\nPure 5-yr-lag-weekly: oracle={s['oracle_mae']:.4f}  mad={s['mad']:.4f}  std={s['std']:.3f}  mean={s['mean']:.3f}  pred_pub={pp:.4f}")

# Blend
print('Blend 5-yr-lag with ext150:')
for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90]:
    blend = alpha * lag5 + (1 - alpha) * e
    blend = np.clip(blend, 0, 5)
    df_b = pd.DataFrame(blend, columns=PRED); df_b.insert(0, 'region_id', common)
    s = candidate_stats(df_b, ext150, truth)
    pp = predict_public(coef, s)
    fname = f"_v18_lag5y_blend_a{int(alpha*100):02d}.csv"
    df_b.to_csv(f'submissions/{fname}', index=False)
    print(f'  a={alpha:.2f}  oracle={s["oracle_mae"]:.4f}  mad={s["mad"]:.4f}  pred_pub={pp:.4f}')
