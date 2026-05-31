#!/usr/bin/env python3
"""Minimal fast LGBM with just weekly score lags including 5-yr lag.
Should run in ~5 min."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
PRED = [f'pred_week{i+1}' for i in range(5)]
WEATHER_COLS = ['prec', 'surf_pre', 'humidity', 'tmp', 'dp_tmp', 'wb_tmp',
                'tmp_max', 'tmp_min', 'tmp_range', 'surf_tmp',
                'wind', 'wind_max', 'wind_min', 'wind_range']
SCORE_LAGS_DAYS = [7, 14, 21, 28, 35, 49, 70, 91, 182, 364, 728, 1092, 1456, 1820, 2548]

print("Loading...")
train_df = pd.read_csv(ROOT / "data" / "train.csv")
test_df = pd.read_csv(ROOT / "data" / "test.csv")
train_df = train_df.sort_values(['region_id', 'date']).reset_index(drop=True)
test_df = test_df.sort_values(['region_id', 'date']).reset_index(drop=True)

regions = train_df['region_id'].unique().tolist()
n_per_region = train_df.groupby('region_id').size().iloc[0]
s_all = train_df['score'].values.astype(np.float32).reshape(len(regions), n_per_region)
w_all = train_df[WEATHER_COLS].values.astype(np.float32).reshape(len(regions), n_per_region, len(WEATHER_COLS))

# Test weather (91 days each)
test_n_per = test_df.groupby('region_id').size().iloc[0]
w_test = test_df[WEATHER_COLS].values.astype(np.float32).reshape(len(regions), test_n_per, len(WEATHER_COLS))

# Build train anchors: for each (region, t), use weekly anchors
def build_features(anchor_t_per_region, w_full, s_full):
    """For each (r, t), build feature row."""
    rows = []
    for r, t in anchor_t_per_region:
        feats = {}
        # Weather rolling stats (28, 91 days back)
        for win in [28, 91]:
            if t + 1 - win < 0:
                start = 0
            else:
                start = t + 1 - win
            chunk = w_full[r, start:t+1, :]
            for c, col in enumerate(WEATHER_COLS):
                feats[f'{col}_w{win}_mean'] = float(chunk[:, c].mean())
                feats[f'{col}_w{win}_sum'] = float(chunk[:, c].sum())
        # Score lags
        for lag in SCORE_LAGS_DAYS:
            tl = t - lag
            if tl < 0:
                feats[f'sl_{lag}'] = -1.0
            else:
                # search nearest non-null within ±5
                s = s_full[r, tl]
                if not np.isnan(s):
                    feats[f'sl_{lag}'] = float(s)
                else:
                    for d in range(1, 6):
                        if tl - d >= 0 and not np.isnan(s_full[r, tl - d]):
                            feats[f'sl_{lag}'] = float(s_full[r, tl - d]); break
                        if tl + d <= t and not np.isnan(s_full[r, tl + d]):
                            feats[f'sl_{lag}'] = float(s_full[r, tl + d]); break
                    else:
                        feats[f'sl_{lag}'] = -1.0
        # Anchor position (within region)
        feats['anchor_t'] = float(t)
        # Multi-year phase
        for period in [1825, 1368, 365]:
            feats[f'phase_sin_{period}'] = float(np.sin(2 * np.pi * t / period))
            feats[f'phase_cos_{period}'] = float(np.cos(2 * np.pi * t / period))
        rows.append(feats)
    return pd.DataFrame(rows)

# Pick anchors from train: 200 per region
print("Building train anchors...")
train_anchors = []
train_targets = []
np.random.seed(0)
for r in range(len(regions)):
    s = s_all[r]
    valid_anchors = np.where(~np.isnan(s))[0]
    valid_anchors = valid_anchors[(valid_anchors >= 91) & (valid_anchors <= n_per_region - 36)]
    if len(valid_anchors) > 200:
        valid_anchors = valid_anchors[::len(valid_anchors)//200][:200]
    for t in valid_anchors:
        future = s[t + 7 * np.arange(1, 6)]
        if np.isnan(future).any(): continue
        train_anchors.append((r, t))
        train_targets.append(future)
print(f"  {len(train_anchors)} train anchors")

t0 = time.time()
X_tr_df = build_features(train_anchors, w_all, s_all)
y_tr = np.stack(train_targets).astype(np.float32)
print(f"  X_tr: {X_tr_df.shape}  build time: {time.time()-t0:.1f}s")

# Test anchors: each region's last day of test (= train + 91 - 1)
test_anchors = [(r, n_per_region + test_n_per - 1) for r in range(len(regions))]
# Concat train + test weather, train + nan score
full_w = np.concatenate([w_all, w_test], axis=1)
full_s = np.concatenate([s_all, np.full((len(regions), test_n_per), np.nan, dtype=np.float32)], axis=1)
X_te_df = build_features(test_anchors, full_w, full_s)
print(f"  X_te: {X_te_df.shape}")

# Train LGBM per horizon
print("Training LGBM per horizon...")
np.random.seed(0)
mask = np.random.rand(len(X_tr_df)) < 0.9
X_tr = X_tr_df.iloc[mask].reset_index(drop=True)
y_tr_h = y_tr[mask]
X_va = X_tr_df.iloc[~mask].reset_index(drop=True)
y_va_h = y_tr[~mask]

test_preds = np.zeros((len(X_te_df), 5), dtype=np.float32)
for h in range(5):
    params = dict(
        objective='regression_l1', metric='mae',
        num_leaves=127, min_data_in_leaf=200,
        learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        verbosity=-1, seed=42,
    )
    d_tr = lgb.Dataset(X_tr, label=y_tr_h[:, h])
    d_va = lgb.Dataset(X_va, label=y_va_h[:, h], reference=d_tr)
    model = lgb.train(params, d_tr, num_boost_round=800,
                       valid_sets=[d_va], callbacks=[lgb.early_stopping(60)])
    test_preds[:, h] = model.predict(X_te_df, num_iteration=model.best_iteration)
    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    val_mae = float(np.abs(val_pred - y_va_h[:, h]).mean())
    print(f"  h{h+1} val_mae={val_mae:.4f}  best_iter={model.best_iteration}")
test_preds = np.clip(test_preds, 0, 5)
out_df = pd.DataFrame({'region_id': regions})
for i, c in enumerate(PRED):
    out_df[c] = test_preds[:, i]
out_path = ROOT / "submissions" / "_v18_quick_lgbm_lags.csv"
out_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"  mean={test_preds.mean():.4f} std={test_preds.std():.4f}")

# Quick gate eval
import sys; sys.path.insert(0, 'scripts')
from local_eval_gate import candidate_stats, predict_public, fit_calibration
truth = pd.read_csv(ROOT / "reports" / "_local_eval_oracle.csv")
ext150 = pd.read_csv(ROOT / "submissions" / "submission_round5_pb30_x150_repro.csv")
df_report = pd.read_csv(ROOT / "reports" / "_local_eval_gate_report.csv")
known = df_report.dropna(subset=['public']).copy()
coef, info = fit_calibration(known)
s = candidate_stats(out_df, ext150, truth)
pp = predict_public(coef, s)
print(f'oracle={s["oracle_mae"]:.4f}  mad={s["mad"]:.4f}  std={s["std"]:.3f}  mean={s["mean"]:.3f}  pred_pub={pp:.4f}')

# Calculate rho with ext150 errors
common = sorted(set(ext150['region_id']) & set(regions))
e = ext150.set_index('region_id').loc[common][PRED].values
v = out_df.set_index('region_id').loc[common][PRED].values
tr = truth.set_index('region_id').loc[common][PRED].values
e_err = (e - tr).flatten()
v_err = (v - tr).flatten()
mask = ~np.isnan(e_err) & ~np.isnan(v_err)
rho = np.corrcoef(e_err[mask], v_err[mask])[0, 1]
print(f'ρ vs ext150 errors = {rho:.4f}')
