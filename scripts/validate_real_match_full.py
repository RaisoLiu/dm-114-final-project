#!/usr/bin/env python3
"""Full validation across all high-rho regions.

For each region with rho>0.95:
- Take the LAST synth-train weekly score date as anchor
- Look back 5 anchors (i.e., 5 weeks earlier) — those are the "true future" relative to (anchor - 5 weeks)
- For each, compare real_lookup score to synth_train score
- Report MAE per horizon, exact-match%, distribution
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"

print("Loading match table...", flush=True)
m = pd.read_csv(REP / "region_match_to_real.csv", parse_dates=['matched_test_end_date'])

print("Loading synth test end-date...", flush=True)
synth_test = pd.read_csv(ROOT / "data" / "test.csv", usecols=['region_id','date'])
synth_test['date'] = synth_test['date'].astype(str)
synth_test_end = synth_test.groupby('region_id')['date'].max().reset_index()
synth_test_end['synth_test_year'] = synth_test_end['date'].str.slice(0, -6).astype(int)
m = m.merge(synth_test_end[['region_id','synth_test_year']], on='region_id', how='left')
m['real_match_year'] = m['matched_test_end_date'].dt.year
m['year_offset'] = m['real_match_year'] - m['synth_test_year']

print("Loading synth train scores...", flush=True)
synth_train = pd.read_csv(ROOT / "data" / "train.csv", usecols=['region_id','date','score'])
synth_train = synth_train.dropna(subset=['score'])
synth_train['date'] = synth_train['date'].astype(str)
print(f"  {len(synth_train):,} score rows", flush=True)

# Per-region synth scores indexed by string date
synth_by_region = {r: g.set_index('date')['score'] for r, g in synth_train.groupby('region_id')}

print("Loading real scores...", flush=True)
parts = []
for split in ['train_timeseries', 'validation_timeseries', 'test_timeseries']:
    df = pd.read_csv(ROOT / "data" / "external" / split / f"{split}.csv", usecols=['fips','date','score'])
    df = df.dropna(subset=['score'])
    parts.append(df)
real_scores = pd.concat(parts, ignore_index=True)
real_scores['date'] = pd.to_datetime(real_scores['date'])
real_by_fips = {f: g.set_index('date')['score'].sort_index() for f, g in real_scores.groupby('fips')}
print(f"  {len(real_scores):,} score rows", flush=True)

print("Validating across all high-rho regions...", flush=True)
matched_high = m[(m.match_rho > 0.95) & m.matched_fips.gt(0) & m.year_offset.notna()].copy()
print(f"  {len(matched_high)} high-rho regions to validate", flush=True)

# For each region, take many anchor dates from synth train and check
ae_per_horizon = [[] for _ in range(5)]
n_anchors_checked = 0
n_anchors_valid = 0
t0 = time.time()
for ri, (_, row) in enumerate(matched_high.iterrows()):
    rid = row['region_id']
    fips = int(row['matched_fips'])
    offset = int(row['year_offset'])
    sb_syn = synth_by_region.get(rid)
    sb_real = real_by_fips.get(fips)
    if sb_syn is None or sb_real is None:
        continue
    # Get the last 10 synth dates with scores (use only ones where we can look ahead 5 weeks)
    syn_dates = sorted(sb_syn.index.tolist())
    if len(syn_dates) < 10:
        continue
    # Take 10 anchors evenly spaced
    anchors = [syn_dates[i] for i in np.linspace(20, len(syn_dates)-50, 10).astype(int)]
    for anchor in anchors:
        n_anchors_checked += 1
        # Synth target dates for horizons 1..5 = anchor + 7,14,...,35 days
        # We need to compute real dates corresponding to these synth dates
        a_parts = anchor.split('-')
        if len(a_parts) != 3:
            continue
        syn_yr = int(a_parts[0]); md = a_parts[1] + '-' + a_parts[2]
        try:
            real_anchor = pd.Timestamp(f"{syn_yr + offset}-{md}")
        except Exception:
            continue
        if not (pd.Timestamp('2000-01-01') <= real_anchor <= pd.Timestamp('2020-11-26')):
            continue  # ensure 5 weeks ahead in data
        ok_this_anchor = False
        for h in range(1, 6):
            real_target = real_anchor + pd.Timedelta(days=7 * h)
            if real_target not in sb_real.index:
                near = sb_real.index[(sb_real.index >= real_target - pd.Timedelta(days=3)) & (sb_real.index <= real_target + pd.Timedelta(days=3))]
                if len(near) == 0:
                    continue
                real_target = near[abs((near - real_target).total_seconds()).argmin()]
            real_score = float(sb_real.loc[real_target])
            synth_target_year = real_target.year - offset
            synth_target = f"{synth_target_year:04d}-{real_target.strftime('%m-%d')}"
            if synth_target in sb_syn.index:
                synth_score = float(sb_syn.loc[synth_target])
                ae_per_horizon[h-1].append(abs(real_score - synth_score))
                ok_this_anchor = True
        if ok_this_anchor:
            n_anchors_valid += 1
    if (ri + 1) % 500 == 0 or ri + 1 == len(matched_high):
        el = time.time() - t0
        print(f"  [{ri+1:4d}/{len(matched_high)}] el={el:5.1f}s n_checked={n_anchors_checked} n_valid={n_anchors_valid}", flush=True)

print("\nValidation results (synth_train score vs real_lookup):", flush=True)
total = 0
for h in range(5):
    a = np.array(ae_per_horizon[h])
    if len(a):
        print(f"  h{h+1}: n={len(a):6d}  MAE={a.mean():.4f}  exact={a[a==0].size/len(a)*100:.1f}%  near_match<0.5: {a[a<0.5].size/len(a)*100:.1f}%  max={a.max():.2f}", flush=True)
        total += len(a)

all_ae = np.concatenate([np.array(a) for a in ae_per_horizon if a])
print(f"\nOverall n={total}, MAE={all_ae.mean():.4f}, median={np.median(all_ae):.4f}", flush=True)
print(f"  Histogram: [0,0.05):{(all_ae<0.05).sum()/total*100:.1f}%, [0.05,0.5):{((all_ae>=0.05)&(all_ae<0.5)).sum()/total*100:.1f}%, [0.5,1):{((all_ae>=0.5)&(all_ae<1)).sum()/total*100:.1f}%, [1,2):{((all_ae>=1)&(all_ae<2)).sum()/total*100:.1f}%, >=2:{(all_ae>=2).sum()/total*100:.1f}%", flush=True)
