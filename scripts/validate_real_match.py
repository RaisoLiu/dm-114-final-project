#!/usr/bin/env python3
"""Validate the synth→real match by checking known synth-train scores against real-data lookup.

For each region:
1. We know matched (FIPS, year) from region_match_to_real.csv. This gives the year-offset:
   synth_test_end_year minus real_match_year = constant per region.
2. Pick a 91-day window from synth TRAIN ending on a synth date.
3. The corresponding real date = synth_date with year shifted by the offset.
4. Look up real score at real_date + 7,14,21,28,35 days.
5. Compare to actual synth train score at synth_date + 7,...,35 days.

Reports MAE of (real_lookup, synth_train_actual).
"""
from __future__ import annotations
import pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"

# Load match table
m = pd.read_csv(REP / "region_match_to_real.csv", parse_dates=['matched_test_end_date'])

# Load synth train scores
print("Loading synth train scores...", flush=True)
synth_train = pd.read_csv(ROOT / "data" / "train.csv", usecols=['region_id','date','score'])
synth_train = synth_train.dropna(subset=['score'])
synth_train['date'] = synth_train['date'].astype(str)
synth_train['synth_year'] = synth_train['date'].str.slice(0, -6).astype(int)
synth_train['md'] = synth_train['date'].str[-5:]
print(f"  {len(synth_train):,} train score rows", flush=True)

# Load synth test end_date to compute synth_test_year per region
synth_test = pd.read_csv(ROOT / "data" / "test.csv", usecols=['region_id','date'])
synth_test['date'] = synth_test['date'].astype(str)
synth_test_end = synth_test.groupby('region_id')['date'].max().reset_index()
synth_test_end['synth_test_year'] = synth_test_end['date'].str.slice(0, -6).astype(int)
synth_test_end['synth_test_md']   = synth_test_end['date'].str[-5:]

m = m.merge(synth_test_end[['region_id','synth_test_year','synth_test_md']], on='region_id', how='left')
m['real_match_year'] = m['matched_test_end_date'].dt.year
m['year_offset']     = m['real_match_year'] - m['synth_test_year']  # real_year = synth_year + offset

# Quick offset distribution
print("\nYear-offset distribution (real_year - synth_year):", flush=True)
print(m['year_offset'].describe(), flush=True)

# Load real scores fast lookup
print("\nLoading real scores (all 3 splits)...", flush=True)
parts = []
for split in ['train_timeseries', 'validation_timeseries', 'test_timeseries']:
    df = pd.read_csv(ROOT / "data" / "external" / split / f"{split}.csv", usecols=['fips','date','score'])
    df = df.dropna(subset=['score'])
    parts.append(df)
real_scores = pd.concat(parts, ignore_index=True)
real_scores['date'] = pd.to_datetime(real_scores['date'])
real_scores = real_scores.sort_values(['fips','date']).reset_index(drop=True)
print(f"  {len(real_scores):,} score rows", flush=True)
scores_by_fips = {f: g.set_index('date')['score'] for f, g in real_scores.groupby('fips')}

# For each region with rho > 0.95 and valid offset, sample 3 train-score windows
# Take 91-day window ending on some train_date, then predict next 5 weeks
np.random.seed(0)
results = []
n_check = 0
matched_high = m[(m['match_rho'] > 0.95) & m['year_offset'].notna()]
print(f"\nValidating on {len(matched_high)} regions with rho>0.95...", flush=True)

# We'll pick the SYNTH last-observed-score date per region, then look at "what was the next 5 weeks score"
synth_last_scores = synth_train.groupby('region_id')['date'].max().reset_index().rename(columns={'date':'last_score_date'})
matched_high = matched_high.merge(synth_last_scores, on='region_id', how='left')
matched_high = matched_high.dropna(subset=['last_score_date'])

# Sample 200 regions for speed
sample = matched_high.sample(n=min(200, len(matched_high)), random_state=42)
print(f"  Sampled {len(sample)} regions for end-to-end check", flush=True)

# Need real score lookup: for synth date X (synth_year=Y, md=M), real date = (Y + offset)-M
def synth_to_real(synth_date_str: str, offset: int) -> pd.Timestamp:
    """e.g., '3019-12-25' with offset=-1001 → '2018-12-25'"""
    parts = synth_date_str.split('-')
    if len(parts) != 3:
        return pd.NaT
    syn_year = int(parts[0])
    md = parts[1] + '-' + parts[2]
    real_year = syn_year + offset
    return pd.Timestamp(f"{real_year}-{md}")

# Build per-region synth_train index (date -> score)
synth_train_by_region = {r: g.set_index('date')['score'] for r, g in synth_train.groupby('region_id')}

ok = 0
total_check = 0
ae_per_horizon = [[] for _ in range(5)]
for _, row in sample.iterrows():
    rid = row['region_id']
    fips = int(row['matched_fips'])
    offset = int(row['year_offset'])
    sb = scores_by_fips.get(fips)
    sb_syn = synth_train_by_region.get(rid)
    if sb is None or sb_syn is None:
        continue
    # Find some "synthetic end_dates" inside training data that have scores at +7,...,+35
    # Pick a synth date where there are at least 5 weekly scores after it within train
    syn_dates_with_score = sorted(sb_syn.index)
    # Try multiple anchor dates per region
    anchor_dates = syn_dates_with_score[100:200] if len(syn_dates_with_score) >= 200 else syn_dates_with_score[10:30]
    for anchor in anchor_dates[:5]:
        # Compute 5-week-ahead synth dates
        try:
            anchor_dt = pd.Timestamp(anchor)
        except Exception:
            continue  # synth years out of pd range
        # We can't use Timestamp arithmetic on synth fake years easily — use string add
        # Synth dates use format YYYY-MM-DD; offset days by string-to-real
        real_anchor = synth_to_real(anchor, offset)
        if pd.isna(real_anchor):
            continue
        # Check 5 weekly targets
        per_h_ok = 0
        for h in range(1, 6):
            real_target = real_anchor + pd.Timedelta(days=7 * h)
            # Find real score within ±3 days
            if real_target not in sb.index:
                near = sb.index[(sb.index >= real_target - pd.Timedelta(days=3)) & (sb.index <= real_target + pd.Timedelta(days=3))]
                if len(near) == 0:
                    continue
                real_target = near[abs((near - real_target).total_seconds()).argmin()]
            real_score = float(sb.loc[real_target])
            # Compute synth target date
            # anchor is like '3019-06-09'; add 7h days
            # We need synth_year + month/day + 7h days
            # Use a hack: compute the date string by shifting real_anchor backward by offset
            synth_target_real_dt = real_anchor + pd.Timedelta(days=7 * h)
            synth_target = f"{synth_target_real_dt.year - offset:04d}-{synth_target_real_dt.strftime('%m-%d')}"
            if synth_target in sb_syn.index:
                synth_score = float(sb_syn.loc[synth_target])
                ae_per_horizon[h-1].append(abs(real_score - synth_score))
                per_h_ok += 1
                total_check += 1
        if per_h_ok > 0:
            ok += 1

print(f"\nValidated {total_check} (synth, real) score pairs across {ok} anchor-region rows", flush=True)
for h in range(5):
    if ae_per_horizon[h]:
        ae = np.array(ae_per_horizon[h])
        print(f"  horizon h={h+1}: n={len(ae):4d}  MAE={ae.mean():.4f}  exact_match%={(ae == 0).mean()*100:.1f}%  near_match (<0.5)%={(ae < 0.5).mean()*100:.1f}%", flush=True)

# Overall
all_ae = np.concatenate([np.array(a) for a in ae_per_horizon if a])
if len(all_ae):
    print(f"\nOverall MAE(real_lookup, synth_train) = {all_ae.mean():.4f}, exact={float((all_ae == 0).mean()):.3f}", flush=True)
    print(f"Histogram of |delta|: [0,0.1):{(all_ae<0.1).sum()}, [0.1,0.5):{((all_ae>=0.1)&(all_ae<0.5)).sum()}, [0.5,1):{((all_ae>=0.5)&(all_ae<1)).sum()}, [1,2):{((all_ae>=1)&(all_ae<2)).sum()}, >=2:{(all_ae>=2).sum()}", flush=True)
