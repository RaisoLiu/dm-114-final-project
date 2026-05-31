#!/usr/bin/env python3
"""Plan v17 v2 — Vectorized matcher: synthetic region → real FIPS + real test_end_date.

Approach:
1. Stack real val+test (2017-2020) as fips × day × var array
2. Climate fingerprint kNN → top-K candidate FIPS per region
3. For each region × (K candidates × 4 years), slice 91-day window ending on synth's month-day
4. Pearson against synth test → pick best
"""
from __future__ import annotations
import pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REP = ROOT / "reports"

print("Loading data...", flush=True)
synth_tst = pd.read_csv(DATA / "test.csv")
synth_tst['date'] = synth_tst['date'].astype(str)
synth_tst['md'] = synth_tst['date'].str[-5:]

with open(REP / "real_by_fips.pkl", "rb") as f:
    real_by_fips = pickle.load(f)
fips_list = sorted(real_by_fips.keys())
fips_to_idx = {f: i for i, f in enumerate(fips_list)}
n_fips = len(fips_list)
print(f"  {n_fips} fips, ~{len(real_by_fips[fips_list[0]])} days each", flush=True)

# --- Stack real data into fips × day × var array ---
# Real cols
REAL_COLS = ['PRECTOT','PS','QV2M','T2M','T2MDEW','T2MWET','T2M_MAX','T2M_MIN','T2M_RANGE','TS','WS10M','WS10M_MAX','WS10M_MIN','WS10M_RANGE']
SYNTH_COLS = ['prec','surf_pre','humidity','tmp','dp_tmp','wb_tmp','tmp_max','tmp_min','tmp_range','surf_tmp','wind','wind_max','wind_min','wind_range']

# Use dates 2017-01-01 → 2020-12-31 → 1461 days, but a few fips may lack early-2017 → align by date
all_dates = pd.date_range('2017-01-01','2020-12-31',freq='D')
date_to_idx = {d: i for i, d in enumerate(all_dates)}
n_days = len(all_dates)
print(f"  date range: {all_dates[0].date()} → {all_dates[-1].date()} ({n_days} days)", flush=True)

R = np.full((n_fips, n_days, len(REAL_COLS)), np.nan, dtype=np.float32)
SCORE_R = np.full((n_fips, n_days), np.nan, dtype=np.float32)
for fi, f in enumerate(fips_list):
    g = real_by_fips[f]
    # Map dates to global index
    dts = g['date'].dt.normalize()
    idxs = dts.map(date_to_idx)
    valid = idxs.notna()
    g_v = g.loc[valid]
    iv = idxs[valid].astype(int).values
    R[fi, iv] = g_v[REAL_COLS].astype(np.float32).values
    if 'score' in g_v.columns:
        SCORE_R[fi, iv] = g_v['score'].astype(np.float32).values
print(f"  R shape={R.shape}, %nan={np.isnan(R).mean():.3f}", flush=True)
print(f"  SCORE_R: %valid={(~np.isnan(SCORE_R)).mean():.3f}", flush=True)

# --- Stack synth test ---
region_ids = np.array(synth_tst['region_id'].unique())
n_regions = len(region_ids)
print(f"  synth regions: {n_regions}", flush=True)
S = np.zeros((n_regions, 91, len(SYNTH_COLS)), dtype=np.float32)
synth_end_md = np.empty(n_regions, dtype=object)
region_to_ri = {}
for ri, rid in enumerate(region_ids):
    g = synth_tst[synth_tst.region_id == rid].sort_values('date').reset_index(drop=True)
    if len(g) != 91:
        continue
    S[ri] = g[SYNTH_COLS].astype(np.float32).values
    synth_end_md[ri] = g['md'].iloc[-1]
    region_to_ri[rid] = ri
print(f"  S shape={S.shape}", flush=True)

# --- Climate fingerprint kNN ---
fp_synth = pd.read_csv(REP / "climate_fingerprints.csv", index_col=0)
fp_ext   = pd.read_csv(REP / "external_fingerprints.csv", index_col=0)
common = [c for c in fp_synth.columns if c in fp_ext.columns and c != "hemisphere"]
X_ext  = fp_ext[common].astype(float).values
mu = X_ext.mean(0); sd = X_ext.std(0) + 1e-9
X_ext_n = (X_ext - mu)/sd
X_synth_n = ((fp_synth[common].astype(float).values) - mu)/sd
K = 50  # candidates
nn = NearestNeighbors(n_neighbors=K, metric='euclidean'); nn.fit(X_ext_n)
_, candidate_idxs = nn.kneighbors(X_synth_n)
# Map candidate row idxs → fips
fips_arr = fp_ext.index.values  # but order may differ. Use fips_to_idx mapping
# Build per-region candidate FIPS indices in R
fip_in_R = np.array([fips_to_idx.get(int(f), -1) for f in fips_arr])
region_idx_to_fips_R = []  # list[arr] per region row in fp_synth
synth_id_order = fp_synth.index.tolist()
synth_id_to_pos = {r: i for i, r in enumerate(synth_id_order)}
for r in region_ids:
    pos = synth_id_to_pos.get(r, -1)
    if pos < 0:
        region_idx_to_fips_R.append(np.array([], dtype=int))
        continue
    cs = candidate_idxs[pos]
    region_idx_to_fips_R.append(fip_in_R[cs])

print("Match loop ...", flush=True)
def pearson_batch(s_91x14: np.ndarray, r_NxDx14: np.ndarray) -> np.ndarray:
    """Vector cross-Pearson: mean over 14 vars of Pearson(s, r) for each of N candidates.
    s_91x14 shape (91,14). r_NxDx14 shape (N, 91, 14). Returns (N,) array of mean rho.
    Skips vars with zero variance."""
    s = s_91x14.astype(np.float32)
    r = r_NxDx14.astype(np.float32)
    s_mean = s.mean(axis=0, keepdims=True)
    r_mean = r.mean(axis=1, keepdims=True)
    s_c = s - s_mean
    r_c = r - r_mean
    num = (s_c[None] * r_c).sum(axis=1)  # (N, 14)
    s_std = np.sqrt((s_c**2).sum(axis=0))  # (14,)
    r_std = np.sqrt((r_c**2).sum(axis=1))  # (N, 14)
    denom = s_std[None] * r_std + 1e-9
    rho = num / denom
    # Mask any zero-var vars by setting their rho to nan, then nanmean
    rho[r_std == 0] = np.nan
    return np.nanmean(rho, axis=1)  # (N,)

out_rows = []
t0 = time.time()
for ri, rid in enumerate(region_ids):
    if synth_end_md[ri] is None:
        out_rows.append((rid, -1, None, np.nan, np.nan, 0))
        continue
    end_md = synth_end_md[ri]
    s_arr = S[ri]   # (91, 14)
    candidates_R = region_idx_to_fips_R[ri]
    candidates_R = candidates_R[candidates_R >= 0]

    # For each candidate FIPS × each year (2017-2020), find the end_date matching end_md
    best_rho = -1e9; best_fips = -1; best_end_date = None; best_year = -1
    end_year_set = list(range(2017, 2021))
    for year in end_year_set:
        end_date_str = f"{year}-{end_md}"
        try:
            end_dt = pd.Timestamp(end_date_str)
        except Exception:
            continue
        end_idx = date_to_idx.get(end_dt)
        if end_idx is None or end_idx < 90:
            continue
        slc = slice(end_idx - 90, end_idx + 1)  # 91 days inclusive
        win = R[candidates_R, slc, :]  # (n_cand, 91, 14)
        # Skip candidates with any nan
        nan_mask = np.isnan(win).any(axis=(1,2))
        if nan_mask.all():
            continue
        valid_win = win[~nan_mask]
        valid_R = candidates_R[~nan_mask]
        rhos = pearson_batch(s_arr, valid_win)
        bi = int(np.nanargmax(rhos))
        if rhos[bi] > best_rho:
            best_rho = float(rhos[bi])
            best_fips = int(fips_list[valid_R[bi]])
            best_end_date = end_dt
            best_year = year

    out_rows.append((rid, best_fips, best_end_date, best_rho, best_year, len(candidates_R)))
    if (ri + 1) % 100 == 0 or ri + 1 == n_regions:
        el = time.time() - t0
        rate = (ri + 1) / el
        eta = (n_regions - ri - 1) / rate
        print(f"  [{ri+1:4d}/{n_regions}] el={el:5.1f}s rate={rate:.1f} reg/s eta={eta/60:.1f}min last={rid}: fips={best_fips} year={best_year} rho={best_rho:.4f}", flush=True)

print("Saving...", flush=True)
out_df = pd.DataFrame(out_rows, columns=['region_id','matched_fips','matched_test_end_date','match_rho','matched_year','n_candidates'])
out_df.to_csv(REP / "region_match_to_real.csv", index=False)
print()
print("Match rho distribution:", flush=True)
print(out_df['match_rho'].describe(), flush=True)
print()
print(f"Matches with rho>0.95: {(out_df['match_rho']>0.95).sum()} / {len(out_df)}", flush=True)
print(f"Matches with rho>0.90: {(out_df['match_rho']>0.90).sum()} / {len(out_df)}", flush=True)
print(f"Matches with rho>0.80: {(out_df['match_rho']>0.80).sum()} / {len(out_df)}", flush=True)
print(f"Unique matched FIPS: {out_df['matched_fips'].nunique()}", flush=True)
print()
print("Matched year distribution:", flush=True)
print(out_df['matched_year'].value_counts().sort_index(), flush=True)
