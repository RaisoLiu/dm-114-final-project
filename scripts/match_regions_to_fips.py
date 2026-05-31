#!/usr/bin/env python3
"""Plan v17 — Match synthetic regions to real FIPS counties using exact 91-day weather pattern.

Strategy:
1. For each synthetic region, climate fingerprint → top-K candidate FIPS
2. For each candidate FIPS, slide 91-day window through real val+test (2017-2020)
   ending on the same month-day as synth test end
3. Pick (FIPS, real_end_year) with highest Pearson on tmp + prec daily series
4. Save match table + matched real end date for downstream score lookup
"""
from __future__ import annotations
import pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"

# --- Synth side ---
synth_tst = pd.read_csv(ROOT / "data" / "test.csv")
synth_tst['date'] = synth_tst['date'].astype(str)
synth_tst['md'] = synth_tst['date'].str[-5:]  # mm-dd
print(f"Synth test rows: {len(synth_tst):,}, regions: {synth_tst.region_id.nunique()}")

# --- Real side: load cached ---
with open(REP / "real_by_fips.pkl", "rb") as f:
    real_by_fips = pickle.load(f)
print(f"Real FIPS loaded: {len(real_by_fips)}")

# --- Fingerprint kNN ---
fp_synth = pd.read_csv(REP / "climate_fingerprints.csv", index_col=0)
fp_ext   = pd.read_csv(REP / "external_fingerprints.csv", index_col=0)
common = [c for c in fp_synth.columns if c in fp_ext.columns and c != "hemisphere"]
X_ext  = fp_ext[common].astype(float).values
mu = X_ext.mean(0); sd = X_ext.std(0) + 1e-9
X_ext_n = (X_ext - mu)/sd
nn = NearestNeighbors(n_neighbors=30, metric='euclidean'); nn.fit(X_ext_n)
fips_arr = fp_ext.index.values

X_synth = fp_synth[common].astype(float).values
X_synth_n = (X_synth - mu)/sd
dists, idxs = nn.kneighbors(X_synth_n)
region_ids = fp_synth.index.values
region_to_candidates = {r: fips_arr[idxs[i]] for i, r in enumerate(region_ids)}

# Mapping synth column → real column
COL_MAP = {
    'prec': 'PRECTOT', 'surf_pre': 'PS', 'humidity': 'QV2M',
    'tmp': 'T2M', 'dp_tmp': 'T2MDEW', 'wb_tmp': 'T2MWET',
    'tmp_max': 'T2M_MAX', 'tmp_min': 'T2M_MIN', 'tmp_range': 'T2M_RANGE',
    'surf_tmp': 'TS',
    'wind': 'WS10M', 'wind_max': 'WS10M_MAX', 'wind_min': 'WS10M_MIN', 'wind_range': 'WS10M_RANGE',
}
PHYS_VARS = list(COL_MAP.keys())

def score_window(s_arr: np.ndarray, r_arr: np.ndarray) -> float:
    """Cross-variable mean of |Pearson(s_var, r_var)|, capped to [0,1]."""
    n_vars = s_arr.shape[1]
    rho = np.zeros(n_vars)
    for j in range(n_vars):
        sv = s_arr[:, j]; rv = r_arr[:, j]
        if sv.std() > 1e-6 and rv.std() > 1e-6:
            rho[j] = np.corrcoef(sv, rv)[0, 1]
        else:
            rho[j] = 0.0
    return float(np.mean(rho))


# --- Match per region ---
out_rows = []
t0 = time.time()
SYNTH_GROUPS = {r: g.reset_index(drop=True) for r, g in synth_tst.groupby('region_id')}
N = len(region_ids)
for i, rid in enumerate(region_ids):
    g = SYNTH_GROUPS.get(rid)
    if g is None or len(g) != 91:
        out_rows.append((rid, -1, None, 0.0, 0.0))
        continue
    g = g.sort_values('date').reset_index(drop=True)
    end_md = g['md'].iloc[-1]   # mm-dd e.g. "12-17"
    s_arr = g[PHYS_VARS].astype(float).values  # 91 × 14
    candidates = region_to_candidates[rid][:30]
    best = (-1, None, -1.0, 0.0)  # fips, real_end_date, score, n_match
    for fips in candidates:
        rdf = real_by_fips.get(int(fips))
        if rdf is None or len(rdf) == 0:
            continue
        real_cols = [COL_MAP[v] for v in PHYS_VARS]
        rdf_md = rdf['date'].dt.strftime('%m-%d')
        for end_dt in rdf.loc[rdf_md == end_md, 'date']:
            start_dt = end_dt - pd.Timedelta(days=90)
            win = rdf[(rdf['date'] >= start_dt) & (rdf['date'] <= end_dt)]
            if len(win) != 91:
                continue
            r_arr = win[real_cols].values
            sc = score_window(s_arr, r_arr)
            if sc > best[2]:
                best = (int(fips), end_dt, sc, len(win))
    out_rows.append((rid, best[0], best[1], best[2], best[3]))
    if (i + 1) % 50 == 0 or i + 1 == N:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (N - i - 1) / rate
        print(f"  [{i+1:4d}/{N}] elapsed {elapsed:6.1f}s, rate {rate:.2f} reg/s, ETA {eta/60:.1f}min, last={rid}: fips={best[0]} score={best[2]:.4f}")

# Save
out_df = pd.DataFrame(out_rows, columns=['region_id','matched_fips','matched_test_end_date','match_score','n_match_days'])
out_df.to_csv(REP / "region_match_to_real.csv", index=False)
print()
print("Match score distribution:")
print(out_df['match_score'].describe())
print()
print(f"Matches with score>0.95: {(out_df['match_score']>0.95).sum()} / {len(out_df)}")
print(f"Matches with score>0.90: {(out_df['match_score']>0.90).sum()} / {len(out_df)}")
print(f"Matches with score>0.80: {(out_df['match_score']>0.80).sum()} / {len(out_df)}")
print(f"Unique matched FIPS: {out_df['matched_fips'].nunique()}")
