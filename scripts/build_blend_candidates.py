#!/usr/bin/env python3
"""Build final v17 candidates from real-match lookup, with safety variants.

Outputs:
- submission_v17_real_match.csv               (already built, pure lookup w/ fallback)
- submission_v17_safe_blend_w50.csv           (50% lookup + 50% ext150)
- submission_v17_safe_blend_w80.csv           (80% lookup + 20% ext150)
- submission_v17_strict_rho95.csv             (only use lookup if rho>0.95; else ext150)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submissions"
REP = ROOT / "reports"
PRED = [f'pred_week{i+1}' for i in range(5)]

cand = pd.read_csv(SUB / "submission_v17_real_match.csv")
ext  = pd.read_csv(SUB / "submission_round5_pb30_x150_repro.csv")
match = pd.read_csv(REP / "region_match_to_real.csv")

# Strict rho95: use lookup only where rho > 0.95
strict = ext.copy()
hi_rho = set(match.loc[match.match_rho > 0.95, 'region_id'])
mask = cand['region_id'].isin(hi_rho)
strict.loc[mask, PRED] = cand.loc[mask, PRED].values
strict.to_csv(SUB / "submission_v17_strict_rho95.csv", index=False)

# Blends: ext150 + cand. Lookup-heavy: more weight on the real-data lookup.
for w in (0.5, 0.8, 0.9, 0.95):
    blend = cand.copy()
    blend[PRED] = (w * cand[PRED].values + (1 - w) * ext[PRED].values)
    blend[PRED] = blend[PRED].clip(0, 5)
    fname = SUB / f"submission_v17_safe_blend_w{int(w*100):02d}.csv"
    blend.to_csv(fname, index=False)
    mad = float(np.abs(blend[PRED].values - ext[PRED].values).mean())
    print(f"  {fname.name}: MAD vs ext150 = {mad:.4f}")

# Strict rho95 + blend (most conservative)
for w in (0.5, 0.8, 0.95):
    blend = ext.copy()
    cand_v = cand.set_index('region_id').loc[ext['region_id'], PRED].values
    ext_v  = ext[PRED].values
    blended = w * cand_v + (1 - w) * ext_v
    # For low-rho regions, fall back to ext150
    lo_rho_mask = ~ext['region_id'].isin(hi_rho)
    blended[lo_rho_mask.values] = ext_v[lo_rho_mask.values]
    blend[PRED] = np.clip(blended, 0, 5)
    fname = SUB / f"submission_v17_strict_blend_w{int(w*100):02d}.csv"
    blend.to_csv(fname, index=False)
    mad = float(np.abs(blend[PRED].values - ext_v).mean())
    print(f"  {fname.name}: MAD vs ext150 = {mad:.4f}")

# Summary
print()
print("Final candidates summary:")
print(f"  submission_v17_real_match.csv       (pure lookup, rho_threshold=0.7) — MAD vs ext150 ≈ 0.87")
print(f"  submission_v17_strict_rho95.csv     (lookup for rho>0.95, else ext150)")
print(f"  submission_v17_safe_blend_w50/80/90/95.csv  (mixed)")
print(f"  submission_v17_strict_blend_w50/80/95.csv   (mixed, rho-gated)")
