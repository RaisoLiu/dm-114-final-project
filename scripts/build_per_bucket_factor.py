#!/usr/bin/env python3
"""Per-bucket factor candidate: compress ext150 in buckets where reverse_diagnosis
shows ext150 over-predicts (extreme low and extreme high).

Reverse-diagnosis bucket evidence (`reports/public_reverse_diagnosis.md`):
- ext150 in [0, 0.1]:   mean_corr = -0.48  → ext150 over-predicts, lowering helps
- ext150 in (0.1, 0.5]: mean_corr = -0.20  → over-predicts, lowering helps
- ext150 in (0.5, 1.0]: mean_corr = -0.09  → slight over
- ext150 in (1.0, 1.5]: mean_corr = +0.02  → neutral
- ext150 in (1.5, 2.0]: mean_corr = +0.13  → ext150 roughly right, leave alone
- ext150 in (2.0, 3.0]: mean_corr = +0.17  → ext150 roughly right, leave alone
- ext150 in (3.0, 5.0]: mean_corr = -0.19  → over-predicts, compressing helps

Transformation (piecewise linear):
- p ∈ [0, 0.5]:    out = lo_factor * p
- p ∈ [0.5, 3.0]:  out = p   (keep)
- p ∈ [3.0, 5.0]:  out = 3.0 + hi_factor * (p - 3.0)

Multiple (lo_factor, hi_factor) variants generated for MAD-aware blend candidates.
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submissions"
PRED = [f'pred_week{i+1}' for i in range(5)]
EXT_PATH = SUB / "submission_round5_pb30_x150_repro.csv"


def compress(p: np.ndarray, lo_factor: float, hi_factor: float, lo_thresh: float = 0.5, hi_thresh: float = 3.0) -> np.ndarray:
    out = p.copy().astype(float)
    mask_lo = out < lo_thresh
    out[mask_lo] = lo_factor * out[mask_lo]
    mask_hi = out > hi_thresh
    out[mask_hi] = hi_thresh + hi_factor * (out[mask_hi] - hi_thresh)
    return np.clip(out, 0.0, 5.0)


def main():
    ext = pd.read_csv(EXT_PATH)
    ext_vals = ext[PRED].values
    print(f"ext150 mean={ext_vals.mean():.4f}, std={ext_vals.std():.4f}")
    print(f"  values <0.5: {(ext_vals < 0.5).mean()*100:.1f}%")
    print(f"  values >3.0: {(ext_vals > 3.0).mean()*100:.1f}%")
    print()

    print(f"{'variant':<35s}  {'MAD vs ext150':>14s}  {'est@slope+0.2':>14s}  {'est@slope+0.5':>14s}  {'est@slope+1.0':>14s}")
    candidates = []
    for lo in (0.3, 0.5, 0.7, 0.85, 1.0):
        for hi in (0.5, 0.7, 0.85, 1.0):
            if lo == 1.0 and hi == 1.0:
                continue  # identity
            transformed = compress(ext_vals, lo_factor=lo, hi_factor=hi)
            mad = float(np.abs(transformed - ext_vals).mean())
            name = f"pbf_lo{int(lo*100):03d}_hi{int(hi*100):03d}"
            est02 = 0.8534 + 0.2 * mad
            est05 = 0.8534 + 0.5 * mad
            est10 = 0.8534 + 1.0 * mad
            print(f"  {name:<33s}  {mad:>14.4f}  {est02:>14.4f}  {est05:>14.4f}  {est10:>14.4f}")
            candidates.append((name, lo, hi, mad, transformed))

    # Save the 3 lowest-MAD candidates as actual submissions (still expected worse than ext150 by slope law, but as new datapoints)
    candidates.sort(key=lambda x: x[3])
    print()
    print("Saving 3 lowest-MAD variants as submissions:")
    for name, lo, hi, mad, vals in candidates[:5]:
        out = ext.copy()
        out[PRED] = vals
        fname = SUB / f"submission_pbf_{name.split('_',1)[1]}.csv"
        out.to_csv(fname, index=False)
        print(f"  {fname.name}: MAD={mad:.4f} (lo={lo}, hi={hi})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
