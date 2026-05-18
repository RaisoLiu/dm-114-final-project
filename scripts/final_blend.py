#!/usr/bin/env python3
"""Final ensemble + blend pipeline for Plan v5 Path B.

Discovers all submission_deep_*_fixed_*.csv legs in submissions/, filters out broken ones
(test mean near 0 = trans collapse, or test mean way off target), averages survivors,
and outputs convex blends with pb30 / ext150 at multiple ratios.

For each candidate, prints MAD vs ext150 (the 0.8534 public best) so eligibility gate
(MAD ≤ 0.10) can be enforced before upload.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/final_blend.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUB = PROJECT_ROOT / "submissions"
PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]

# Anchors
PB30 = "submission_redo_blend_pb30.csv"
EXT150 = "submission_round5_pb30_x150_repro.csv"

# Health bounds
MEAN_OK = (0.5, 2.5)
SAT5_FRAC_MAX = 0.30


def load_aligned(path: Path, region_order):
    df = pd.read_csv(path)
    df["region_id"] = df["region_id"].astype(str)
    df = df.set_index("region_id").reindex(region_order).reset_index()
    return df[PRED_COLS].to_numpy(dtype=np.float64)


def main() -> int:
    sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
    sample["region_id"] = sample["region_id"].astype(str)
    region_order = sample["region_id"].tolist()

    pb30 = load_aligned(SUB / PB30, region_order)
    ext150 = load_aligned(SUB / EXT150, region_order)
    print(f"[anchors] pb30 mean={pb30.mean():.4f}, ext150 mean={ext150.mean():.4f}")
    print(f"[anchors] MAD(pb30, ext150) = {np.abs(pb30 - ext150).mean():.4f}")
    print()

    print("[step 1] Discover fixed deep legs ...")
    legs: Dict[str, np.ndarray] = {}
    for p in sorted(SUB.glob("submission_deep_*_fixed_*.csv")):
        arr = load_aligned(p, region_order)
        m = arr.mean()
        sat = (arr >= 4.99).mean()
        sat_min = (arr <= 0.01).mean()
        key = p.stem.replace("submission_deep_", "")
        # Filter health
        if not (MEAN_OK[0] <= m <= MEAN_OK[1]):
            print(f"  [skip] {key} (mean={m:.4f} out of bounds)")
            continue
        if sat > SAT5_FRAC_MAX:
            print(f"  [skip] {key} (sat5={sat * 100:.1f}% > {SAT5_FRAC_MAX * 100:.0f}%)")
            continue
        if sat_min > 0.50:
            print(f"  [skip] {key} (sat0={sat_min * 100:.1f}% > 50%, suspect trans collapse)")
            continue
        legs[key] = arr
        print(f"  [keep] {key:35} mean={m:.4f} sat5={sat * 100:.1f}% sat0={sat_min * 100:.1f}%")
    if not legs:
        print("[error] no usable legs")
        return 1
    print(f"\n[info] using {len(legs)} legs: {list(legs.keys())}")

    print("\n[step 2] Build ensembles")
    ens = np.mean(list(legs.values()), axis=0)
    print(f"  simple avg: mean={ens.mean():.4f}, MAD vs ext150 = {np.abs(ens - ext150).mean():.4f}")

    print("\n[step 3] Convex blends with ext150 (gate: MAD ≤ 0.10)")
    print("  w_deep | blend mean | MAD vs ext150 | MAD vs pb30 | status")
    candidates = []
    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        blend = w * ens + (1 - w) * ext150
        mad_ext = float(np.abs(blend - ext150).mean())
        mad_pb = float(np.abs(blend - pb30).mean())
        status = "PASS" if mad_ext <= 0.10 else "FAIL"
        print(f"  {w:.2f}   | {blend.mean():.4f}     | {mad_ext:.4f}        | {mad_pb:.4f}     | {status}")
        if status == "PASS":
            candidates.append((w, blend, mad_ext))

    print("\n[step 4] Convex blends with pb30 (gate: MAD ≤ 0.10 vs ext150)")
    for w in [0.05, 0.10, 0.15, 0.20]:
        blend = w * ens + (1 - w) * pb30
        mad_ext = float(np.abs(blend - ext150).mean())
        mad_pb = float(np.abs(blend - pb30).mean())
        status = "PASS" if mad_ext <= 0.10 else "FAIL"
        print(f"  pb_w{w:.2f} | {blend.mean():.4f}     | {mad_ext:.4f}        | {mad_pb:.4f}     | {status}")

    print("\n[step 5] Write candidate submissions (top picks)")
    def write(arr, name):
        df = pd.DataFrame(arr, columns=PRED_COLS)
        df.insert(0, "region_id", region_order)
        p = SUB / f"submission_deep_ensemble_{name}.csv"
        df.to_csv(p, index=False)
        print(f"  wrote {p.name}  mean={arr.mean():.4f}  range=[{arr.min():.4f}, {arr.max():.4f}]")
        return p

    write(ens, "pure_avg")
    for w in [0.05, 0.10, 0.15]:
        blend = w * ens + (1 - w) * ext150
        write(blend, f"ext150_w{int(w * 100):02d}")
    for w in [0.05, 0.10]:
        blend = w * ens + (1 - w) * pb30
        write(blend, f"pb30_w{int(w * 100):02d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
