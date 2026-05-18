#!/usr/bin/env python3
"""Calibrate + average 6 deep-leg submissions and (optionally) extrapolate.

Pipeline:
  1. For each deep leg `submission_deep_<arch>_s<seed>.csv`:
       a. Compute current per-leg test mean.
       b. Additive shift so mean lands at TARGET_MEAN (cap |shift| ≤ 0.40).
  2. Average across legs (per (region, horizon)).
  3. Write `submission_deep_ensemble_raw.csv` and a sweep over shift / extrap factors.

MAD eligibility gate (Plan v5): vs `submission_redo_extrapolate_150_mean12334.csv` (0.8534 public best).

Usage:
  .venv/bin/python scripts/avg_deep_ensemble.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS = PROJECT_ROOT / "submissions"
DATA = PROJECT_ROOT / "data"
PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]
TARGET_MEAN = 1.234
SHIFT_CAP = 0.40
PUBLIC_BEST_CSV = "submission_redo_extrapolate_150_mean12334.csv"
MAD_GATE = 0.10

LEGS = [
    "submission_deep_cnn_fixed_s114.csv",
    "submission_deep_cnn_fixed_s271828.csv",
    "submission_deep_lstm_fixed_s114.csv",
    "submission_deep_lstm_fixed_s271828.csv",
    "submission_deep_trans_fixed_s114.csv",
    "submission_deep_trans_fixed_s271828.csv",
]


def load_aligned(path: Path, region_order: list[str]) -> np.ndarray:
    df = pd.read_csv(path)
    df["region_id"] = df["region_id"].astype(str)
    df = df.set_index("region_id").reindex(region_order).reset_index()
    if df[PRED_COLS].isna().any().any():
        raise SystemExit(f"[error] {path.name} missing some regions after reindex")
    return df[PRED_COLS].to_numpy(dtype=np.float64)


def write(arr: np.ndarray, region_order: list[str], name: str) -> Path:
    df = pd.DataFrame(arr, columns=PRED_COLS)
    df.insert(0, "region_id", region_order)
    p = SUBMISSIONS / f"submission_deep_ensemble_{name}.csv"
    df.to_csv(p, index=False)
    print(f"  wrote {p.name}  mean={arr.mean():.4f}  range=[{arr.min():.4f}, {arr.max():.4f}]")
    return p


def main() -> int:
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sample["region_id"] = sample["region_id"].astype(str)
    region_order = sample["region_id"].tolist()

    print("[step 1] load legs and compute per-leg shifts")
    legs_data = {}
    found = 0
    for fname in LEGS:
        p = SUBMISSIONS / fname
        if not p.exists():
            print(f"  [skip] {fname} not yet trained")
            continue
        arr = load_aligned(p, region_order)
        cur_mean = float(arr.mean())
        shift = TARGET_MEAN - cur_mean
        shift_clamped = float(np.clip(shift, -SHIFT_CAP, SHIFT_CAP))
        arr_shifted = np.clip(arr + shift_clamped, 0.0, 5.0)
        legs_data[fname] = arr_shifted
        found += 1
        print(f"  {fname:<42s}  raw_mean={cur_mean:.4f}  shift={shift_clamped:+.4f}  new_mean={arr_shifted.mean():.4f}")
    if found == 0:
        print("[error] no deep submissions found yet")
        return 1
    print(f"[info] loaded {found} legs")

    print("\n[step 2] average across legs")
    stack = np.stack(list(legs_data.values()), axis=0)
    ens = stack.mean(axis=0)
    print(f"  ensemble mean: {ens.mean():.4f}")

    print("\n[step 3] MAD vs public best (gate ≤ 0.10)")
    pb_path = SUBMISSIONS / PUBLIC_BEST_CSV
    if pb_path.exists():
        pb = load_aligned(pb_path, region_order)
        mad = float(np.abs(ens - pb).mean())
        status = "PASS" if mad <= MAD_GATE else "FAIL"
        print(f"  MAD vs pb_ext150 = {mad:.4f}  [{status}]")
    else:
        mad = None
        print(f"  [warn] {PUBLIC_BEST_CSV} missing; skipping MAD gate")

    print("\n[step 4] write candidates")
    write(ens, region_order, "raw")
    for shift in [0.0, 0.05, 0.10]:
        if shift == 0.0:
            continue
        shifted = np.clip(ens + shift, 0.0, 5.0)
        write(shifted, region_order, f"shift{int(shift * 100):02d}")
    for factor in [1.3, 1.5]:
        m = ens.mean()
        extrap = np.clip(m + factor * (ens - m), 0.0, 5.0)
        write(extrap, region_order, f"x{int(factor * 100):03d}")

    if mad is not None and mad > MAD_GATE:
        print(f"\n[warn] ensemble MAD {mad:.4f} exceeds gate {MAD_GATE} — review before upload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
