#!/usr/bin/env python3
"""Plan v6 E2/E3 blend: convex-blend a new model's submission with ext150.

Inputs:
  --candidate PATH    : path to candidate model submission CSV (e.g. E1 output)
  --ext150 PATH       : path to ext150 anchor (default: round5_pb30_x150_repro)
  --pb30 PATH         : path to pb30 baseline (default)
  --output-prefix STR : prefix for blend output CSVs (default submissions/v6_blend)

Output: multiple blends at weights {0.05, 0.10, 0.15, 0.20, 0.30, 0.50} with
ext150 and pb30 anchors. Prints MAD vs each anchor and PASS/FAIL on a 0.12
gate vs ext150.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUB = PROJECT_ROOT / "submissions"
DATA = PROJECT_ROOT / "data"
PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]


def load_aligned(path: Path, region_order: list[str]) -> np.ndarray:
    df = pd.read_csv(path)
    df["region_id"] = df["region_id"].astype(str)
    df = df.set_index("region_id").reindex(region_order).reset_index()
    if df[PRED_COLS].isna().any().any():
        raise SystemExit(f"[error] {path.name} missing regions after reindex")
    return df[PRED_COLS].to_numpy(dtype=np.float64)


def write(arr: np.ndarray, region_order: list[str], name: str) -> Path:
    df = pd.DataFrame(arr, columns=PRED_COLS)
    df.insert(0, "region_id", region_order)
    p = SUB / f"{name}.csv"
    df.to_csv(p, index=False)
    print(f"  wrote {p.name}  mean={arr.mean():.4f}  range=[{arr.min():.4f}, {arr.max():.4f}]")
    return p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--ext150", default=str(SUB / "submission_round5_pb30_x150_repro.csv"))
    parser.add_argument("--pb30", default=str(SUB / "submission_redo_blend_pb30.csv"))
    parser.add_argument("--output-prefix", default="submission_v6_blend")
    parser.add_argument("--mad-gate", type=float, default=0.12)
    args = parser.parse_args()

    sample = pd.read_csv(DATA / "sample_submission.csv")
    sample["region_id"] = sample["region_id"].astype(str)
    region_order = sample["region_id"].tolist()

    cand = load_aligned(Path(args.candidate), region_order)
    ext150 = load_aligned(Path(args.ext150), region_order)
    pb30 = load_aligned(Path(args.pb30), region_order)

    print(f"[info] candidate mean={cand.mean():.4f}  ext150 mean={ext150.mean():.4f}  pb30 mean={pb30.mean():.4f}")
    mad_cand_ext = float(np.abs(cand - ext150).mean())
    mad_cand_pb = float(np.abs(cand - pb30).mean())
    print(f"[info] MAD(candidate, ext150) = {mad_cand_ext:.4f}")
    print(f"[info] MAD(candidate, pb30)   = {mad_cand_pb:.4f}")
    print(f"[info] MAD(ext150, pb30)      = {float(np.abs(ext150 - pb30).mean()):.4f}")

    print(f"\n[step 1] Convex blends candidate × ext150 (gate MAD ≤ {args.mad_gate}):")
    print("  w_cand | mean   | MAD vs ext150 | MAD vs pb30  | status")
    pass_list = []
    for w in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        blend = w * cand + (1 - w) * ext150
        mad_ext = float(np.abs(blend - ext150).mean())
        mad_pb = float(np.abs(blend - pb30).mean())
        status = "PASS" if mad_ext <= args.mad_gate else "FAIL"
        print(f"  {w:.2f}    | {blend.mean():.4f} | {mad_ext:.4f}       | {mad_pb:.4f}       | {status}")
        if status == "PASS":
            pass_list.append((w, blend, mad_ext))

    print(f"\n[step 2] Convex blends candidate × pb30 (gate MAD ≤ {args.mad_gate} vs ext150):")
    print("  w_cand | mean   | MAD vs ext150 | MAD vs pb30  | status")
    for w in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        blend = w * cand + (1 - w) * pb30
        mad_ext = float(np.abs(blend - ext150).mean())
        mad_pb = float(np.abs(blend - pb30).mean())
        status = "PASS" if mad_ext <= args.mad_gate else "FAIL"
        print(f"  pb-{w:.2f} | {blend.mean():.4f} | {mad_ext:.4f}       | {mad_pb:.4f}       | {status}")

    print("\n[step 3] Write top PASS candidates ...")
    for w in [0.05, 0.10, 0.15, 0.20, 0.30]:
        blend = w * cand + (1 - w) * ext150
        mad_ext = float(np.abs(blend - ext150).mean())
        if mad_ext <= args.mad_gate:
            write(np.clip(blend, 0.0, 5.0), region_order, f"{args.output_prefix}_ext150_w{int(w * 100):02d}")
    # Also write pure candidate for reference
    write(np.clip(cand, 0.0, 5.0), region_order, f"{args.output_prefix}_pure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
