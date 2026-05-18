"""Tiny probe: read train.csv and report future-target mean at each candidate validation delta.

Reproduces the validation-anchor logic of `train_gap_model.py` (subset) without training:
- Per region, find last_usable_anchor = len(group) - 36
- For each delta in args.deltas, compute the validation-anchor at last_usable_anchor - delta
- Look up the next 5 weekly score targets (the same way score_targets does)
- Aggregate: mean and std of next-5-week scores across all regions for that delta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REGION_COL = "region_id"
DATE_COL = "date"
TARGET_COL = "score"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--deltas", default="0,365,721,728,735,742,749,756,1095,1460,1825,2200")
    return p.parse_args()


def find_score_anchors(score_idx_arr: np.ndarray, score_values_arr: np.ndarray, anchor: int):
    """Find the next 5 weekly target rows starting AFTER anchor.

    Mirrors the logic from `train_gap_model.score_targets`: take the next 5 score rows
    (which are spaced ~7 days apart) starting from the first score whose index > anchor.
    """
    after = score_idx_arr[score_idx_arr > anchor]
    if after.size < 5:
        return None
    return score_values_arr[after[:5].astype(int) - score_idx_arr.min() if False else 0]
    # we won't use that path; instead use the helper below


def score_targets(score_idx: np.ndarray, score_values: np.ndarray, anchor: int):
    after_mask = score_idx > anchor
    if after_mask.sum() < 5:
        return None
    after_idx = np.where(after_mask)[0][:5]
    return score_values[after_idx]


def main() -> int:
    args = parse_args()
    deltas = [int(x.strip()) for x in args.deltas.split(",") if x.strip()]
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        print(f"[error] {train_path} not found")
        return 1
    print(f"[info] loading {train_path}")
    df = pd.read_csv(train_path)
    print(f"[info] loaded {len(df):,} rows, {df[REGION_COL].nunique()} regions")

    per_region_targets = {d: [] for d in deltas}

    for region, group in df.groupby(REGION_COL, sort=False):
        group = group.reset_index(drop=True)
        last_usable_anchor = len(group) - 36
        if last_usable_anchor < 91:
            continue
        score_idx = np.where(group[TARGET_COL].notna().to_numpy())[0]
        if score_idx.size == 0:
            continue
        score_values = group.loc[score_idx, TARGET_COL].to_numpy(dtype=np.float32)
        for d in deltas:
            anchor = last_usable_anchor - d
            if anchor < 91:
                continue
            t = score_targets(score_idx, score_values, anchor)
            if t is None:
                continue
            per_region_targets[d].append(t)

    print("\n=== Slice target distribution ===")
    print(f"{'delta':>6}  {'regions':>8}  {'targets':>8}  {'mean':>8}  {'std':>8}  {'severity':>10}")
    for d in deltas:
        rows = per_region_targets[d]
        if not rows:
            print(f"{d:>6}  {'0':>8}  {'0':>8}  {'-':>8}  {'-':>8}  {'no data':>10}")
            continue
        all_targets = np.concatenate(rows)
        mean = float(all_targets.mean())
        std = float(all_targets.std())
        if mean < 0.5:
            severity = "very low"
        elif mean < 0.85:
            severity = "low-med"
        elif mean < 1.10:
            severity = "med"
        elif mean < 1.30:
            severity = "high"
        else:
            severity = "very high"
        print(f"{d:>6}  {len(rows):>8}  {all_targets.size:>8}  {mean:>8.4f}  {std:>8.4f}  {severity:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
