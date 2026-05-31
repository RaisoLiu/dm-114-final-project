#!/usr/bin/env python3
"""Plan v10 — Chronos zero-shot foundation-model inference on per-region score series.

For each of 2248 regions:
- Take last K (default 512) weekly scores from train as context.
- Predict 110 weeks ahead using chronos-bolt-base (or specified model).
- For each h ∈ 1..5, compute prediction index based on (test_end - last_score_date + 7*h).
- Map to pred_week_h in submission CSV.

Output: submissions/submission_chronos_zeroshot.csv + reports/chronos_zeroshot.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
SUB = PROJECT_ROOT / "submissions"
REPORTS = PROJECT_ROOT / "reports"
PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]

REGION_COL = "region_id"
DATE_COL = "date"
TARGET_COL = "score"


def date_to_ord(s: str) -> int:
    """Convert ISO date 'YYYY-MM-DD' to a synthetic day ordinal. Approx ignoring leap years for cross-region diffs."""
    y, m, d = [int(p) for p in s.split("-")]
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return y * 365 + sum(days_in_month[:m - 1]) + d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="amazon/chronos-bolt-base")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=110)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=str(SUB / "submission_chronos_zeroshot.csv"))
    parser.add_argument("--report", default=str(REPORTS / "chronos_zeroshot.json"))
    args = parser.parse_args()

    t0 = time.time()
    print(f"Loading model: {args.model} on {args.device}")
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        args.model, device_map=args.device, dtype=torch.float32,
    )
    print(f"  model loaded in {time.time() - t0:.1f}s")

    print("Loading data ...")
    train = pd.read_csv(DATA / "train.csv", usecols=[REGION_COL, DATE_COL, TARGET_COL])
    train[TARGET_COL] = pd.to_numeric(train[TARGET_COL], errors="coerce")
    train = train[train[TARGET_COL].notna()].copy()
    train[REGION_COL] = train[REGION_COL].astype(str)

    test = pd.read_csv(DATA / "test.csv", usecols=[REGION_COL, DATE_COL])
    test[REGION_COL] = test[REGION_COL].astype(str)
    test_end_date = test.groupby(REGION_COL, sort=False)[DATE_COL].last().to_dict()

    sample = pd.read_csv(DATA / "sample_submission.csv", usecols=[REGION_COL])
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    region_order = sample[REGION_COL].tolist()

    # Per-region: last score date + array of scores
    print("Preparing per-region context ...")
    region_data: dict[str, tuple[np.ndarray, str]] = {}  # region -> (scores, last_score_date)
    train_groups = {str(r): g for r, g in train.groupby(REGION_COL, sort=False)}
    for region in region_order:
        if region not in train_groups:
            continue
        g = train_groups[region].sort_values(DATE_COL)
        scores = g[TARGET_COL].to_numpy(dtype=np.float32)
        last_date = str(g[DATE_COL].iloc[-1])
        region_data[region] = (scores, last_date)
    print(f"  {len(region_data)} regions have score data")

    # Run Chronos in batches
    print(f"Running Chronos predict ({args.prediction_length} weeks ahead, batch size {args.batch_size}) ...")
    results = {}  # region -> forecast mean array of length prediction_length
    t_start = time.time()
    batch_inputs: list[torch.Tensor] = []
    batch_regions: list[str] = []
    n_done = 0
    n_total = len(region_order)
    for region in region_order:
        if region not in region_data:
            results[region] = np.full(args.prediction_length, fill_value=0.5)
            continue
        scores, _ = region_data[region]
        ctx = torch.tensor(scores[-args.context_length:], dtype=torch.float32)
        batch_inputs.append(ctx)
        batch_regions.append(region)
        if len(batch_inputs) >= args.batch_size:
            with torch.no_grad():
                quantiles, mean = pipe.predict_quantiles(
                    inputs=batch_inputs,
                    prediction_length=args.prediction_length,
                    quantile_levels=[0.5],
                )
            mean_np = mean.cpu().numpy()
            for i, r in enumerate(batch_regions):
                results[r] = mean_np[i]
            n_done += len(batch_inputs)
            if n_done % 256 == 0 or n_done == n_total:
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_total - n_done) / rate if rate > 0 else 0
                print(f"  {n_done}/{n_total} regions  ({elapsed:.0f}s, {rate:.1f} r/s, eta {eta:.0f}s)")
            batch_inputs.clear()
            batch_regions.clear()
    # Flush final batch
    if batch_inputs:
        with torch.no_grad():
            quantiles, mean = pipe.predict_quantiles(
                inputs=batch_inputs,
                prediction_length=args.prediction_length,
                quantile_levels=[0.5],
            )
        mean_np = mean.cpu().numpy()
        for i, r in enumerate(batch_regions):
            results[r] = mean_np[i]
        n_done += len(batch_inputs)
        print(f"  {n_done}/{n_total} regions (final batch)")

    print(f"All regions done in {time.time() - t_start:.1f}s")

    # Map per-region forecast to pred_week_h
    print("Mapping forecast to pred_week_h ...")
    out_rows = []
    for region in region_order:
        forecast = results[region]
        if region in region_data and region in test_end_date:
            _, last_date = region_data[region]
            test_end = test_end_date[region]
            days_to_test_end = date_to_ord(test_end) - date_to_ord(last_date)
            preds = []
            for h in range(1, 6):
                days_ahead = days_to_test_end + 7 * h
                # forecast index 0 = 7 days ahead; index 1 = 14 days ahead; etc.
                idx = days_ahead // 7 - 1
                idx = max(0, min(idx, len(forecast) - 1))
                preds.append(float(np.clip(forecast[idx], 0.0, 5.0)))
            out_rows.append([region] + preds)
        else:
            out_rows.append([region] + [0.5] * 5)

    out_df = pd.DataFrame(out_rows, columns=[REGION_COL] + PRED_COLS)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[info] wrote {out_path}")

    # Sanity
    print("\nSanity checks ...")
    print(f"  per-horizon mean: {[round(out_df[c].mean(), 4) for c in PRED_COLS]}")
    print(f"  overall mean: {out_df[PRED_COLS].values.mean():.4f}")
    print(f"  range: [{out_df[PRED_COLS].values.min():.4f}, {out_df[PRED_COLS].values.max():.4f}]")

    # Compare to ext150
    ext150 = pd.read_csv(SUB / "submission_round5_pb30_x150_repro.csv")
    ext150[REGION_COL] = ext150[REGION_COL].astype(str)
    common = set(out_df[REGION_COL]) & set(ext150[REGION_COL])
    e = ext150.set_index(REGION_COL).loc[list(common)][PRED_COLS].to_numpy(dtype=np.float64)
    o = out_df.set_index(REGION_COL).loc[list(common)][PRED_COLS].to_numpy(dtype=np.float64)
    mad = float(np.abs(o - e).mean())
    print(f"  MAD vs ext150: {mad:.4f}")
    # Per-horizon Pearson
    rhos = [float(np.corrcoef(o[:, h], e[:, h])[0, 1]) for h in range(5)]
    print(f"  Pearson(o, ext150) per horizon: {[round(r, 4) for r in rhos]}")

    report = {
        "model": args.model,
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "per_horizon_mean": [float(out_df[c].mean()) for c in PRED_COLS],
        "overall_mean": float(out_df[PRED_COLS].values.mean()),
        "min": float(out_df[PRED_COLS].values.min()),
        "max": float(out_df[PRED_COLS].values.max()),
        "mad_vs_ext150": mad,
        "pearson_vs_ext150": rhos,
        "n_regions": int(len(out_df)),
        "elapsed_sec_predict": time.time() - t_start,
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[info] wrote {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
