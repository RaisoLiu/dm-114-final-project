#!/usr/bin/env python3
"""Plan v11 — Chronos-2 LoRA fine-tune on per-region weekly score series.

Approach:
- Per region, prepare full weekly score history (~782 points).
- Hold out last 110 weeks for validation. Train Chronos-2 with LoRA.
- After fine-tune, predict 110 weeks ahead from FULL train history.
- For each region, extract prediction at gap_days + 7*h days ahead.
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
    y, m, d = [int(p) for p in s.split("-")]
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return y * 365 + sum(days_in_month[:m - 1]) + d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--finetune-mode", choices=["full", "lora"], default="lora")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--prediction-length", type=int, default=110)
    parser.add_argument("--val-horizon", type=int, default=50, help="hold out last N weeks per region")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-csv", default=str(SUB / "submission_chronos2_finetuned.csv"))
    parser.add_argument("--report-json", default=str(REPORTS / "chronos2_finetuned.json"))
    args = parser.parse_args()

    t0 = time.time()
    print(f"Loading {args.model} on {args.device} ...")
    from chronos.chronos2 import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(args.model, device_map=args.device, dtype=torch.float32)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("Loading data ...")
    train = pd.read_csv(DATA / "train.csv", usecols=[REGION_COL, DATE_COL, TARGET_COL])
    train[TARGET_COL] = pd.to_numeric(train[TARGET_COL], errors="coerce")
    train = train[train[TARGET_COL].notna()].copy()
    train[REGION_COL] = train[REGION_COL].astype(str)

    test = pd.read_csv(DATA / "test.csv", usecols=[REGION_COL, DATE_COL])
    test[REGION_COL] = test[REGION_COL].astype(str)
    test_end = test.groupby(REGION_COL, sort=False)[DATE_COL].last().to_dict()

    sample = pd.read_csv(DATA / "sample_submission.csv", usecols=[REGION_COL])
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    region_order = sample[REGION_COL].tolist()

    # Per-region: scores + last date
    region_scores: dict[str, np.ndarray] = {}
    region_last_date: dict[str, str] = {}
    for region, group in train.groupby(REGION_COL, sort=False):
        g = group.sort_values(DATE_COL)
        region_scores[str(region)] = g[TARGET_COL].to_numpy(dtype=np.float32)
        region_last_date[str(region)] = str(g[DATE_COL].iloc[-1])

    # Build training inputs: per-region series with last `val_horizon` weeks held out
    # The fit() method needs inputs and prediction_length — it will internally chunk the series.
    print(f"Preparing fit() inputs (holdout last {args.val_horizon} weeks per region) ...")
    train_inputs: list[np.ndarray] = []
    val_inputs: list[np.ndarray] = []
    for region in region_order:
        if region not in region_scores:
            continue
        s = region_scores[region]
        if len(s) < args.val_horizon + args.context_length:
            continue
        train_inputs.append(s[:-args.val_horizon].astype(np.float32))
        val_inputs.append(s.astype(np.float32))  # validation can see full

    print(f"  train_inputs: {len(train_inputs)} regions")

    # Fine-tune
    print(f"Fine-tuning Chronos-2 ({args.finetune_mode}, num_steps={args.num_steps}, batch_size={args.batch_size}) ...")
    t_train = time.time()
    pipe.fit(
        inputs=train_inputs,
        prediction_length=args.prediction_length,
        finetune_mode=args.finetune_mode,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        validation_inputs=None,
        disable_data_parallel=True,
    )
    print(f"  fine-tune done in {time.time() - t_train:.0f}s")

    # Inference: use FULL train history (no holdout) to predict 110 weeks ahead
    print("Generating test predictions ...")
    inference_inputs: list[np.ndarray] = []
    inference_regions: list[str] = []
    for region in region_order:
        if region not in region_scores:
            continue
        inference_inputs.append(region_scores[region].astype(np.float32))
        inference_regions.append(region)

    t_pred = time.time()
    with torch.no_grad():
        forecasts = pipe.predict(
            inputs=inference_inputs,
            prediction_length=args.prediction_length,
            batch_size=args.batch_size,
            limit_prediction_length=False,
        )
    print(f"  predict done in {time.time() - t_pred:.0f}s")

    # forecasts: list of tensors, each shape (n_samples=20 or so, prediction_length) probably
    # Use median across samples
    forecast_arrs: dict[str, np.ndarray] = {}
    for r, f in zip(inference_regions, forecasts):
        if f.ndim == 2:
            median = f.median(dim=0).values.cpu().numpy()
        elif f.ndim == 3:
            median = f.median(dim=1).values[0].cpu().numpy()
        else:
            median = f.cpu().numpy()
        forecast_arrs[r] = median.astype(np.float32)

    # Map per-region forecast to pred_week_h
    out_rows = []
    for region in region_order:
        if region not in forecast_arrs or region not in test_end:
            out_rows.append([region] + [0.5] * 5)
            continue
        last_date = region_last_date[region]
        days_to_test_end = date_to_ord(test_end[region]) - date_to_ord(last_date)
        forecast = forecast_arrs[region]
        preds = []
        for h in range(1, 6):
            days_ahead = days_to_test_end + 7 * h
            idx = days_ahead // 7 - 1
            idx = max(0, min(idx, len(forecast) - 1))
            preds.append(float(np.clip(forecast[idx], 0.0, 5.0)))
        out_rows.append([region] + preds)

    out_df = pd.DataFrame(out_rows, columns=[REGION_COL] + PRED_COLS)
    out_path = Path(args.output_csv)
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
    common = list(set(out_df[REGION_COL]) & set(ext150[REGION_COL]))
    e = ext150.set_index(REGION_COL).loc[common][PRED_COLS].to_numpy(dtype=np.float64)
    o = out_df.set_index(REGION_COL).loc[common][PRED_COLS].to_numpy(dtype=np.float64)
    mad = float(np.abs(o - e).mean())
    print(f"  MAD vs ext150: {mad:.4f}")
    rhos = [float(np.corrcoef(o[:, h], e[:, h])[0, 1]) for h in range(5)]
    print(f"  Pearson(o, ext150) per horizon: {[round(r, 4) for r in rhos]}")

    report = {
        "model": args.model,
        "finetune_mode": args.finetune_mode,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "prediction_length": args.prediction_length,
        "per_horizon_mean": [float(out_df[c].mean()) for c in PRED_COLS],
        "overall_mean": float(out_df[PRED_COLS].values.mean()),
        "mad_vs_ext150": mad,
        "pearson_vs_ext150": rhos,
        "n_regions": int(len(out_df)),
    }
    rp = Path(args.report_json)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[info] wrote {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
