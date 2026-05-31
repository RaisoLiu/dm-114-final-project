#!/usr/bin/env python3
"""Plan v12 Track C — Chronos2 fine-tune with weather past-covariates.

Approach:
- Per region, prepare:
  - target: 782 weekly scores
  - past_covariates: 14 weather features sampled at same weekly cadence
- Fine-tune Chronos2 with these inputs (LoRA or full)
- Predict prediction_length ahead (110 weeks) from full history
- Map predictions to pred_week_h for each region

Run on GB10 GPU.
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
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]


def date_to_ord(s: str) -> int:
    y, m, d = [int(p) for p in s.split("-")]
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return y * 365 + sum(days_in_month[:m - 1]) + d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--prediction-length", type=int, default=110)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-covariates", action="store_true", default=True)
    parser.add_argument("--no-covariates", dest="use_covariates", action="store_false")
    parser.add_argument("--output-csv", default=str(SUB / "submission_chronos2_with_covariates.csv"))
    parser.add_argument("--report-json", default=str(REPORTS / "chronos2_with_covariates.json"))
    args = parser.parse_args()

    t0 = time.time()
    print(f"Loading {args.model} on {args.device} ...")
    from chronos.chronos2 import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(args.model, device_map=args.device, dtype=torch.float32)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("Loading data ...")
    cols_to_load = [REGION_COL, DATE_COL, TARGET_COL] + WEATHER_COLS
    train = pd.read_csv(DATA / "train.csv", usecols=cols_to_load)
    train[TARGET_COL] = pd.to_numeric(train[TARGET_COL], errors="coerce")
    train_score_rows = train[train[TARGET_COL].notna()].copy()
    train[REGION_COL] = train[REGION_COL].astype(str)
    train_score_rows[REGION_COL] = train_score_rows[REGION_COL].astype(str)
    print(f"  {len(train_score_rows):,} non-null score rows across {train_score_rows[REGION_COL].nunique()} regions")

    test_full = pd.read_csv(DATA / "test.csv", usecols=[REGION_COL, DATE_COL] + WEATHER_COLS)
    test_full[REGION_COL] = test_full[REGION_COL].astype(str)
    test_end_date = test_full.groupby(REGION_COL, sort=False)[DATE_COL].last().to_dict()

    sample = pd.read_csv(DATA / "sample_submission.csv", usecols=[REGION_COL])
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    region_order = sample[REGION_COL].tolist()

    # Build per-region dict: {'target': scores, 'past_covariates': {wname: arr, ...}}
    print("Preparing fit() inputs ...")
    train_inputs: list[dict] = []
    region_last_date: dict[str, str] = {}
    region_target_for_inference: dict[str, dict] = {}
    train_score_groups = {str(r): g.sort_values(DATE_COL).reset_index(drop=True) for r, g in train_score_rows.groupby(REGION_COL, sort=False)}
    for region in region_order:
        if region not in train_score_groups:
            continue
        g = train_score_groups[region]
        n = len(g)
        if n < args.context_length:
            continue
        scores = g[TARGET_COL].to_numpy(dtype=np.float32)
        weather = {w: g[w].to_numpy(dtype=np.float32) for w in WEATHER_COLS} if args.use_covariates else None
        task = {"target": scores}
        if args.use_covariates:
            task["past_covariates"] = weather
        train_inputs.append(task)
        region_last_date[region] = str(g[DATE_COL].iloc[-1])
        region_target_for_inference[region] = task  # use full history for inference

    print(f"  {len(train_inputs)} regions in training")

    # Fine-tune
    print(f"Fine-tuning Chronos-2 (full, steps={args.num_steps}, batch={args.batch_size}, lr={args.learning_rate}) ...")
    t_train = time.time()
    pipe.fit(
        inputs=train_inputs,
        prediction_length=args.prediction_length,
        finetune_mode="full",
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        validation_inputs=None,
        disable_data_parallel=True,
    )
    print(f"  fine-tune done in {time.time() - t_train:.0f}s")

    # Inference
    print("Generating test predictions ...")
    inference_inputs = []
    inference_regions = []
    for region in region_order:
        if region in region_target_for_inference:
            inference_inputs.append(region_target_for_inference[region])
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

    # Median across samples
    forecast_arrs = {}
    for r, f in zip(inference_regions, forecasts):
        if f.ndim == 3:
            f_med = f[0].median(dim=0).values.cpu().numpy()
        elif f.ndim == 2:
            f_med = f.median(dim=0).values.cpu().numpy()
        else:
            f_med = f.cpu().numpy()
        forecast_arrs[r] = f_med.astype(np.float32)

    # Map to pred_week
    out_rows = []
    for region in region_order:
        if region not in forecast_arrs or region not in test_end_date:
            out_rows.append([region] + [0.5] * 5)
            continue
        last_date = region_last_date[region]
        days_to_test_end = date_to_ord(test_end_date[region]) - date_to_ord(last_date)
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
    print()
    print("Sanity checks ...")
    print(f"  per-horizon mean: {[round(out_df[c].mean(), 4) for c in PRED_COLS]}")
    print(f"  overall mean: {out_df[PRED_COLS].values.mean():.4f}")
    print(f"  range: [{out_df[PRED_COLS].values.min():.4f}, {out_df[PRED_COLS].values.max():.4f}]")

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
        "use_covariates": bool(args.use_covariates),
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
