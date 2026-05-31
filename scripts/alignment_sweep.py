#!/usr/bin/env python3
"""Plan v9 D1 — Empirical alignment sanity check.

The Phase-1 code audit confirmed train target offsets [(h+1)*7 for h in 0..4] = [7,14,21,28,35]
match the submission spec test_end + 7*h for h in 1..5. This script verifies empirically
that ext150 (0.8534 team best) is producing predictions consistent with a FORWARD-looking
interpretation, not a BACKWARD or off-by-one bug.

Key tests:
  T1 — per-horizon prediction means: are they monotone in a way that makes sense?
  T2 — correlation of (pred_week5 − pred_week1) with recent observed trend
       (last_train_score − train_score_5_weeks_earlier): if alignment is forward,
       correlation should be positive (persistence).
  T3 — if we assumed an off-by-one bug (shift +1 or −1 horizon), would the
       prediction-trend↔observed-trend correlation be higher?

If T1/T2 look fine and T3 doesn't reveal a better-fitting shift, alignment is correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
SUB = PROJECT_ROOT / "submissions"
REPORTS = PROJECT_ROOT / "reports"

PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]


def main() -> int:
    print("Loading data ...")
    train = pd.read_csv(DATA / "train.csv", usecols=["region_id", "date", "score"])
    train["score"] = pd.to_numeric(train["score"], errors="coerce")
    train = train[train["score"].notna()].copy()
    train["region_id"] = train["region_id"].astype(str)
    # NOTE: dates are in years 3004-3020 (synthetic) — pandas datetime OVERFLOWS for >2262.
    # Keep date as ISO-format string; lexicographic sort matches chronological.

    ext150 = pd.read_csv(SUB / "submission_round5_pb30_x150_repro.csv")
    ext150["region_id"] = ext150["region_id"].astype(str)

    sample = pd.read_csv(DATA / "sample_submission.csv", usecols=["region_id"])
    sample["region_id"] = sample["region_id"].astype(str)
    region_order = sample["region_id"].tolist()
    ext150 = ext150.set_index("region_id").reindex(region_order).reset_index()

    print(f"  train: {len(train):,} non-null score rows, {train['region_id'].nunique()} regions")
    print(f"  ext150: {len(ext150)} rows, columns: {list(ext150.columns)}")
    print()

    # ---------------------------------------------------------
    # T1 — per-horizon prediction means
    # ---------------------------------------------------------
    print("=" * 60)
    print("T1 — ext150 per-horizon means and std")
    print("=" * 60)
    for col in PRED_COLS:
        m = ext150[col].mean()
        s = ext150[col].std()
        print(f"  {col}: mean={m:.4f}  std={s:.4f}")
    overall_mean = ext150[PRED_COLS].values.mean()
    print(f"  OVERALL mean: {overall_mean:.4f}")
    print()

    # ---------------------------------------------------------
    # T2 — correlation of prediction trend with recent observed trend
    # ---------------------------------------------------------
    print("=" * 60)
    print("T2 — Forward-alignment trend correlation")
    print("=" * 60)
    print("Hypothesis: if alignment is FORWARD, then (pred_w5 - pred_w1) should be")
    print("positively correlated with the recent observed (last_score - score_5wk_earlier)")
    print("on a per-region basis (persistence).")
    print()

    # For each region, get the last N=5 observed scores (sorted by date)
    rows = []
    for region in region_order:
        sub = train[train["region_id"] == region].sort_values("date")
        if len(sub) < 6:
            continue
        # take last 5 + the 5-weeks-earlier comparison
        last_score = sub["score"].iloc[-1]
        score_5wk_back = sub["score"].iloc[-6]  # 5 weeks earlier
        score_4wk_back = sub["score"].iloc[-5]
        score_3wk_back = sub["score"].iloc[-4]
        score_2wk_back = sub["score"].iloc[-3]
        score_1wk_back = sub["score"].iloc[-2]
        rows.append({
            "region_id": region,
            "y_last": last_score,
            "y_5wk_back": score_5wk_back,
            "y_4wk_back": score_4wk_back,
            "y_3wk_back": score_3wk_back,
            "y_2wk_back": score_2wk_back,
            "y_1wk_back": score_1wk_back,
        })
    trend_df = pd.DataFrame(rows).set_index("region_id")
    e_df = ext150.set_index("region_id")

    # Align
    common = trend_df.index.intersection(e_df.index)
    print(f"  Regions with sufficient history: {len(common)} / {len(region_order)}")
    trend_df = trend_df.loc[common]
    e_df = e_df.loc[common]

    obs_trend = trend_df["y_last"] - trend_df["y_5wk_back"]
    pred_trend_fwd = e_df["pred_week5"] - e_df["pred_week1"]
    rho_fwd = float(np.corrcoef(obs_trend, pred_trend_fwd)[0, 1])

    print(f"  corr( (pred_w5 − pred_w1), (y_last − y_5wk_back) ) = {rho_fwd:+.4f}")
    print()
    print("  Interpretation:")
    print("    rho > 0.2  →  forward alignment looks correct (persistence holds)")
    print("    rho ~ 0    →  prediction independent of recent trend (weak signal)")
    print("    rho < -0.2 →  REVERSED — possible alignment bug")
    print()

    # ---------------------------------------------------------
    # T3 — what if we assumed a backward / off-by-one alignment?
    # ---------------------------------------------------------
    print("=" * 60)
    print("T3 — Alternative-alignment correlations")
    print("=" * 60)
    print("If ext150 were OFF-BY-ONE (or mislabeled), an alternative alignment might fit better.")
    print()

    # Suppose pred_week1..5 actually corresponds to past scores y_{-5,-4,-3,-2,-1} weeks back
    # (i.e., the team accidentally predicted the past). Then the per-region match between
    # ext150 and recent past scores would be HIGH.
    past_vector = np.column_stack([
        trend_df["y_5wk_back"].values,
        trend_df["y_4wk_back"].values,
        trend_df["y_3wk_back"].values,
        trend_df["y_2wk_back"].values,
        trend_df["y_1wk_back"].values,
    ])  # (n_regions, 5)
    pred_vector = e_df[PRED_COLS].values  # (n_regions, 5)

    # MAE if ext150 = past 5 weekly scores (reverse alignment, hypothetical bug)
    mae_reverse = float(np.abs(pred_vector - past_vector).mean())
    # MAE if ext150 = persistence (y_last replicated 5 times — forward but trivial)
    persistence = np.repeat(trend_df["y_last"].values.reshape(-1, 1), 5, axis=1)
    mae_persistence = float(np.abs(pred_vector - persistence).mean())

    print(f"  MAE(ext150, past_5_weeks)        = {mae_reverse:.4f}")
    print(f"  MAE(ext150, last_score_persisted) = {mae_persistence:.4f}")
    print()
    print("  Interpretation:")
    print("    If MAE(ext150, past_5_weeks) << MAE(ext150, persistence), ext150 may be")
    print("    accidentally predicting the past instead of the future.")
    print()

    # ---------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    if rho_fwd >= 0.20:
        verdict = "FORWARD alignment confirmed (rho_fwd >= 0.20)."
    elif rho_fwd >= -0.05:
        verdict = "Inconclusive (rho_fwd near zero). Likely no off-by-one bug, but signal is weak."
    else:
        verdict = "WARNING — negative correlation; possible reversed-alignment bug. Investigate."
    print(f"  {verdict}")
    if mae_reverse < mae_persistence * 0.9:
        print("  ALSO: ext150 fits past 5 weeks BETTER than persistence — RED FLAG.")
    else:
        print(f"  ext150 does NOT fit past 5 weeks better than persistence (good — not a backwards-bug).")
    print()

    # Persist results
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "alignment_sweep.md"
    with out.open("w") as f:
        f.write("# Plan v9 D1 — Alignment sweep result\n\n")
        f.write(f"**Test sample**: {len(common)} regions with ≥6 historical weekly scores in train.csv.\n\n")
        f.write("## T1 — ext150 per-horizon means\n\n")
        f.write("| Horizon | Mean | Std |\n|---|---|---|\n")
        for col in PRED_COLS:
            f.write(f"| {col} | {e_df[col].mean():.4f} | {e_df[col].std():.4f} |\n")
        f.write(f"| **Overall** | **{overall_mean:.4f}** | — |\n\n")
        f.write("## T2 — Forward trend correlation\n\n")
        f.write(f"`corr( (pred_w5 − pred_w1), (y_last − y_5wk_back) ) = {rho_fwd:+.4f}`\n\n")
        f.write("- rho > 0.20 → forward alignment confirmed (persistence holds in test direction)\n")
        f.write("- rho ~ 0 → weak signal but no bug\n")
        f.write("- rho < −0.2 → reversed alignment bug suspected\n\n")
        f.write("## T3 — Alternative alignments\n\n")
        f.write(f"- MAE(ext150, past_5_weeks_observed) = {mae_reverse:.4f}\n")
        f.write(f"- MAE(ext150, last_score_persisted) = {mae_persistence:.4f}\n\n")
        f.write("If reverse << persistence by > 10%, suggests ext150 may be predicting past instead of future.\n\n")
        f.write("## Verdict\n\n")
        f.write(f"{verdict}\n\n")
        if mae_reverse < mae_persistence * 0.9:
            f.write("**RED FLAG**: ext150 fits past 5 weeks better than persistence.\n")
        else:
            f.write("No backwards-alignment bug detected.\n")

    print(f"[info] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
