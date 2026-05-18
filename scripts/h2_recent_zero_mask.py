#!/usr/bin/env python3
"""H2 — Recent-Zero Mask submission generator.

For each region where (a) the last 12 observed train scores are ALL 0
AND (b) ext150 average prediction > 1.0, shrink ext150 predictions toward
0 by factor α (per-horizon).

  pred_new[masked_region, h] = (1 - α) * ext150[masked_region, h]
  pred_new[other_region, h]  = ext150[other_region, h]

Workflow:
  1. Build trigger region set (576 regions).
  2. Build adversarial-val slice using top-p_test training anchors.
  3. For each candidate α, compute val-slice MAE (overall + masked-only)
     and Pearson ρ vs ext150 per-region errors.
  4. Pick α minimising val-MAE-overall subject to MAD vs ext150 ≤ 0.10.
  5. Write submission.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/h2_recent_zero_mask.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/raiso/DM_114_FinalProject_claude")
DATA = PROJECT_ROOT / "data"
SUB = PROJECT_ROOT / "submissions"
REPORTS = PROJECT_ROOT / "reports"

EXT150_PATH = SUB / "submission_round5_pb30_x150_repro.csv"
ADV_PATH = REPORTS / "adversarial_scores.csv"

PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]
ALPHA_GRID = [0.0, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00]
MAD_GATE = 0.10
# Top-p_test fraction used to define the adversarial-val slice.
# Adversarial scores covers ~1.72M (region, anchor) candidates; we use the
# tail with highest p_test to approximate the test-style distribution.
VAL_SLICE_PCT = 0.01  # top 1% by p_test
ADV_VAL_GAP_DAYS = 358  # not directly used; we use score-row offsets {6,13,20,27,34}


def build_trigger_regions(train: pd.DataFrame, ext150: pd.DataFrame) -> set[str]:
    scored = train.dropna(subset=["score"])
    last12 = scored.groupby("region_id").tail(12)
    counts = last12.groupby("region_id").size()
    maxes = last12.groupby("region_id")["score"].max()
    all_zero = (counts == 12) & (maxes == 0.0)

    ext150_avg = ext150.set_index("region_id")[PRED_COLS].mean(axis=1)
    triggers = {
        r for r in all_zero.index
        if bool(all_zero.loc[r]) and float(ext150_avg.get(r, 0.0)) > 1.0
    }
    return triggers


def build_val_slice(adv: pd.DataFrame, train: pd.DataFrame, top_pct: float) -> pd.DataFrame:
    """Return per-(region, anchor) val rows with 5-horizon y_true filled.

    Horizon targets are at score-bearing rows i+6, i+13, i+20, i+27, i+34.
    """
    threshold = adv["p_test"].quantile(1.0 - top_pct)
    slice_df = adv[adv["p_test"] >= threshold].copy().reset_index(drop=True)
    print(
        f"[val-slice] top {top_pct:.2%} cutoff p_test={threshold:.4f}  "
        f"rows={len(slice_df):,}  regions={slice_df['region_id'].nunique():,}"
    )

    # Build per-region row arrays for fast lookup
    train["rownum"] = train.groupby("region_id").cumcount()
    region_scores: dict[str, np.ndarray] = {}
    for r, g in train.groupby("region_id", sort=False):
        region_scores[str(r)] = g["score"].to_numpy(dtype=np.float64)

    horizon_offsets = np.array([6, 13, 20, 27, 34], dtype=np.int64)
    n = len(slice_df)
    y_true = np.full((n, 5), np.nan, dtype=np.float64)

    for i, (region, anchor) in enumerate(zip(slice_df["region_id"].astype(str), slice_df["anchor_index"].astype(int))):
        arr = region_scores.get(region)
        if arr is None:
            continue
        for h_idx, off in enumerate(horizon_offsets):
            tgt = int(anchor) + int(off)
            if 0 <= tgt < len(arr):
                y_true[i, h_idx] = arr[tgt]

    # Drop rows where any horizon is NaN (anchor too close to series end)
    good_mask = np.isfinite(y_true).all(axis=1)
    slice_df = slice_df.iloc[good_mask].reset_index(drop=True)
    y_true = y_true[good_mask]
    print(f"[val-slice] after dropping NaN-y_true rows: {len(slice_df):,}")
    for h in range(5):
        col = f"y_true_h{h + 1}"
        slice_df[col] = y_true[:, h]
    return slice_df


def main() -> int:
    print("[step 0] Load inputs ...")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sample["region_id"] = sample["region_id"].astype(str)
    region_order = sample["region_id"].tolist()
    print(f"  regions: {len(region_order):,}")

    train = pd.read_csv(DATA / "train.csv", usecols=["region_id", "date", "score"])
    train["region_id"] = train["region_id"].astype(str)
    ext150 = pd.read_csv(EXT150_PATH)
    ext150["region_id"] = ext150["region_id"].astype(str)
    ext150 = ext150.set_index("region_id").reindex(region_order).reset_index()

    print("[step 1] Build trigger regions (last 12 obs all 0 AND ext150 avg > 1.0) ...")
    triggers = build_trigger_regions(train, ext150)
    print(f"  trigger regions: {len(triggers):,}")
    is_masked = np.array([r in triggers for r in region_order], dtype=bool)

    print("[step 2] Build adversarial-val slice ...")
    adv = pd.read_csv(ADV_PATH)
    adv["region_id"] = adv["region_id"].astype(str)
    val = build_val_slice(adv, train, VAL_SLICE_PCT)

    y_true = val[[f"y_true_h{h + 1}" for h in range(5)]].to_numpy(dtype=np.float64)
    val_regions = val["region_id"].astype(str).to_numpy()

    # Lookup ext150 preds by region
    ext_arr = ext150.set_index("region_id")[PRED_COLS].to_numpy(dtype=np.float64)
    ext_idx_map = {r: i for i, r in enumerate(region_order)}
    region_to_ext = {r: ext_arr[ext_idx_map[r]] for r in region_order}

    # For each val row, look up ext150 prediction
    val_ext_pred = np.array(
        [region_to_ext[r] for r in val_regions], dtype=np.float64
    )  # shape (N_val, 5)

    val_is_masked = np.array([r in triggers for r in val_regions], dtype=bool)
    print(f"  val rows total={len(val_regions):,}  masked={val_is_masked.sum():,}  unmasked={(~val_is_masked).sum():,}")

    print(f"\n[step 3] Sweep α grid {ALPHA_GRID} ...")
    print("  alpha | val-MAE-all | val-MAE-masked | rho-vs-ext150-region-err | MAD-vs-ext150-full")
    results = []
    ext150_full = ext150[PRED_COLS].to_numpy(dtype=np.float64)

    # Per-region ext150 errors on val slice (for ρ calc): mean abs err per region
    def per_region_err(preds: np.ndarray) -> dict[str, float]:
        # preds shape (N_val, 5), aligned with val_regions
        abs_err = np.abs(preds - y_true)  # (N_val, 5)
        row_mae = abs_err.mean(axis=1)
        df = pd.DataFrame({"region": val_regions, "mae": row_mae})
        return df.groupby("region")["mae"].mean().to_dict()

    ext_region_err_map = per_region_err(val_ext_pred)
    common_regions_all = sorted(ext_region_err_map.keys())

    for alpha in ALPHA_GRID:
        # Build full submission pred
        pred_full = ext150_full.copy()
        pred_full[is_masked] = (1.0 - alpha) * pred_full[is_masked]
        pred_full = np.clip(pred_full, 0.0, 5.0)

        # MAD vs ext150 full
        mad_full = float(np.abs(pred_full - ext150_full).mean())

        # Val preds with this alpha
        val_pred = val_ext_pred.copy()
        val_pred[val_is_masked] = (1.0 - alpha) * val_pred[val_is_masked]
        val_pred = np.clip(val_pred, 0.0, 5.0)
        val_mae_all = float(np.abs(val_pred - y_true).mean())

        # Masked-only MAE
        if val_is_masked.any():
            val_mae_masked = float(np.abs(val_pred[val_is_masked] - y_true[val_is_masked]).mean())
        else:
            val_mae_masked = float("nan")

        # ρ vs ext150 per-region errors
        cand_region_err = per_region_err(val_pred)
        e_ext = np.array([ext_region_err_map[r] for r in common_regions_all])
        e_cand = np.array([cand_region_err[r] for r in common_regions_all])
        if e_ext.std() > 0 and e_cand.std() > 0:
            rho = float(np.corrcoef(e_ext, e_cand)[0, 1])
        else:
            rho = float("nan")

        results.append({
            "alpha": alpha,
            "val_mae_all": val_mae_all,
            "val_mae_masked": val_mae_masked,
            "rho_vs_ext_err": rho,
            "mad_vs_ext_full": mad_full,
        })
        print(
            f"  {alpha:.2f}  | {val_mae_all:.5f}    | {val_mae_masked:.5f}        | "
            f"{rho:+.4f}                  | {mad_full:.4f}"
        )

    print("\n[step 4] Pick α ...")
    ext_val_mae = results[0]["val_mae_all"]
    print(f"  ext150 (α=0) val-MAE-all = {ext_val_mae:.5f}")

    feasible = [r for r in results if r["alpha"] > 0 and r["mad_vs_ext_full"] <= MAD_GATE and r["val_mae_all"] <= ext_val_mae]
    if not feasible:
        print(
            f"  WARNING: no α satisfies (MAD ≤ {MAD_GATE}) AND (val-MAE ≤ baseline {ext_val_mae:.5f})"
        )
        # Fallback: pick the α with lowest val-MAE among those with MAD ≤ gate
        mad_ok = [r for r in results if r["alpha"] > 0 and r["mad_vs_ext_full"] <= MAD_GATE]
        if mad_ok:
            chosen = min(mad_ok, key=lambda r: r["val_mae_all"])
            print(f"  Fallback within MAD gate: α={chosen['alpha']:.2f}")
        else:
            print("  NO α even within MAD gate. Using α=0.0 (ext150 passthrough).")
            chosen = results[0]
    else:
        chosen = min(feasible, key=lambda r: r["val_mae_all"])
        print(f"  Chosen α={chosen['alpha']:.2f}  val-MAE={chosen['val_mae_all']:.5f}  MAD={chosen['mad_vs_ext_full']:.4f}")

    alpha_star = chosen["alpha"]

    print(f"\n[step 5] Write submission with α={alpha_star:.2f} ...")
    pred_full = ext150_full.copy()
    pred_full[is_masked] = (1.0 - alpha_star) * pred_full[is_masked]
    pred_full = np.clip(pred_full, 0.0, 5.0)

    out_df = pd.DataFrame(pred_full, columns=PRED_COLS)
    out_df.insert(0, "region_id", region_order)
    name = f"submission_v7_u2_recent_zero_mask_a{int(alpha_star * 100):02d}.csv"
    out_path = SUB / name
    out_df.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")
    print(f"  mean={pred_full.mean():.4f}  range=[{pred_full.min():.4f}, {pred_full.max():.4f}]  MAD vs ext150 = {chosen['mad_vs_ext_full']:.4f}")

    # Persist α-sweep table
    df_results = pd.DataFrame(results)
    df_results.to_csv(REPORTS / "h2_alpha_sweep.csv", index=False)
    print(f"  α-sweep table: {REPORTS / 'h2_alpha_sweep.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
