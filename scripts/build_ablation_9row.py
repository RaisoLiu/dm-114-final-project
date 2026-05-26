#!/usr/bin/env python3
"""Build the 9-row controlled ablation table for the iter4 report.

All rows are computed from OOF predictions aligned on (row_index, region_id, horizon).
Sources:
  reports/oof_tensor.csv                                — 6 GBDT legs (b91 + pl families)
  reports/lag_2215_oof_validation_predictions.csv       — Phase 1 regen
  reports/deep_cnn_*_validation_predictions.csv         — 5 cached CNN runs
  reports/track1_ssl_oof_validation_predictions.csv     — Phase 2 regen
  reports/training_menu_v1.json                         — convex blend weights + postproc

Outputs:
  reports/ablation_9row.csv
  reports/ablation_9row.md
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def affine_clip(pred: np.ndarray, menu: dict) -> np.ndarray:
    pp = menu["postproc"]
    out = pred * pp["scale"] + pp["shift"]
    out = np.clip(out, 0.0, pp["clip_max"])
    return out


def main() -> None:
    menu = json.loads((REPORTS / "training_menu_v1.json").read_text())
    weights = menu["ensemble"]["convex_weights"]  # A=0.2 B=0.45 C=0.25 D=0.1

    oof = pd.read_csv(REPORTS / "oof_tensor.csv")
    oof["region_id"] = oof["region_id"].astype(str)
    print(f"[ablation] oof_tensor: {len(oof)} rows × {len(oof.columns)} cols", flush=True)

    gbdt_cols = [c for c in oof.columns if c.startswith("pred_b91_") or c.startswith("pred_pl_")]
    print(f"[ablation] GBDT cols: {gbdt_cols}", flush=True)

    # GBDT anchor = mean of all 6 GBDT legs (the "ext150 family blend" the report calls anchor)
    oof["gbdt_anchor"] = oof[gbdt_cols].mean(axis=1)
    # Weather-only-single-GBDT: best of the 6 legs by OOF MAE
    leg_maes = {c: mae(oof[c].to_numpy(), oof["y_true"].to_numpy()) for c in gbdt_cols}
    best_single_col = min(leg_maes, key=leg_maes.get)
    oof["gbdt_single_best"] = oof[best_single_col]

    # b91 family blend (3 legs) and pl family blend (3 legs)
    b91_cols = [c for c in gbdt_cols if "b91" in c]
    pl_cols = [c for c in gbdt_cols if "pl" in c]
    oof["gbdt_b91"] = oof[b91_cols].mean(axis=1)
    oof["gbdt_pl"] = oof[pl_cols].mean(axis=1)

    # Lag-2215 leg (joined on (row_index, region_id, horizon))
    lag_df = pd.read_csv(REPORTS / "lag_2215_oof_validation_predictions.csv",
                          usecols=["row_index", "region_id", "horizon", "pred_final_calibrated"])
    lag_df["region_id"] = lag_df["region_id"].astype(str)
    lag_df = lag_df.rename(columns={"pred_final_calibrated": "pred_lag2215"})
    oof = oof.merge(lag_df, on=["row_index", "region_id", "horizon"], how="left")
    assert oof["pred_lag2215"].notna().all(), "lag-2215 join produced NaNs — alignment broken"

    # CNN: average over all cached deep_cnn validation predictions
    cnn_files = sorted(REPORTS.glob("deep_cnn_*_validation_predictions.csv"))
    print(f"[ablation] CNN files: {[p.name for p in cnn_files]}", flush=True)
    cnn_preds = []
    for path in cnn_files:
        d = pd.read_csv(path, usecols=["row_index", "region_id", "horizon", "pred_final_calibrated"])
        d["region_id"] = d["region_id"].astype(str)
        d = d.rename(columns={"pred_final_calibrated": f"pred_{path.stem}"})
        cnn_preds.append(d)
    cnn_merged = cnn_preds[0]
    for d in cnn_preds[1:]:
        cnn_merged = cnn_merged.merge(d, on=["row_index", "region_id", "horizon"], how="inner")
    cnn_cols = [c for c in cnn_merged.columns if c.startswith("pred_")]
    cnn_merged["pred_cnn_mean"] = cnn_merged[cnn_cols].mean(axis=1)
    oof = oof.merge(cnn_merged[["row_index", "region_id", "horizon", "pred_cnn_mean"]],
                    on=["row_index", "region_id", "horizon"], how="left")
    assert oof["pred_cnn_mean"].notna().all(), "CNN join produced NaNs — alignment broken"

    # SSL Transformer
    ssl_df = pd.read_csv(REPORTS / "track1_ssl_oof_validation_predictions.csv",
                          usecols=["row_index", "region_id", "horizon", "pred_final_calibrated"])
    ssl_df["region_id"] = ssl_df["region_id"].astype(str)
    ssl_df = ssl_df.rename(columns={"pred_final_calibrated": "pred_ssl"})
    oof = oof.merge(ssl_df, on=["row_index", "region_id", "horizon"], how="left")
    assert oof["pred_ssl"].notna().all(), "SSL join produced NaNs — alignment broken"

    y = oof["y_true"].to_numpy()

    # Lag-residual correction: lag-2215 with per-horizon mean bias removed (the cheapest possible "residual correction")
    oof_lag_resid = oof["pred_lag2215"].to_numpy().copy()
    for h in (1, 2, 3, 4, 5):
        mask = (oof["horizon"] == h).to_numpy()
        bias = (y[mask] - oof_lag_resid[mask]).mean()
        oof_lag_resid[mask] = oof_lag_resid[mask] + bias

    # Convex blends per training_menu
    A = oof["gbdt_anchor"].to_numpy()
    C_lag = oof["pred_lag2215"].to_numpy()
    B_deep_cnn = oof["pred_cnn_mean"].to_numpy()
    B_deep_ssl = oof["pred_ssl"].to_numpy()
    # "B_Deep" averages the 2 deep members (track1_ssl_finetuned + track3_cnn_ttt) per menu
    B_deep = 0.5 * B_deep_cnn + 0.5 * B_deep_ssl

    # Row 5 (GBDT + lag): renormalize A+C to sum 1
    w_AC = weights["A"] + weights["C"]
    pred_gbdt_lag = (weights["A"] / w_AC) * A + (weights["C"] / w_AC) * C_lag

    # Row 6 (GBDT + lag + CNN): renormalize A+C+B_cnn
    # B_cnn weight = B/2 (since B_Deep is CNN+SSL averaged)
    w_AC_Bcnn = weights["A"] + weights["C"] + weights["B"] / 2
    pred_gbdt_lag_cnn = (
        (weights["A"] / w_AC_Bcnn) * A
        + (weights["C"] / w_AC_Bcnn) * C_lag
        + ((weights["B"] / 2) / w_AC_Bcnn) * B_deep_cnn
    )

    # Row 7 (GBDT + lag + CNN + SSL): renormalize A+C+B (B is full deep family)
    w_ACB = weights["A"] + weights["C"] + weights["B"]
    pred_gbdt_lag_cnn_ssl = (
        (weights["A"] / w_ACB) * A
        + (weights["C"] / w_ACB) * C_lag
        + (weights["B"] / w_ACB) * B_deep
    )

    # Row 8 (full blend WITHOUT affine/clip): A + B + C + D (D approximated by GBDT anchor — D members not preserved)
    # The plan said: full blend = A + B + C + D. We don't have D-track predictions on disk; approximate D ≈ A.
    pred_full_no_postproc = (
        weights["A"] * A + weights["B"] * B_deep + weights["C"] * C_lag + weights["D"] * A
    )

    # Row 9 (full blend WITH affine/clip = final 0.7628)
    pred_full_postproc = affine_clip(pred_full_no_postproc, menu)

    # Public MAE references (from _local_eval_gate_report.csv / training_menu)
    public_refs = {
        "weather_only_GBDT": None,
        "lag_only": None,
        "lag_plus_residual": None,
        "gbdt_anchor_ext150": 0.8534,
        "gbdt_plus_lag": None,
        "gbdt_plus_lag_plus_cnn": 0.8017,  # +4-way from Table IV
        "gbdt_plus_lag_plus_cnn_plus_ssl": 0.7952,  # +7-way from Table IV
        "full_blend_no_postproc": None,  # not uploaded
        "full_blend_postproc_final": 0.7628,
    }

    rows = [
        ("1. Weather-only single GBDT (best of 6)", best_single_col,
            mae(oof["gbdt_single_best"].to_numpy(), y), public_refs["weather_only_GBDT"]),
        ("2. Lag-only baseline (2215 d lookup)", "pred_lag2215",
            mae(C_lag, y), public_refs["lag_only"]),
        ("3. Lag + per-horizon bias correction", "pred_lag2215 + bias",
            mae(oof_lag_resid, y), public_refs["lag_plus_residual"]),
        ("4. GBDT anchor (6-leg b91+pl blend)", "mean(gbdt_cols)",
            mae(A, y), public_refs["gbdt_anchor_ext150"]),
        ("5. GBDT + lag (A+C renormalized)", "0.444*A + 0.556*C",
            mae(pred_gbdt_lag, y), public_refs["gbdt_plus_lag"]),
        ("6. GBDT + lag + CNN", "weighted A+C+B_cnn",
            mae(pred_gbdt_lag_cnn, y), public_refs["gbdt_plus_lag_plus_cnn"]),
        ("7. GBDT + lag + CNN + SSL (full deep)", "weighted A+C+B",
            mae(pred_gbdt_lag_cnn_ssl, y), public_refs["gbdt_plus_lag_plus_cnn_plus_ssl"]),
        ("8. Full blend, NO affine/clip", "0.2A + 0.45B + 0.25C + 0.1D~A",
            mae(pred_full_no_postproc, y), public_refs["full_blend_no_postproc"]),
        ("9. Full blend + affine + clip (final)", f"affine(shift={menu['postproc']['shift']},scale={menu['postproc']['scale']},clip<={menu['postproc']['clip_max']})",
            mae(pred_full_postproc, y), public_refs["full_blend_postproc_final"]),
    ]

    out_csv = REPORTS / "ablation_9row.csv"
    out_md = REPORTS / "ablation_9row.md"
    df_out = pd.DataFrame(rows, columns=["Configuration", "Formula", "OOF_MAE", "Public_MAE"])
    df_out.to_csv(out_csv, index=False)

    md = ["# Controlled OOF ablation (iter4)\n",
          f"All rows computed on the same OOF tensor ({len(oof)} rows; 6,744 anchors × 5 horizons; 5-fold region-CV).\n",
          "| # | Configuration | OOF MAE | Public MAE | Source |",
          "|---|---|---:|---:|---|"]
    for cfg, formula, m, p in rows:
        pub_str = f"{p:.4f}" if p is not None else "—"
        md.append(f"| {cfg.split('.')[0]} | {cfg.split('. ',1)[1]} | {m:.4f} | {pub_str} | `{formula}` |")
    out_md.write_text("\n".join(md) + "\n")

    print(f"[ablation] wrote {out_csv}", flush=True)
    print(f"[ablation] wrote {out_md}", flush=True)
    print(out_md.read_text(), flush=True)


if __name__ == "__main__":
    main()
