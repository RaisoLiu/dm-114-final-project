#!/usr/bin/env python3
"""Build the OOF tensor for Track A (stacking).

Joins all `*_validation_predictions.csv` files in reports/ on (row_index, region_id, horizon),
producing a single CSV where each row is (anchor, region, horizon) and the columns are
the predictions from each leg + y_true.

Output: reports/oof_tensor.csv

Usage:
  PYTHONPATH=src .venv/bin/python scripts/build_oof_tensor.py [--include-pattern PAT]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
DEFAULT_INCLUDE = [
    "redo_lgbm_seed114_validation_predictions.csv",
    "redo_lgbm_seed271828_validation_predictions.csv",
    "redo_hgb_seed31415_validation_predictions.csv",
    "round4_pl_lgbm_seed114_validation_predictions.csv",
    "round4_pl_lgbm_seed271828_validation_predictions.csv",
    "round4_pl_hgb_seed31415_validation_predictions.csv",
    "v2_lgbm_seed114_validation_predictions.csv",
    "v2_lgbm_seed271828_validation_predictions.csv",
    "v2_hgb_seed31415_validation_predictions.csv",
    "v2_pl_lgbm_seed114_validation_predictions.csv",
    "v2_pl_lgbm_seed271828_validation_predictions.csv",
    "v2_pl_hgb_seed31415_validation_predictions.csv",
    "deep_cnn_blackout91_seed114_validation_predictions.csv",
    "deep_cnn_blackout91_seed271828_validation_predictions.csv",
    "deep_cnn_public_like_seed114_validation_predictions.csv",
    "deep_cnn_public_like_seed271828_validation_predictions.csv",
    "deep_cnn_s114_validation_predictions.csv",
    "deep_cnn_s271828_validation_predictions.csv",
    "deep_lstm_s114_validation_predictions.csv",
    "deep_lstm_s271828_validation_predictions.csv",
    "deep_trans_s114_validation_predictions.csv",
    "deep_trans_s271828_validation_predictions.csv",
]
LEG_NAMES = {
    "redo_lgbm_seed114": "b91_lgbm_s114",
    "redo_lgbm_seed271828": "b91_lgbm_s271828",
    "redo_hgb_seed31415": "b91_hgb_s31415",
    "round4_pl_lgbm_seed114": "pl_lgbm_s114",
    "round4_pl_lgbm_seed271828": "pl_lgbm_s271828",
    "round4_pl_hgb_seed31415": "pl_hgb_s31415",
    "v2_lgbm_seed114": "v2_b91_lgbm_s114",
    "v2_lgbm_seed271828": "v2_b91_lgbm_s271828",
    "v2_hgb_seed31415": "v2_b91_hgb_s31415",
    "v2_pl_lgbm_seed114": "v2_pl_lgbm_s114",
    "v2_pl_lgbm_seed271828": "v2_pl_lgbm_s271828",
    "v2_pl_hgb_seed31415": "v2_pl_hgb_s31415",
    "deep_cnn_blackout91_seed114": "deep_b91_s114",
    "deep_cnn_blackout91_seed271828": "deep_b91_s271828",
    "deep_cnn_public_like_seed114": "deep_pl_s114",
    "deep_cnn_public_like_seed271828": "deep_pl_s271828",
    "deep_cnn_s114": "deep_cnn_s114",
    "deep_cnn_s271828": "deep_cnn_s271828",
    "deep_lstm_s114": "deep_lstm_s114",
    "deep_lstm_s271828": "deep_lstm_s271828",
    "deep_trans_s114": "deep_trans_s114",
    "deep_trans_s271828": "deep_trans_s271828",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(REPORTS / "oof_tensor.csv"))
    p.add_argument("--report", default=str(REPORTS / "oof_tensor.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[info] looking for validation predictions in {REPORTS}/")
    available: dict[str, Path] = {}
    for name in DEFAULT_INCLUDE:
        p = REPORTS / name
        if p.exists():
            leg_key = name.replace("_validation_predictions.csv", "")
            short = LEG_NAMES.get(leg_key, leg_key)
            available[short] = p
            print(f"  found {short:<25s} -> {p.name}")
    if not available:
        print("[error] no validation prediction CSVs found")
        return 1

    print(f"[info] joining {len(available)} legs ...")
    base: pd.DataFrame | None = None
    for short, path in available.items():
        df = pd.read_csv(path)
        df["region_id"] = df["region_id"].astype(str)
        df = df.rename(columns={"pred_final_calibrated": f"pred_{short}"})
        keep = ["row_index", "region_id", "horizon", "y_true", f"pred_{short}"]
        df = df[keep]
        if base is None:
            base = df
        else:
            base = base.merge(df.drop(columns=["y_true"]), on=["row_index", "region_id", "horizon"], how="inner")
    print(f"[info] joined OOF tensor: {len(base)} rows × {len(base.columns)} cols")
    print(f"[info] columns: {list(base.columns)}")
    # Sanity: count distinct anchors / regions / horizons
    n_anchors = base["row_index"].nunique()
    n_regions = base["region_id"].nunique()
    n_horizons = base["horizon"].nunique()
    print(f"[info] {n_anchors} anchors × {n_regions} regions × {n_horizons} horizons")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(out, index=False)
    print(f"[info] wrote {out}")

    # Per-leg validation MAE
    rep: dict = {"legs": {}, "n_rows": len(base), "n_anchors": int(n_anchors), "n_regions": int(n_regions), "n_horizons": int(n_horizons)}
    for short in available:
        col = f"pred_{short}"
        if col not in base.columns:
            continue
        mae = float(np.mean(np.abs(base[col].to_numpy() - base["y_true"].to_numpy())))
        rep["legs"][short] = {"validation_mae": mae, "source": str(available[short])}
        print(f"  leg {short:<25s} val MAE: {mae:.4f}")

    import json
    with Path(args.report).open("w") as f:
        json.dump(rep, f, indent=2)
    print(f"[info] wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
