#!/usr/bin/env python3
"""Apply trained stacker to leg test submission CSVs (Track A).

For each leg, load `submissions/submission_<leg>.csv`, build the meta-feature matrix
(per region), apply the per-horizon stacker, clip to [0, 5], write the stacked submission.

Input:
  - models/stacker_<model>.joblib (from train_stacker.py)
  - submission CSVs for each leg referenced in the OOF tensor

Output:
  - submissions/submission_stacker_<model>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS = PROJECT_ROOT / "submissions"
MODELS = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"

LEG_TO_SUBMISSION = {
    "b91_lgbm_s114": "submission_redo_lgbm_seed114.csv",
    "b91_lgbm_s271828": "submission_redo_lgbm_seed271828.csv",
    "b91_hgb_s31415": "submission_redo_hgb_seed31415.csv",
    "pl_lgbm_s114": "submission_round4_pl_lgbm_seed114.csv",
    "pl_lgbm_s271828": "submission_round4_pl_lgbm_seed271828.csv",
    "pl_hgb_s31415": "submission_round4_pl_hgb_seed31415.csv",
    "v2_b91_lgbm_s114": "submission_v2_lgbm_seed114.csv",
    "v2_b91_lgbm_s271828": "submission_v2_lgbm_seed271828.csv",
    "v2_b91_hgb_s31415": "submission_v2_hgb_seed31415.csv",
    "v2_pl_lgbm_s114": "submission_v2_pl_lgbm_seed114.csv",
    "v2_pl_lgbm_s271828": "submission_v2_pl_lgbm_seed271828.csv",
    "v2_pl_hgb_s31415": "submission_v2_pl_hgb_seed31415.csv",
    "deep_b91_s114": "submission_deep_cnn_blackout91_seed114.csv",
    "deep_b91_s271828": "submission_deep_cnn_blackout91_seed271828.csv",
    "deep_pl_s114": "submission_deep_cnn_public_like_seed114.csv",
    "deep_pl_s271828": "submission_deep_cnn_public_like_seed271828.csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ridge", "mlp"], default="ridge")
    p.add_argument("--model-path", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path or (MODELS / f"stacker_{args.model}.joblib"))
    print(f"[info] loading stacker: {model_path}")
    fitted = joblib.load(model_path)
    per_horizon = fitted["per_horizon"]
    leg_cols = fitted["leg_cols"]

    # leg_cols are like "pred_<short>"; map to submission filenames
    leg_shorts = [c.removeprefix("pred_") for c in leg_cols]
    sample_path = PROJECT_ROOT / "data" / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    sample["region_id"] = sample["region_id"].astype(str)
    region_order = sample["region_id"].tolist()
    sample_idx = {r: i for i, r in enumerate(region_order)}

    pred_cols = [f"pred_week{i + 1}" for i in range(5)]

    leg_preds: dict[str, np.ndarray] = {}  # leg_short -> (N_regions, 5)
    for short in leg_shorts:
        sub_name = LEG_TO_SUBMISSION.get(short)
        if sub_name is None:
            raise SystemExit(f"[error] no submission filename registered for leg {short}")
        path = SUBMISSIONS / sub_name
        if not path.exists():
            raise SystemExit(f"[error] missing submission CSV for leg {short}: {path}")
        df = pd.read_csv(path)
        df["region_id"] = df["region_id"].astype(str)
        df = df.set_index("region_id").reindex(region_order).reset_index()
        if df[pred_cols].isna().any().any():
            raise SystemExit(f"[error] {path} missing some regions after reindex")
        leg_preds[short] = df[pred_cols].to_numpy(dtype=np.float64)
        print(f"  loaded leg {short:<25s} mean={leg_preds[short].mean():.4f}  from {sub_name}")

    # Apply per-horizon stacker
    out = np.zeros((len(region_order), 5), dtype=np.float64)
    for h_idx in range(5):
        h = h_idx + 1
        spec = per_horizon[h]
        m = spec["model"]
        te_map: dict[str, float] = spec["region_te"]
        global_te: float = spec["global_te"]
        X_legs = np.stack([leg_preds[short][:, h_idx] for short in leg_shorts], axis=1)  # (N_regions, n_legs)
        leg_mean = X_legs.mean(axis=1, keepdims=True)
        leg_std = X_legs.std(axis=1, keepdims=True)
        region_te = np.array([te_map.get(str(r), global_te) for r in region_order], dtype=np.float64).reshape(-1, 1)
        X_meta = np.concatenate([X_legs, leg_mean, leg_std, region_te], axis=1)
        residual_pred = m.predict(X_meta)
        stacked = leg_mean[:, 0] + residual_pred
        out[:, h_idx] = np.clip(stacked, 0.0, 5.0)

    out_path = Path(args.output or (SUBMISSIONS / f"submission_stacker_{args.model}.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(out, columns=pred_cols)
    out_df.insert(0, "region_id", region_order)
    out_df.to_csv(out_path, index=False)
    print(f"[info] wrote {out_path} (mean {out.mean():.4f}, range [{out.min():.4f}, {out.max():.4f}])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
