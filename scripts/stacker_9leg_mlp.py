#!/usr/bin/env python3
"""9-leg MLP stacker: v1 + v2 + v2_pl. Produces test submissions with shift+extrap variants.

Pre-validated OOF tensor val MAE: 0.3130 (vs 6-leg v1+v2 0.3279, single-best 0.376).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
SUBMISSIONS = PROJECT_ROOT / "submissions"

LEGS = [
    ("v1_lgbm_s114", "submission_redo_lgbm_seed114.csv", "redo_lgbm_seed114_validation_predictions.csv"),
    ("v1_lgbm_s271828", "submission_redo_lgbm_seed271828.csv", "redo_lgbm_seed271828_validation_predictions.csv"),
    ("v1_hgb_s31415", "submission_redo_hgb_seed31415.csv", "redo_hgb_seed31415_validation_predictions.csv"),
    ("v2_lgbm_s114", "submission_v2_lgbm_seed114.csv", "v2_lgbm_seed114_validation_predictions.csv"),
    ("v2_lgbm_s271828", "submission_v2_lgbm_seed271828.csv", "v2_lgbm_seed271828_validation_predictions.csv"),
    ("v2_hgb_s31415", "submission_v2_hgb_seed31415.csv", "v2_hgb_seed31415_validation_predictions.csv"),
    ("v2pl_lgbm_s114", "submission_v2_pl_lgbm_seed114.csv", "v2_pl_lgbm_seed114_validation_predictions.csv"),
    ("v2pl_lgbm_s271828", "submission_v2_pl_lgbm_seed271828.csv", "v2_pl_lgbm_seed271828_validation_predictions.csv"),
    ("v2pl_hgb_s31415", "submission_v2_pl_hgb_seed31415.csv", "v2_pl_hgb_seed31415_validation_predictions.csv"),
]
PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]
HIDDEN = (32, 16)
TARGET_MEAN = 1.2334


def main() -> int:
    print("[step 1] join 9-leg OOF tensor")
    base = None
    for tag, _, valpred_name in LEGS:
        df = pd.read_csv(REPORTS / valpred_name)
        df["region_id"] = df["region_id"].astype(str)
        df = df.rename(columns={"pred_final_calibrated": f"pred_{tag}"})[["row_index", "region_id", "horizon", "y_true", f"pred_{tag}"]]
        base = df if base is None else base.merge(df.drop(columns=["y_true"]), on=["row_index", "region_id", "horizon"], how="inner")
    leg_cols = [f"pred_{tag}" for tag, _, _ in LEGS]
    print(f"  OOF rows: {len(base)}, legs: {len(leg_cols)}")

    print("\n[step 2] train MLP per horizon")
    per_horizon = {}
    overall = 0.0
    for h in [1, 2, 3, 4, 5]:
        sub = base[base["horizon"] == h].reset_index(drop=True)
        X = sub[leg_cols].to_numpy(dtype=np.float64)
        y = sub["y_true"].to_numpy(dtype=np.float64)
        regions = sub["region_id"].to_numpy()
        leg_mean = X.mean(axis=1, keepdims=True)
        leg_std = X.std(axis=1, keepdims=True)
        gkf = GroupKFold(n_splits=5)
        oof_pred = np.zeros(len(sub))
        for tr_idx, va_idx in gkf.split(X, y, groups=regions):
            X_tr = np.concatenate([X[tr_idx], leg_mean[tr_idx], leg_std[tr_idx]], axis=1)
            X_va = np.concatenate([X[va_idx], leg_mean[va_idx], leg_std[va_idx]], axis=1)
            m = MLPRegressor(hidden_layer_sizes=HIDDEN, max_iter=200, early_stopping=True, alpha=1e-3, random_state=42)
            m.fit(X_tr, y[tr_idx] - leg_mean[tr_idx, 0])
            oof_pred[va_idx] = leg_mean[va_idx, 0] + m.predict(X_va)
        oof_mae = float(np.abs(oof_pred - y).mean())
        # Refit on all
        X_all = np.concatenate([X, leg_mean, leg_std], axis=1)
        final = MLPRegressor(hidden_layer_sizes=HIDDEN, max_iter=200, early_stopping=True, alpha=1e-3, random_state=42)
        final.fit(X_all, y - leg_mean[:, 0])
        per_horizon[h] = final
        overall += oof_mae
        print(f"  H{h}: stacker val MAE {oof_mae:.4f}")
    print(f"\n[info] overall 9-leg MLP val MAE: {overall/5:.4f}")

    print("\n[step 3] load test submissions and apply")
    sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
    sample["region_id"] = sample["region_id"].astype(str)
    region_order = sample["region_id"].tolist()
    leg_test = {}
    for tag, sub_name, _ in LEGS:
        df = pd.read_csv(SUBMISSIONS / sub_name)
        df["region_id"] = df["region_id"].astype(str)
        df = df.set_index("region_id").reindex(region_order).reset_index()
        leg_test[tag] = df[PRED_COLS].to_numpy(dtype=np.float64)
        print(f"  {tag:<22s} mean={leg_test[tag].mean():.4f}")

    out = np.zeros((len(region_order), 5))
    for h_idx in range(5):
        h = h_idx + 1
        X = np.stack([leg_test[t][:, h_idx] for t, _, _ in LEGS], axis=1)
        leg_mean = X.mean(axis=1, keepdims=True)
        leg_std = X.std(axis=1, keepdims=True)
        X_meta = np.concatenate([X, leg_mean, leg_std], axis=1)
        residual = per_horizon[h].predict(X_meta)
        out[:, h_idx] = np.clip(leg_mean[:, 0] + residual, 0, 5)

    print(f"\n[info] 9-leg stacker raw test mean: {out.mean():.4f}")

    def write(arr, name):
        df = pd.DataFrame(arr, columns=PRED_COLS); df.insert(0, "region_id", region_order)
        p = SUBMISSIONS / f"submission_stacker_9leg_{name}.csv"
        df.to_csv(p, index=False)
        print(f"  wrote {p.name}  mean={arr.mean():.4f}")

    write(out, "raw")
    for shift in [0.25, 0.27, 0.30, 0.33]:
        shifted = np.clip(out + shift, 0, 5)
        write(shifted, f"shift{int(shift*100):02d}")
        for factor in [1.3, 1.5]:
            mean = shifted.mean()
            extrap = np.clip(mean + factor * (shifted - mean), 0, 5)
            write(extrap, f"shift{int(shift*100):02d}_x{int(factor*100):03d}")

    joblib.dump({"per_horizon": per_horizon, "legs": LEGS}, PROJECT_ROOT / "models" / "stacker_9leg_mlp.joblib")
    rep = {"model": "mlp", "legs": [t for t, _, _ in LEGS], "overall_oof_mae": overall/5}
    with (REPORTS / "stacker_9leg_mlp.json").open("w") as f:
        json.dump(rep, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
