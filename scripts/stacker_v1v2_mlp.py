#!/usr/bin/env python3
"""Full pipeline: build v1+v2 6-leg OOF tensor, train MLP stacker, apply to test.

Legs (all gap_mode=blackout91, valid-deltas={728,735,742}):
  v1: redo_lgbm_seed114, redo_lgbm_seed271828, redo_hgb_seed31415
  v2: v2_lgbm_seed114, v2_lgbm_seed271828, v2_hgb_seed31415 (with EXTRA_SIGNAL_FEATURES)

OOF analysis (earlier inline test): v1+v2 6-leg MLP val MAE ~0.328 (vs leg-mean 0.367, vs v1-only MLP 0.342).
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
]
PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]
HIDDEN = (32, 16)
MAX_ITER = 200
ALPHA = 1e-3
SEED = 42
TARGET_MEAN = 1.2334  # matches pb30/extrapolate_150 anchor


def main() -> int:
    print("[step 1] join 6-leg OOF tensor")
    base = None
    for tag, _, valpred_name in LEGS:
        p = REPORTS / valpred_name
        df = pd.read_csv(p)
        df["region_id"] = df["region_id"].astype(str)
        df = df.rename(columns={"pred_final_calibrated": f"pred_{tag}"})[["row_index", "region_id", "horizon", "y_true", f"pred_{tag}"]]
        if base is None:
            base = df
        else:
            base = base.merge(df.drop(columns=["y_true"]), on=["row_index", "region_id", "horizon"], how="inner")
    leg_cols = [f"pred_{tag}" for tag, _, _ in LEGS]
    print(f"  OOF rows: {len(base)}; legs: {leg_cols}")
    for col in leg_cols:
        mae = float(np.abs(base[col] - base["y_true"]).mean())
        print(f"  {col}: val MAE {mae:.4f}")

    print("\n[step 2] CV-train MLP stacker per horizon")
    per_horizon = {}
    overall_oof = 0.0
    overall_leg_mean_mae = 0.0
    for h in [1, 2, 3, 4, 5]:
        sub = base[base["horizon"] == h].reset_index(drop=True)
        X = sub[leg_cols].to_numpy(dtype=np.float64)
        y = sub["y_true"].to_numpy(dtype=np.float64)
        regions = sub["region_id"].to_numpy()
        leg_mean = X.mean(axis=1, keepdims=True)
        leg_std = X.std(axis=1, keepdims=True)
        base_mae = float(np.abs(leg_mean[:, 0] - y).mean())

        gkf = GroupKFold(n_splits=5)
        oof_pred = np.zeros(len(sub))
        for tr_idx, va_idx in gkf.split(X, y, groups=regions):
            X_tr = np.concatenate([X[tr_idx], leg_mean[tr_idx], leg_std[tr_idx]], axis=1)
            X_va = np.concatenate([X[va_idx], leg_mean[va_idx], leg_std[va_idx]], axis=1)
            m = MLPRegressor(hidden_layer_sizes=HIDDEN, max_iter=MAX_ITER, early_stopping=True,
                             alpha=ALPHA, random_state=SEED)
            m.fit(X_tr, y[tr_idx] - leg_mean[tr_idx, 0])
            oof_pred[va_idx] = leg_mean[va_idx, 0] + m.predict(X_va)
        oof_mae = float(np.abs(oof_pred - y).mean())

        # Refit on all
        X_all = np.concatenate([X, leg_mean, leg_std], axis=1)
        final = MLPRegressor(hidden_layer_sizes=HIDDEN, max_iter=MAX_ITER, early_stopping=True,
                             alpha=ALPHA, random_state=SEED)
        final.fit(X_all, y - leg_mean[:, 0])

        per_horizon[h] = {
            "baseline_mean_mae": base_mae,
            "stacker_oof_mae": oof_mae,
            "improvement": base_mae - oof_mae,
            "model": final,
        }
        overall_oof += oof_mae
        overall_leg_mean_mae += base_mae
        print(f"  H{h}: leg-mean {base_mae:.4f} -> stacker {oof_mae:.4f}  (Δ {base_mae - oof_mae:+.4f})")

    overall_oof /= 5
    overall_leg_mean_mae /= 5
    print(f"\n[info] overall: leg-mean {overall_leg_mean_mae:.4f} -> stacker {overall_oof:.4f}  (Δ {overall_leg_mean_mae - overall_oof:+.4f})")

    print("\n[step 3] load all 6 leg test submission CSVs")
    sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
    sample["region_id"] = sample["region_id"].astype(str)
    region_order = sample["region_id"].tolist()
    leg_test = {}
    for tag, sub_name, _ in LEGS:
        df = pd.read_csv(SUBMISSIONS / sub_name)
        df["region_id"] = df["region_id"].astype(str)
        df = df.set_index("region_id").reindex(region_order).reset_index()
        if df[PRED_COLS].isna().any().any():
            raise SystemExit(f"[error] {sub_name} missing some regions after reindex")
        leg_test[tag] = df[PRED_COLS].to_numpy(dtype=np.float64)
        print(f"  loaded {tag:<20s} mean={leg_test[tag].mean():.4f}")

    print("\n[step 4] apply stacker per horizon")
    out = np.zeros((len(region_order), 5), dtype=np.float64)
    for h_idx in range(5):
        h = h_idx + 1
        X = np.stack([leg_test[tag][:, h_idx] for tag, _, _ in LEGS], axis=1)
        leg_mean = X.mean(axis=1, keepdims=True)
        leg_std = X.std(axis=1, keepdims=True)
        X_meta = np.concatenate([X, leg_mean, leg_std], axis=1)
        residual = per_horizon[h]["model"].predict(X_meta)
        out[:, h_idx] = np.clip(leg_mean[:, 0] + residual, 0.0, 5.0)

    print(f"\n[info] raw stacker output mean: {out.mean():.4f}")

    def write(arr, name):
        df = pd.DataFrame(arr, columns=PRED_COLS)
        df.insert(0, "region_id", region_order)
        p = SUBMISSIONS / f"submission_stacker_v1v2_mlp_{name}.csv"
        df.to_csv(p, index=False)
        print(f"  wrote {p.name}  mean={arr.mean():.4f}  range=[{arr.min():.4f}, {arr.max():.4f}]")
        return p

    write(out, "raw")
    # Shift sweep
    for shift in [0.20, 0.25, 0.27, 0.30]:
        shifted = np.clip(out + shift, 0.0, 5.0)
        write(shifted, f"shift{int(shift*100):02d}")
        for factor in [1.3, 1.5]:
            mean = shifted.mean()
            extrap = np.clip(mean + factor * (shifted - mean), 0.0, 5.0)
            write(extrap, f"shift{int(shift*100):02d}_x{int(factor*100):03d}")

    # Save model
    joblib.dump({"per_horizon": per_horizon, "legs": LEGS}, PROJECT_ROOT / "models" / "stacker_v1v2_mlp.joblib")

    rep = {
        "model": "mlp",
        "legs": [t for t, _, _ in LEGS],
        "hidden": HIDDEN,
        "per_horizon": {int(h): {k: (v if k != "model" else None) for k, v in d.items()} for h, d in per_horizon.items()},
        "overall_baseline_mae": overall_leg_mean_mae,
        "overall_stacker_mae": overall_oof,
        "improvement": overall_leg_mean_mae - overall_oof,
    }
    with (REPORTS / "stacker_v1v2_mlp.json").open("w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(f"\n[info] wrote {REPORTS / 'stacker_v1v2_mlp.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
