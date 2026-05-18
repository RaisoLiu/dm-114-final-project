#!/usr/bin/env python3
"""MLP stacker on 3 blackout91 legs — train per-horizon, apply to test, extrapolate.

OOF eval shows MLP (hidden 32,16) beats Ridge by ~0.014 more val MAE improvement
(MLP val MAE ~0.342 vs Ridge ~0.357 vs leg-mean 0.376).
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
    ("b91_lgbm_s114", "submission_redo_lgbm_seed114.csv"),
    ("b91_lgbm_s271828", "submission_redo_lgbm_seed271828.csv"),
    ("b91_hgb_s31415", "submission_redo_hgb_seed31415.csv"),
]
PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]
HIDDEN = (32, 16)
MAX_ITER = 200
ALPHA = 1e-3
EXTRAPOLATE_FACTORS = [1.0, 1.3, 1.5]
TARGET_MEAN = 1.2334  # matches extrapolate_150 anchor


def main() -> int:
    print("[step 1] loading OOF tensor")
    oof = pd.read_csv(REPORTS / "oof_tensor.csv")
    oof["region_id"] = oof["region_id"].astype(str)
    leg_cols = [f"pred_{short}" for short, _ in LEGS]
    keep = ["row_index", "region_id", "horizon", "y_true"] + leg_cols
    oof = oof[keep].dropna()
    print(f"  OOF rows: {len(oof)}")

    print("[step 2] CV train MLP per-horizon")
    per_horizon = {}
    for h in [1, 2, 3, 4, 5]:
        sub = oof[oof["horizon"] == h].reset_index(drop=True)
        X = sub[leg_cols].to_numpy(dtype=np.float64)
        y = sub["y_true"].to_numpy(dtype=np.float64)
        regions = sub["region_id"].to_numpy()
        leg_mean = X.mean(axis=1, keepdims=True)
        leg_std = X.std(axis=1, keepdims=True)
        base_mae = float(np.abs(leg_mean[:, 0] - y).mean())

        gkf = GroupKFold(n_splits=5)
        oof_pred = np.zeros(len(sub))
        fold_maes = []
        for tr_idx, va_idx in gkf.split(X, y, groups=regions):
            X_tr = np.concatenate([X[tr_idx], leg_mean[tr_idx], leg_std[tr_idx]], axis=1)
            X_va = np.concatenate([X[va_idx], leg_mean[va_idx], leg_std[va_idx]], axis=1)
            m = MLPRegressor(hidden_layer_sizes=HIDDEN, max_iter=MAX_ITER, early_stopping=True,
                             alpha=ALPHA, random_state=42)
            m.fit(X_tr, y[tr_idx] - leg_mean[tr_idx, 0])
            oof_pred[va_idx] = leg_mean[va_idx, 0] + m.predict(X_va)
            fold_maes.append(float(np.abs(oof_pred[va_idx] - y[va_idx]).mean()))
        oof_mae = float(np.abs(oof_pred - y).mean())

        # Refit on all
        X_all = np.concatenate([X, leg_mean, leg_std], axis=1)
        final = MLPRegressor(hidden_layer_sizes=HIDDEN, max_iter=MAX_ITER, early_stopping=True,
                             alpha=ALPHA, random_state=42)
        final.fit(X_all, y - leg_mean[:, 0])

        per_horizon[h] = {
            "baseline_mean_mae": base_mae,
            "stacker_oof_mae": oof_mae,
            "improvement": base_mae - oof_mae,
            "model": final,
            "fold_maes": fold_maes,
        }
        print(f"  H{h}: leg-mean {base_mae:.4f} -> mlp {oof_mae:.4f}  (Δ {base_mae - oof_mae:+.4f})")

    overall_stacker = float(np.mean([per_horizon[h]["stacker_oof_mae"] for h in [1, 2, 3, 4, 5]]))
    overall_baseline = float(np.mean([per_horizon[h]["baseline_mean_mae"] for h in [1, 2, 3, 4, 5]]))
    print(f"\n[info] overall: baseline {overall_baseline:.4f} -> mlp {overall_stacker:.4f}  (Δ {overall_baseline - overall_stacker:+.4f})")

    print("[step 3] load test leg CSVs and apply MLP")
    sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
    sample["region_id"] = sample["region_id"].astype(str)
    region_order = sample["region_id"].tolist()
    leg_test = {}
    for short, fname in LEGS:
        df = pd.read_csv(SUBMISSIONS / fname)
        df["region_id"] = df["region_id"].astype(str)
        df = df.set_index("region_id").reindex(region_order).reset_index()
        leg_test[short] = df[PRED_COLS].to_numpy(dtype=np.float64)
        print(f"  {short:<20s} mean={leg_test[short].mean():.4f}")

    out = np.zeros((len(region_order), 5), dtype=np.float64)
    for h_idx in range(5):
        h = h_idx + 1
        X = np.stack([leg_test[short][:, h_idx] for short, _ in LEGS], axis=1)
        leg_mean = X.mean(axis=1, keepdims=True)
        leg_std = X.std(axis=1, keepdims=True)
        X_meta = np.concatenate([X, leg_mean, leg_std], axis=1)
        residual = per_horizon[h]["model"].predict(X_meta)
        out[:, h_idx] = np.clip(leg_mean[:, 0] + residual, 0.0, 5.0)

    print(f"\n[info] raw MLP test mean: {out.mean():.4f}")

    # Save raw + shifted + extrapolated variants
    def write(arr, name):
        df = pd.DataFrame(arr, columns=PRED_COLS)
        df.insert(0, "region_id", region_order)
        p = SUBMISSIONS / f"submission_stacker_b91_mlp_{name}.csv"
        df.to_csv(p, index=False)
        print(f"  wrote {p.name}  mean={arr.mean():.4f}  range=[{arr.min():.4f}, {arr.max():.4f}]")
        return p

    write(out, "raw")
    # Find good shift to match target mean
    for shift in [0.20, 0.25, 0.27, 0.30]:
        shifted = np.clip(out + shift, 0.0, 5.0)
        write(shifted, f"shift{int(shift*100):02d}")
        # extrapolate at this shift
        for factor in EXTRAPOLATE_FACTORS:
            mean = shifted.mean()
            extrap = np.clip(mean + factor * (shifted - mean), 0.0, 5.0)
            write(extrap, f"shift{int(shift*100):02d}_x{int(factor*100):03d}")

    joblib.dump({"per_horizon": per_horizon, "legs": LEGS}, PROJECT_ROOT / "models" / "stacker_b91_mlp.joblib")

    rep = {
        "model": "mlp",
        "hidden": HIDDEN,
        "legs": [s for s, _ in LEGS],
        "per_horizon": {int(h): {k: (v if k != "model" else None) for k, v in d.items()} for h, d in per_horizon.items()},
        "overall_baseline_mae": overall_baseline,
        "overall_stacker_mae": overall_stacker,
        "improvement": overall_baseline - overall_stacker,
    }
    with (REPORTS / "stacker_b91_mlp.json").open("w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(f"\n[info] wrote {REPORTS / 'stacker_b91_mlp.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
