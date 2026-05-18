#!/usr/bin/env python3
"""15-leg MLP stacker: 9 GBDT (v1+v2+v2_pl) + 6 deep (cnn/lstm/trans × seeds 114/271828).

Per-horizon MLPRegressor with GroupKFold(region). Outputs raw + shift + extrap sweep.

NOTE on Plan v4 lesson (see memory: mlp-stacker-groupkfold-validation-does-not-transfer-to-public):
the 9-leg MLP scored val 0.310 but public 0.8688. Adding deep legs MAY help generalization if
they bring genuinely uncorrelated signal. MAD≤0.10 vs ext150 (0.8534) is the eligibility gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor

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
    ("deep_cnn_s114", "submission_deep_cnn_s114.csv", "deep_cnn_s114_validation_predictions.csv"),
    ("deep_cnn_s271828", "submission_deep_cnn_s271828.csv", "deep_cnn_s271828_validation_predictions.csv"),
    ("deep_lstm_s114", "submission_deep_lstm_s114.csv", "deep_lstm_s114_validation_predictions.csv"),
    ("deep_lstm_s271828", "submission_deep_lstm_s271828.csv", "deep_lstm_s271828_validation_predictions.csv"),
    ("deep_trans_s114", "submission_deep_trans_s114.csv", "deep_trans_s114_validation_predictions.csv"),
    ("deep_trans_s271828", "submission_deep_trans_s271828.csv", "deep_trans_s271828_validation_predictions.csv"),
]
PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]
HIDDEN = (64, 32)
MAX_ITER = 500
ALPHA = 1e-3
SEED = 42


def main() -> int:
    print("[step 1] join 15-leg OOF tensor")
    base = None
    available_legs = []
    for tag, sub_name, valpred_name in LEGS:
        p = REPORTS / valpred_name
        if not p.exists():
            print(f"  [skip] {valpred_name} missing — leg unavailable")
            continue
        df = pd.read_csv(p)
        df["region_id"] = df["region_id"].astype(str)
        df = df.rename(columns={"pred_final_calibrated": f"pred_{tag}"})[["row_index", "region_id", "horizon", "y_true", f"pred_{tag}"]]
        base = df if base is None else base.merge(df.drop(columns=["y_true"]), on=["row_index", "region_id", "horizon"], how="inner")
        available_legs.append((tag, sub_name, valpred_name))
    leg_cols = [f"pred_{tag}" for tag, _, _ in available_legs]
    print(f"  available legs: {len(available_legs)}; OOF rows: {len(base)}")

    print("\n[step 2] train MLP per horizon")
    per_horizon = {}
    overall = 0.0
    overall_legmean = 0.0
    for h in [1, 2, 3, 4, 5]:
        sub = base[base["horizon"] == h].reset_index(drop=True)
        X = sub[leg_cols].to_numpy(dtype=np.float64)
        y = sub["y_true"].to_numpy(dtype=np.float64)
        regions = sub["region_id"].to_numpy()
        leg_mean = X.mean(axis=1, keepdims=True)
        leg_std = X.std(axis=1, keepdims=True)
        baseline_mae = float(np.abs(leg_mean[:, 0] - y).mean())

        gkf = GroupKFold(n_splits=5)
        oof_pred = np.zeros(len(sub))
        for tr_idx, va_idx in gkf.split(X, y, groups=regions):
            X_tr = np.concatenate([X[tr_idx], leg_mean[tr_idx], leg_std[tr_idx]], axis=1)
            X_va = np.concatenate([X[va_idx], leg_mean[va_idx], leg_std[va_idx]], axis=1)
            m = MLPRegressor(hidden_layer_sizes=HIDDEN, max_iter=MAX_ITER, early_stopping=True, alpha=ALPHA, random_state=SEED)
            m.fit(X_tr, y[tr_idx] - leg_mean[tr_idx, 0])
            oof_pred[va_idx] = leg_mean[va_idx, 0] + m.predict(X_va)
        oof_mae = float(np.abs(oof_pred - y).mean())

        X_all = np.concatenate([X, leg_mean, leg_std], axis=1)
        final = MLPRegressor(hidden_layer_sizes=HIDDEN, max_iter=MAX_ITER, early_stopping=True, alpha=ALPHA, random_state=SEED)
        final.fit(X_all, y - leg_mean[:, 0])
        per_horizon[h] = final
        overall += oof_mae
        overall_legmean += baseline_mae
        print(f"  H{h}: leg-mean {baseline_mae:.4f} -> stacker {oof_mae:.4f}  (Δ {baseline_mae - oof_mae:+.4f})")
    print(f"\n[info] 15-leg MLP overall val MAE: {overall / 5:.4f} (leg-mean baseline {overall_legmean / 5:.4f})")

    print("\n[step 3] apply to test submissions")
    sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
    sample["region_id"] = sample["region_id"].astype(str)
    region_order = sample["region_id"].tolist()
    leg_test = {}
    for tag, sub_name, _ in available_legs:
        sp = SUBMISSIONS / sub_name
        if not sp.exists():
            print(f"  [error] test sub {sub_name} missing — abort")
            return 1
        df = pd.read_csv(sp)
        df["region_id"] = df["region_id"].astype(str)
        df = df.set_index("region_id").reindex(region_order).reset_index()
        leg_test[tag] = df[PRED_COLS].to_numpy(dtype=np.float64)
        print(f"  {tag:<22s} test_mean={leg_test[tag].mean():.4f}")

    out = np.zeros((len(region_order), 5))
    for h_idx in range(5):
        h = h_idx + 1
        X = np.stack([leg_test[t][:, h_idx] for t, _, _ in available_legs], axis=1)
        leg_mean = X.mean(axis=1, keepdims=True)
        leg_std = X.std(axis=1, keepdims=True)
        X_meta = np.concatenate([X, leg_mean, leg_std], axis=1)
        residual = per_horizon[h].predict(X_meta)
        out[:, h_idx] = np.clip(leg_mean[:, 0] + residual, 0.0, 5.0)
    print(f"\n[info] 15-leg raw test mean: {out.mean():.4f}")

    def write(arr, name):
        df = pd.DataFrame(arr, columns=PRED_COLS)
        df.insert(0, "region_id", region_order)
        p = SUBMISSIONS / f"submission_stacker_15leg_{name}.csv"
        df.to_csv(p, index=False)
        print(f"  wrote {p.name}  mean={arr.mean():.4f}")

    pb_path = SUBMISSIONS / "submission_redo_extrapolate_150_mean12334.csv"
    pb = None
    if pb_path.exists():
        df = pd.read_csv(pb_path)
        df["region_id"] = df["region_id"].astype(str)
        df = df.set_index("region_id").reindex(region_order).reset_index()
        pb = df[PRED_COLS].to_numpy(dtype=np.float64)

    write(out, "raw")
    for shift in [0.20, 0.25, 0.27, 0.30]:
        shifted = np.clip(out + shift, 0.0, 5.0)
        write(shifted, f"shift{int(shift * 100):02d}")
        for factor in [1.3, 1.5]:
            m = shifted.mean()
            extrap = np.clip(m + factor * (shifted - m), 0.0, 5.0)
            write(extrap, f"shift{int(shift * 100):02d}_x{int(factor * 100):03d}")

    if pb is not None:
        print("\n[info] MAD vs public best (ext150) — gate 0.10:")
        for fname in sorted(SUBMISSIONS.glob("submission_stacker_15leg_*.csv")):
            df = pd.read_csv(fname); df["region_id"] = df["region_id"].astype(str)
            df = df.set_index("region_id").reindex(region_order).reset_index()
            arr = df[PRED_COLS].to_numpy(dtype=np.float64)
            mad = float(np.abs(arr - pb).mean())
            status = "PASS" if mad <= 0.10 else "FAIL"
            print(f"  {fname.name:<55s}  mad={mad:.4f}  [{status}]")

    PROJECT_ROOT.joinpath("models").mkdir(exist_ok=True)
    joblib.dump({"per_horizon": per_horizon, "legs": available_legs}, PROJECT_ROOT / "models" / "stacker_15leg_mlp.joblib")
    rep = {"model": "mlp", "legs": [t for t, _, _ in available_legs], "overall_oof_mae": overall / 5, "leg_mean_baseline": overall_legmean / 5}
    with (REPORTS / "stacker_15leg_mlp.json").open("w") as f:
        json.dump(rep, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
