#!/usr/bin/env python3
"""Train a per-horizon stacker on the OOF tensor (Track A).

Uses GroupKFold(region_id) for unbiased per-region weight learning.
Supports Ridge (default, robust) and MLPRegressor (more expressive).

Input: reports/oof_tensor.csv (from scripts/build_oof_tensor.py)
Output:
  - models/stacker_<model>.joblib (per-horizon fitted models + target encoder)
  - reports/stacker_<model>.json (per-fold and aggregate MAE)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
MODELS = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(REPORTS / "oof_tensor.csv"))
    p.add_argument("--model", choices=["ridge", "mlp"], default="ridge")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha")
    p.add_argument("--hidden", default="64,32", help="MLP hidden layers comma-separated")
    p.add_argument("--mlp-max-iter", type=int, default=400)
    p.add_argument("--mlp-alpha", type=float, default=1e-3)
    p.add_argument("--output-model", default=None)
    p.add_argument("--output-report", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    df["region_id"] = df["region_id"].astype(str)
    leg_cols = [c for c in df.columns if c.startswith("pred_")]
    print(f"[info] {len(df)} OOF rows, {len(leg_cols)} legs: {leg_cols}")

    # Target encoding for region: mean residual per region (computed across train fold only inside the loop)
    horizons = sorted(df["horizon"].unique())
    print(f"[info] horizons: {horizons}")

    # Per-horizon results
    per_horizon: dict = {}
    fitted_models: dict[int, dict] = {}

    for h in horizons:
        sub = df[df["horizon"] == h].reset_index(drop=True)
        X_legs = sub[leg_cols].to_numpy(dtype=np.float64)
        y = sub["y_true"].to_numpy(dtype=np.float64)
        regions = sub["region_id"].to_numpy()
        # Mean and std of leg predictions per row (meta-features)
        leg_mean = X_legs.mean(axis=1, keepdims=True)
        leg_std = X_legs.std(axis=1, keepdims=True)
        # Baseline: leg mean alone
        base_mae = float(np.abs(leg_mean[:, 0] - y).mean())

        gkf = GroupKFold(n_splits=args.n_folds)
        oof_pred = np.zeros(len(sub), dtype=np.float64)
        fold_maes: list[float] = []
        region_te_map_per_fold: list[dict[str, float]] = []
        last_model = None
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_legs, y, groups=regions)):
            # Region target encoding from training fold only
            tr_residual = y[tr_idx] - leg_mean[tr_idx, 0]
            tr_regions = regions[tr_idx]
            global_mean = float(tr_residual.mean())
            te_map = {}
            for r in np.unique(tr_regions):
                te_map[str(r)] = float(tr_residual[tr_regions == r].mean())
            region_te_map_per_fold.append(te_map)

            def region_te(rs):
                return np.array([te_map.get(str(r), global_mean) for r in rs], dtype=np.float64).reshape(-1, 1)

            X_tr = np.concatenate([
                X_legs[tr_idx],
                leg_mean[tr_idx],
                leg_std[tr_idx],
                region_te(tr_regions),
            ], axis=1)
            X_va = np.concatenate([
                X_legs[va_idx],
                leg_mean[va_idx],
                leg_std[va_idx],
                region_te(regions[va_idx]),
            ], axis=1)
            # Residual target so stacker fits the correction
            y_tr_residual = y[tr_idx] - leg_mean[tr_idx, 0]
            if args.model == "ridge":
                m = Ridge(alpha=args.alpha, random_state=args.seed)
            else:
                hidden = tuple(int(x) for x in args.hidden.split(","))
                m = MLPRegressor(
                    hidden_layer_sizes=hidden,
                    activation="relu",
                    solver="adam",
                    alpha=args.mlp_alpha,
                    max_iter=args.mlp_max_iter,
                    early_stopping=True,
                    random_state=args.seed,
                )
            m.fit(X_tr, y_tr_residual)
            va_residual_pred = m.predict(X_va)
            oof_pred[va_idx] = leg_mean[va_idx, 0] + va_residual_pred
            fold_mae = float(np.abs(oof_pred[va_idx] - y[va_idx]).mean())
            fold_maes.append(fold_mae)
            last_model = m

        oof_mae = float(np.abs(oof_pred - y).mean())
        # Final refit on all data (for test-time application)
        global_mean = float((y - leg_mean[:, 0]).mean())
        full_te_map: dict[str, float] = {}
        for r in np.unique(regions):
            mask = regions == r
            full_te_map[str(r)] = float((y[mask] - leg_mean[mask, 0]).mean())
        def full_te(rs):
            return np.array([full_te_map.get(str(r), global_mean) for r in rs], dtype=np.float64).reshape(-1, 1)
        X_all = np.concatenate([X_legs, leg_mean, leg_std, full_te(regions)], axis=1)
        y_all_residual = y - leg_mean[:, 0]
        if args.model == "ridge":
            final_model = Ridge(alpha=args.alpha, random_state=args.seed)
        else:
            hidden = tuple(int(x) for x in args.hidden.split(","))
            final_model = MLPRegressor(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                alpha=args.mlp_alpha,
                max_iter=args.mlp_max_iter,
                early_stopping=True,
                random_state=args.seed,
            )
        final_model.fit(X_all, y_all_residual)

        per_horizon[h] = {
            "fold_maes": fold_maes,
            "oof_mae": oof_mae,
            "baseline_leg_mean_mae": base_mae,
            "improvement_vs_leg_mean": base_mae - oof_mae,
        }
        fitted_models[int(h)] = {
            "model": final_model,
            "leg_cols": leg_cols,
            "region_te": full_te_map,
            "global_te": global_mean,
        }
        print(f"  horizon {h}: leg-mean MAE {base_mae:.4f} -> stacker OOF MAE {oof_mae:.4f}  (Δ {base_mae - oof_mae:+.4f})  folds {fold_maes}")

    # Overall MAE (just average per-horizon OOF MAE)
    overall = float(np.mean([per_horizon[h]["oof_mae"] for h in horizons]))
    print(f"\n[info] overall stacker OOF MAE (avg over horizons): {overall:.4f}")
    print(f"[info] overall leg-mean baseline MAE (avg over horizons): {float(np.mean([per_horizon[h]['baseline_leg_mean_mae'] for h in horizons])):.4f}")

    model_path = Path(args.output_model or (MODELS / f"stacker_{args.model}.joblib"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"per_horizon": fitted_models, "leg_cols": leg_cols, "model_kind": args.model}, model_path)
    print(f"[info] saved fitted stacker: {model_path}")

    rep = {
        "model": args.model,
        "input": args.input,
        "n_folds": args.n_folds,
        "leg_cols": leg_cols,
        "per_horizon": {int(h): v for h, v in per_horizon.items()},
        "overall_oof_mae": overall,
    }
    report_path = Path(args.output_report or (REPORTS / f"stacker_{args.model}.json"))
    with report_path.open("w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(f"[info] wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
