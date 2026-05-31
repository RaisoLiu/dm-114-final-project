#!/usr/bin/env python3
"""Track 4 — Orthogonal ensemble + gate evaluation across all v18 candidates.

For a list of candidate CSVs:
  1. Compute oracle MAE + MAD vs ext150 + ρ vs ext150 errors
  2. Build pairwise blends (α from {0.05..0.50}) + multi-blend (NNLS-free)
  3. Pass all through local-eval-gate
  4. Identify top-3 candidates by predicted_public
  5. Optionally upload top-1 (if --upload flag and gate passes)

Note: blend alphas are pre-determined (not oracle-optimized).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from local_eval_gate import candidate_stats, predict_public, fit_calibration

PRED = [f'pred_week{i+1}' for i in range(5)]
SUB = ROOT / "submissions"
REP = ROOT / "reports"
EXT150 = "submission_round5_pb30_x150_repro.csv"
EXT150_PUBLIC = 0.8534


def load_aligned(filename: str, common_regions: list[str]) -> np.ndarray:
    df = pd.read_csv(SUB / filename)
    return df.set_index('region_id').loc[common_regions][PRED].values


def evaluate(name: str, vals: np.ndarray, regions: list[str], ext150: pd.DataFrame,
             truth: pd.DataFrame, coef) -> dict:
    df_b = pd.DataFrame(vals, columns=PRED)
    df_b.insert(0, 'region_id', regions)
    s = candidate_stats(df_b, ext150, truth)
    if s is None:
        return None
    pp = predict_public(coef, s)
    return {'name': name, **s, 'pred_pub': pp}


def calibrate_to_ext150(v: np.ndarray, ext150_vals: np.ndarray) -> np.ndarray:
    """Shift+scale predictions to match ext150 mean/std."""
    m, s = v.mean(), v.std()
    em, es = ext150_vals.mean(), ext150_vals.std()
    if s < 1e-6:
        return v.copy()
    return np.clip((v - m) * (es / s) + em, 0, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs='+', default=[
        "_v18_track3_fast.csv",
        "_v18_track25_lgbm_multiyear.csv",
        "_v18_track3_ttt.csv",
    ])
    ap.add_argument("--alphas", type=float, nargs='+',
                    default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70])
    ap.add_argument("--save-prefix", default="_v18_track4")
    args = ap.parse_args()

    # Load oracle + ext150
    truth = pd.read_csv(REP / "_local_eval_oracle.csv")
    ext150 = pd.read_csv(SUB / EXT150)
    common_regions = sorted(set(ext150['region_id']) & set(truth['region_id']))
    print(f"Aligned regions: {len(common_regions)}")

    ext150_aligned = ext150.set_index('region_id').loc[common_regions]
    truth_aligned = truth.set_index('region_id').loc[common_regions]
    e_vals = ext150_aligned[PRED].values
    truth_vals = truth_aligned[PRED].values

    # Fit calibration
    df_report = pd.read_csv(REP / "_local_eval_gate_report.csv")
    known = df_report.dropna(subset=['public']).copy()
    coef, info = fit_calibration(known)
    print(f"Calibration coef: {coef}")
    print(f"  R² = {info['r2']:.4f}, RMSE = {info['rmse']:.4f}\n")

    # Load candidates
    candidates = {}
    for fname in args.candidates:
        path = SUB / fname
        if not path.exists():
            print(f"  [skip] {fname} not found")
            continue
        try:
            vals = load_aligned(fname, common_regions)
            if np.isnan(vals).any():
                print(f"  [skip] {fname} has NaN")
                continue
            candidates[fname] = vals
            print(f"  loaded {fname}: shape {vals.shape}  mean={vals.mean():.3f}  std={vals.std():.3f}")
        except Exception as ex:
            print(f"  [error] {fname}: {ex}")

    # Evaluate raw candidates + calibrated versions
    print("\n=== Raw + calibrated candidate evaluation ===")
    results = []
    results.append(evaluate("ext150 (anchor)", e_vals, common_regions, ext150, truth, coef))
    for fname, v in candidates.items():
        results.append(evaluate(f"{fname} raw", v, common_regions, ext150, truth, coef))
        v_cal = calibrate_to_ext150(v, e_vals)
        out_name = f"{args.save_prefix}_{Path(fname).stem.replace('_v18_', '')}_cal.csv"
        df_cal = pd.DataFrame(v_cal, columns=PRED)
        df_cal.insert(0, 'region_id', common_regions)
        df_cal.to_csv(SUB / out_name, index=False)
        results.append(evaluate(f"{out_name} (calibrated)", v_cal, common_regions, ext150, truth, coef))

    # Pairwise blends: each calibrated candidate × ext150 at various α
    print("\n=== Pairwise blends with ext150 ===")
    for fname, v in candidates.items():
        v_cal = calibrate_to_ext150(v, e_vals)
        for alpha in args.alphas:
            blend = alpha * v_cal + (1 - alpha) * e_vals
            blend = np.clip(blend, 0, 5)
            name = f"{args.save_prefix}_{Path(fname).stem.replace('_v18_', '')}_blendext_a{int(alpha*100):02d}.csv"
            df_b = pd.DataFrame(blend, columns=PRED)
            df_b.insert(0, 'region_id', common_regions)
            df_b.to_csv(SUB / name, index=False)
            results.append(evaluate(name, blend, common_regions, ext150, truth, coef))

    # Multi-blend: average of all calibrated candidates with ext150
    if len(candidates) >= 2:
        print("\n=== Multi-candidate ensembles ===")
        cal_list = [calibrate_to_ext150(v, e_vals) for v in candidates.values()]
        for alpha in args.alphas:
            avg_cand = np.mean(cal_list, axis=0)
            blend = alpha * avg_cand + (1 - alpha) * e_vals
            blend = np.clip(blend, 0, 5)
            name = f"{args.save_prefix}_multi_avg_a{int(alpha*100):02d}.csv"
            df_b = pd.DataFrame(blend, columns=PRED)
            df_b.insert(0, 'region_id', common_regions)
            df_b.to_csv(SUB / name, index=False)
            results.append(evaluate(name, blend, common_regions, ext150, truth, coef))

    # Filter, sort, save
    valid = [r for r in results if r is not None]
    df_res = pd.DataFrame(valid).sort_values('pred_pub')
    out_path = REP / "_track4_ensemble_results.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\n=== Top 20 candidates by predicted public ===")
    for _, r in df_res.head(20).iterrows():
        marker = ''
        if r['pred_pub'] < EXT150_PUBLIC:
            marker = '  ★ BEATS EXT150'
        if r['pred_pub'] < 0.79:
            marker = '  🎯 BEATS TARGET'
        print(f"  {r['name'][:60]:<60s}  oracle={r['oracle_mae']:.4f}  mad={r['mad']:.4f}"
              f"  std={r['std']:.3f}  mean={r['mean']:.3f}  pred={r['pred_pub']:.4f}{marker}")
    print(f"\nFull results: {out_path}")
    print(f"\nBest candidate predicted public: {df_res.iloc[0]['pred_pub']:.4f}")
    if df_res.iloc[0]['pred_pub'] < EXT150_PUBLIC:
        print(f"  ✅ {df_res.iloc[0]['name']} predicted to beat ext150 (0.8534)")
    else:
        print(f"  ❌ No candidate predicted to beat ext150")


if __name__ == "__main__":
    main()
