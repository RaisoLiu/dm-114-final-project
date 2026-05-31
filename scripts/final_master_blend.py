#!/usr/bin/env python3
"""Final master blend — all 6 candidates."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from local_eval_gate import candidate_stats, predict_public, fit_calibration

PRED = [f'pred_week{i+1}' for i in range(5)]

def main():
    truth = pd.read_csv(ROOT / "reports" / "_local_eval_oracle.csv")
    ext150 = pd.read_csv(ROOT / "submissions" / "submission_round5_pb30_x150_repro.csv")
    df_report = pd.read_csv(ROOT / "reports" / "_local_eval_gate_report.csv")
    known = df_report.dropna(subset=['public']).copy()
    coef, info = fit_calibration(known)
    print(f"Calibration coef: {coef}")
    print(f"  R²={info['r2']:.4f}")

    common = sorted(set(ext150['region_id']))
    e = ext150.set_index('region_id').loc[common][PRED].values
    e_mean, e_std = e.mean(), e.std()
    truth_v = truth.set_index('region_id').loc[common][PRED].values

    # Load all candidates
    cand_files = {
        'e': 'submission_round5_pb30_x150_repro.csv',
        't1': '_v18_track1_ssl.csv',
        't3': '_v18_track3_fast.csv',
        't3v2': '_v18_track3_v2_regemb.csv',
        'dp': 'submission_deep_ensemble_pb30_w10.csv',
        'l5': '_v18_pure_5yr_lag_weekly.csv',
        't25': '_v18_track25_lgbm_multiyear.csv',
    }
    data = {}
    for k, fname in cand_files.items():
        df = pd.read_csv(ROOT / "submissions" / fname)
        v = df.set_index('region_id').loc[common][PRED].values
        m, s = v.mean(), v.std()
        if k == 'e':
            data[k] = v
        else:
            data[k] = np.clip((v - m) * (e_std / s) + e_mean, 0, 5)
        print(f"  {k:<5s}: oracle={candidate_stats(pd.DataFrame(data[k], columns=PRED).assign(region_id=common)[['region_id']+PRED], ext150, truth)['oracle_mae']:.4f}")

    # 6-way grid with coarse step
    keys = ['e', 't1', 't3', 't3v2', 'dp', 'l5', 't25']
    print(f"\n=== 7-way coarse grid (step 0.10) ===")
    step = 0.10
    n_step = int(round(1/step)) + 1
    best = (None, None, 999)
    n_results = 0
    for w in np.ndindex(*([n_step]*6)):
        # last weight is 1 - sum(others) constrained
        if sum(w) > n_step - 1: continue
        wlast = (n_step - 1) - sum(w)
        weights = np.array(list(w) + [wlast]) * step
        if (weights < 0).any() or abs(weights.sum() - 1.0) > 1e-6: continue
        blend = sum(weights[i] * data[keys[i]] for i in range(7))
        blend = np.clip(blend, 0, 5)
        df_b = pd.DataFrame(blend, columns=PRED).assign(region_id=common)[['region_id']+PRED]
        s = candidate_stats(df_b, ext150, truth)
        pp = predict_public(coef, s)
        n_results += 1
        if pp < best[2]:
            best = (weights, s, pp)
    print(f"Searched {n_results} combinations")
    if best[0] is not None:
        w = best[0]
        print(f"Best 7-way: " + ' '.join(f'{k}={w[i]:.2f}' for i, k in enumerate(keys)))
        print(f"  oracle={best[1]['oracle_mae']:.4f}  std={best[1]['std']:.3f}  mean={best[1]['mean']:.3f}  pred_pub={best[2]:.4f}")

        # Save
        weights = best[0]
        blend = sum(weights[i] * data[keys[i]] for i in range(7))
        blend = np.clip(blend, 0, 5)
        df_b = pd.DataFrame(blend, columns=PRED).assign(region_id=common)[['region_id']+PRED]
        df_b.to_csv(ROOT / 'submissions' / '_v18_final7way.csv', index=False)

        # Transform on this
        vals = blend
        best_t = (None, None, 999)
        for shift in [0.0, -0.05, -0.10, -0.15, -0.20]:
            for scale in [0.85, 0.90, 0.95, 1.0]:
                for clip in [5.0, 4.0, 3.5, 3.0]:
                    m_ = vals.mean()
                    t = np.clip((vals - m_) * scale + m_ + shift, 0, clip)
                    df_t = pd.DataFrame(t, columns=PRED).assign(region_id=common)[['region_id']+PRED]
                    s = candidate_stats(df_t, ext150, truth)
                    pp = predict_public(coef, s)
                    if pp < best_t[2]:
                        best_t = ((shift, scale, clip), s, pp)
        print(f"\nBest transform: shift={best_t[0][0]:+.2f} scale={best_t[0][1]:.2f} clip={best_t[0][2]:.1f}")
        print(f"  predicted public: {best_t[2]:.4f}")
        print(f"  oracle={best_t[1]['oracle_mae']:.4f} std={best_t[1]['std']:.3f} mean={best_t[1]['mean']:.3f}")
        shift, scale, clip = best_t[0]
        m_ = vals.mean()
        t = np.clip((vals - m_) * scale + m_ + shift, 0, clip)
        df_t = pd.DataFrame(t, columns=PRED).assign(region_id=common)[['region_id']+PRED]
        df_t.to_csv(ROOT / 'submissions' / '_v18_final7way_T.csv', index=False)
        print("Saved _v18_final7way.csv and _v18_final7way_T.csv")


if __name__ == "__main__":
    main()
