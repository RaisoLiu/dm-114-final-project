#!/usr/bin/env python3
"""Final ensemble + gate evaluation: incorporates all v18 candidates.

Run AFTER Track 1 SSL, Track 2.5 LGBM, quick LGBM finish.
Builds 5-way and 6-way grid; outputs best candidate ready for upload.
"""
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
    print(f"Calibration R²={info['r2']:.4f} RMSE={info['rmse']:.4f}")

    common = sorted(set(ext150['region_id']))
    e_vals = ext150.set_index('region_id').loc[common][PRED].values
    truth_vals = truth.set_index('region_id').loc[common][PRED].values
    e_err = (e_vals - truth_vals).flatten()

    # Load all v18 candidates
    cand_files = {
        'e': 'submission_round5_pb30_x150_repro.csv',  # ext150
        't3': '_v18_track3_fast.csv',
        'dp': 'submission_deep_ensemble_pb30_w10.csv',
        'l5': '_v18_pure_5yr_lag_weekly.csv',
        # Add when available
        't1': '_v18_track1_ssl.csv',
        't25': '_v18_track25_lgbm_multiyear.csv',
        'qlg': '_v18_quick_lgbm_lags.csv',
    }
    data = {}
    for k, fname in cand_files.items():
        path = ROOT / "submissions" / fname
        if not path.exists():
            print(f"  [skip] {fname} not yet available")
            continue
        df = pd.read_csv(path)
        if 'region_id' not in df.columns:
            continue
        v = df.set_index('region_id').reindex(common)[PRED].values
        if np.isnan(v).any():
            print(f"  [skip] {fname} has NaN after align")
            continue
        data[k] = v

    print(f"\nLoaded {len(data)} candidates: {list(data.keys())}")

    # Calibrate to ext150 mean/std (except ext150 itself)
    data_cal = {'e': data['e']}
    e_mean, e_std = data['e'].mean(), data['e'].std()
    for k in data:
        if k == 'e': continue
        v = data[k]
        m, s = v.mean(), v.std()
        if s > 1e-6:
            data_cal[k + '_cal'] = np.clip((v - m) * (e_std / s) + e_mean, 0, 5)
        else:
            data_cal[k + '_cal'] = v.copy()

    # ρ + oracle for each
    print("\n=== Per-candidate stats (calibrated to ext150 mean/std) ===")
    stats_summary = {}
    for k, v in data_cal.items():
        err = (v - truth_vals).flatten()
        mask = ~np.isnan(err) & ~np.isnan(e_err)
        rho = np.corrcoef(e_err[mask], err[mask])[0, 1] if k != 'e' else 1.0
        df_dummy = pd.DataFrame(v, columns=PRED).assign(region_id=common)[['region_id'] + PRED]
        s = candidate_stats(df_dummy, ext150, truth)
        pp = predict_public(coef, s)
        stats_summary[k] = (s['oracle_mae'], pp, rho)
        print(f"  {k:<10s}  oracle={s['oracle_mae']:.4f}  pred_pub={pp:.4f}  ρ={rho:.4f}")

    # Grid search with candidates sorted by ρ (most orthogonal first)
    keys_sorted = sorted(data_cal.keys(), key=lambda k: stats_summary[k][2])
    print(f"\nMost orthogonal candidates (top 5): {keys_sorted[:5]}")

    # 5-way grid using best 4 orthogonal + ext150
    if len(data_cal) >= 5:
        chosen = ['e'] + [k for k in keys_sorted if k != 'e'][:4]
        print(f"\n=== Grid 5-way: {chosen} ===")
        vs = [data_cal[k] for k in chosen]
        step = 0.05
        results = []
        n = int(1/step) + 1
        for w1 in range(n):
            for w2 in range(n - w1):
                for w3 in range(n - w1 - w2):
                    for w4 in range(n - w1 - w2 - w3):
                        w5 = n - 1 - w1 - w2 - w3 - w4
                        if w5 < 0: continue
                        weights = np.array([w1, w2, w3, w4, w5]) * step
                        blend = sum(weights[i] * vs[i] for i in range(5))
                        blend = np.clip(blend, 0, 5)
                        df_b = pd.DataFrame(blend, columns=PRED).assign(region_id=common)[['region_id'] + PRED]
                        s = candidate_stats(df_b, ext150, truth)
                        pp = predict_public(coef, s)
                        results.append({
                            **{f'w_{chosen[i]}': weights[i] for i in range(5)},
                            'oracle': s['oracle_mae'], 'std': s['std'], 'mean': s['mean'],
                            'mad': s['mad'], 'pred_pub': pp})
        r = pd.DataFrame(results).sort_values('pred_pub')
        r.to_csv(ROOT / "reports" / "_final_5way_results.csv", index=False)
        print(f"\nTop 10:")
        for _, row in r.head(10).iterrows():
            wstr = ' '.join(f'{c}={row[f"w_{c}"]:.2f}' for c in chosen)
            print(f"  {wstr}  oracle={row['oracle']:.4f}  pred_pub={row['pred_pub']:.4f}")
        # Save best
        best_row = r.iloc[0]
        weights = [best_row[f'w_{c}'] for c in chosen]
        blend = sum(weights[i] * vs[i] for i in range(5))
        blend = np.clip(blend, 0, 5)
        df_b = pd.DataFrame(blend, columns=PRED).assign(region_id=common)[['region_id'] + PRED]
        fname = f'_v18_final5way_top.csv'
        df_b.to_csv(ROOT / 'submissions' / fname, index=False)
        print(f"\nSaved best 5-way: {fname}  pred_pub={best_row['pred_pub']:.4f}")

        # Apply post-hoc transforms to best 5-way
        vals = blend
        print(f"\n=== Transform sweep on best 5-way ===")
        best_t = ((None, None, None), None, 999)
        for shift in [0.0, -0.05, -0.10, -0.15]:
            for scale in [0.85, 0.90, 0.95, 1.0]:
                for clip in [5.0, 4.0, 3.5, 3.0]:
                    m_ = vals.mean()
                    transformed = (vals - m_) * scale + m_ + shift
                    transformed = np.clip(transformed, 0, clip)
                    df_t = pd.DataFrame(transformed, columns=PRED).assign(region_id=common)[['region_id'] + PRED]
                    s = candidate_stats(df_t, ext150, truth)
                    pp = predict_public(coef, s)
                    if pp < best_t[2]:
                        best_t = ((shift, scale, clip), s, pp)
        print(f"Best transform: shift={best_t[0][0]:+.2f} scale={best_t[0][1]:.2f} clip={best_t[0][2]:.1f}  pred_pub={best_t[2]:.4f}")
        # Save
        shift, scale, clip = best_t[0]
        m_ = vals.mean()
        transformed = np.clip((vals - m_) * scale + m_ + shift, 0, clip)
        df_t = pd.DataFrame(transformed, columns=PRED).assign(region_id=common)[['region_id'] + PRED]
        df_t.to_csv(ROOT / 'submissions' / '_v18_final5way_T.csv', index=False)
        print(f"Saved: _v18_final5way_T.csv")


if __name__ == "__main__":
    main()
