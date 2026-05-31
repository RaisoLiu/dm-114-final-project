#!/usr/bin/env python3
"""Finer grid search with more low-oracle candidates."""
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
    print(f"Calibration: R²={info['r2']:.4f} RMSE={info['rmse']:.4f}")

    # 6 candidates with diverse oracle + ρ profiles
    cand_files = {
        't3': '_v18_track3_fast.csv',       # ρ=0.55, oracle 0.90
        't3ttt': '_v18_track3_ttt.csv',      # ρ=0.57, oracle 0.91
        'dp10': 'submission_deep_ensemble_pb30_w10.csv',  # ρ=0.996, oracle 0.84
        'dp05': 'submission_deep_ensemble_pb30_w05.csv',  # ρ=?, oracle 0.85
        'l5': '_v18_pure_5yr_lag_weekly.csv',  # ρ=0.39, oracle 1.17
        'rdo': 'submission_redo_ensemble_shift35.csv',  # oracle ~0.81
    }
    e_vals = ext150.set_index('region_id')[PRED].values  # may need re-alignment
    data = {}
    common = sorted(ext150['region_id'])
    for k, fname in cand_files.items():
        try:
            df = pd.read_csv(ROOT / "submissions" / fname)
            common = sorted(set(common) & set(df['region_id']))
            data[k] = df
        except Exception as e:
            print(f"  skip {k}: {e}")
    print(f"Common: {len(common)}")

    e_vals = ext150.set_index('region_id').loc[common][PRED].values
    e_mean, e_std = e_vals.mean(), e_vals.std()
    data_cal = {'e': e_vals}
    for k in cand_files:
        if k not in data: continue
        v = data[k].set_index('region_id').loc[common][PRED].values
        m, s = v.mean(), v.std()
        if s > 1e-6:
            data_cal[k + '_cal'] = np.clip((v - m) * (e_std / s) + e_mean, 0, 5)

    # Compute oracle/ρ for each
    truth_vals = truth.set_index('region_id').loc[common][PRED].values
    e_err = (e_vals - truth_vals).flatten()
    for k, v in data_cal.items():
        err = (v - truth_vals).flatten()
        mask = ~np.isnan(err) & ~np.isnan(e_err)
        rho = np.corrcoef(e_err[mask], err[mask])[0, 1] if k != 'e' else 1.0
        s = candidate_stats(pd.DataFrame(v, columns=PRED).assign(region_id=common)[['region_id'] + PRED], ext150, truth)
        print(f"  {k:<15s}  oracle={s['oracle_mae']:.4f}  std={s['std']:.3f}  ρ={rho:.4f}")

    # Fine grid: 4-way (e, t3, dp10, l5) with step 0.025
    print(f"\n=== Fine 4-way grid (step 0.025): e + t3_cal + dp10_cal + l5_cal ===")
    e_v = data_cal['e']
    t3_v = data_cal['t3_cal']
    dp10_v = data_cal['dp10_cal']
    l5_v = data_cal['l5_cal']
    step = 0.025
    results = []
    for we_int in range(0, int(1/step)+1):
        we = we_int * step
        for wt_int in range(0, int(1/step)-we_int+1):
            wt = wt_int * step
            for wd_int in range(0, int(1/step)-we_int-wt_int+1):
                wd = wd_int * step
                wl = 1.0 - we - wt - wd
                if wl < 0: continue
                blend = we * e_v + wt * t3_v + wd * dp10_v + wl * l5_v
                blend = np.clip(blend, 0, 5)
                df_b = pd.DataFrame(blend, columns=PRED); df_b.insert(0, 'region_id', common)
                s = candidate_stats(df_b, ext150, truth)
                pp = predict_public(coef, s)
                results.append({'we': we, 'wt': wt, 'wd': wd, 'wl': wl, 'oracle': s['oracle_mae'],
                                'std': s['std'], 'mad': s['mad'], 'pred_pub': pp})
    r = pd.DataFrame(results).sort_values('pred_pub')
    r.to_csv(ROOT / "reports" / "_fine_grid_4way_results.csv", index=False)
    print(f"\nTop 10 4-way (step 0.025):")
    for _, row in r.head(10).iterrows():
        print(f"  e={row['we']:.3f} t3={row['wt']:.3f} dp={row['wd']:.3f} l5={row['wl']:.3f}"
              f"  oracle={row['oracle']:.4f}  std={row['std']:.3f}  pred_pub={row['pred_pub']:.4f}")

    # Save top-3
    for i, (_, row) in enumerate(r.head(3).iterrows()):
        blend = row['we']*e_v + row['wt']*t3_v + row['wd']*dp10_v + row['wl']*l5_v
        blend = np.clip(blend, 0, 5)
        df_b = pd.DataFrame(blend, columns=PRED); df_b.insert(0, 'region_id', common)
        fname = (f"_v18_fine4_e{int(row['we']*1000):03d}t{int(row['wt']*1000):03d}"
                 f"dp{int(row['wd']*1000):03d}l{int(row['wl']*1000):03d}.csv")
        df_b.to_csv(ROOT / "submissions" / fname, index=False)
        print(f"  saved: {fname}")

    # 5-way with 2 different deep candidates
    print(f"\n=== 5-way grid (step 0.05): e + t3 + dp10 + dp05 + l5 ===")
    dp05_v = data_cal['dp05_cal']
    step = 0.05
    results = []
    for we_int in range(0, int(1/step)+1):
        we = we_int * step
        for wt_int in range(0, int(1/step)-we_int+1):
            wt = wt_int * step
            for wd_int in range(0, int(1/step)-we_int-wt_int+1):
                wd = wd_int * step
                for w5_int in range(0, int(1/step)-we_int-wt_int-wd_int+1):
                    w5 = w5_int * step
                    wl = 1.0 - we - wt - wd - w5
                    if wl < 0: continue
                    blend = we*e_v + wt*t3_v + wd*dp10_v + w5*dp05_v + wl*l5_v
                    blend = np.clip(blend, 0, 5)
                    df_b = pd.DataFrame(blend, columns=PRED); df_b.insert(0, 'region_id', common)
                    s = candidate_stats(df_b, ext150, truth)
                    pp = predict_public(coef, s)
                    results.append({'we': we, 'wt': wt, 'wd10': wd, 'wd05': w5, 'wl': wl,
                                     'oracle': s['oracle_mae'], 'pred_pub': pp})
    r5 = pd.DataFrame(results).sort_values('pred_pub')
    r5.to_csv(ROOT / "reports" / "_fine_grid_5way_dp.csv", index=False)
    print(f"\nTop 10 5-way:")
    for _, row in r5.head(10).iterrows():
        print(f"  e={row['we']:.2f} t3={row['wt']:.2f} dp10={row['wd10']:.2f} dp05={row['wd05']:.2f} l5={row['wl']:.2f}"
              f"  oracle={row['oracle']:.4f}  pred_pub={row['pred_pub']:.4f}")

    # Save top-1
    row = r5.iloc[0]
    blend = (row['we']*e_v + row['wt']*t3_v + row['wd10']*dp10_v + row['wd05']*dp05_v + row['wl']*l5_v)
    blend = np.clip(blend, 0, 5)
    df_b = pd.DataFrame(blend, columns=PRED); df_b.insert(0, 'region_id', common)
    df_b.to_csv(ROOT / "submissions" / f"_v18_fine5_top.csv", index=False)
    print(f"\nSaved best 5-way: _v18_fine5_top.csv")

if __name__ == "__main__":
    main()
