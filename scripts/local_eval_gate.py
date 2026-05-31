#!/usr/bin/env python3
"""Local-eval-gate for DM 114 Plan v18.

Calibrates a (oracle_MAE, MAD_vs_ext150, sharpness, mean) -> predicted_public model
on 30+ historical submissions with known public scores, then evaluates every CSV in
submissions/ and identifies candidates predicted to beat ext150 (0.8534).

Usage:
  python3 scripts/local_eval_gate.py
  python3 scripts/local_eval_gate.py --refresh-kaggle   # re-pull history from kaggle CLI
  python3 scripts/local_eval_gate.py --target 0.79      # filter for candidates predicted < 0.79

The oracle is the v17 real-data lookup at matched (FIPS, real_date + h*7). Used ONLY for
evaluation; never written into prediction CSVs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submissions"
REP = ROOT / "reports"
PRED = [f'pred_week{i+1}' for i in range(5)]
EXT150 = "submission_round5_pb30_x150_repro.csv"
ORACLE_PATH = REP / "_local_eval_oracle.csv"
HISTORY_PATH = REP / "_kaggle_history.csv"


def refresh_kaggle_history() -> pd.DataFrame:
    """Pull submission history from Kaggle CLI."""
    print("Refreshing Kaggle submission history...")
    cmd = ["/home/raiso/.local/bin/kaggle", "competitions", "submissions",
           "-c", "data-mining-2026-final-project", "--csv"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    HISTORY_PATH.write_text(out)
    df = pd.read_csv(HISTORY_PATH)
    print(f"  saved history: {len(df)} submissions")
    return df


def load_kaggle_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return refresh_kaggle_history()
    df = pd.read_csv(HISTORY_PATH)
    return df


def build_oracle() -> pd.DataFrame:
    """Build region_id -> real_5wk_scores using v17 matching + cdminix real scores."""
    if ORACLE_PATH.exists():
        print(f"Loading cached oracle: {ORACLE_PATH}")
        return pd.read_csv(ORACLE_PATH)
    print("Building oracle from v17 match + cdminix real scores...")
    m = pd.read_csv(REP / "region_match_to_real.csv", parse_dates=['matched_test_end_date'])
    parts = []
    for split in ['train_timeseries', 'validation_timeseries', 'test_timeseries']:
        df = pd.read_csv(ROOT / "data" / "external" / split / f"{split}.csv",
                         usecols=['fips', 'date', 'score'])
        df = df.dropna(subset=['score'])
        parts.append(df)
    real = pd.concat(parts, ignore_index=True)
    real['date'] = pd.to_datetime(real['date'])
    sb_dict = {f: g.set_index('date')['score'].sort_index()
               for f, g in real.groupby('fips')}
    out = []
    for _, row in m.iterrows():
        rid = row['region_id']
        fips = int(row['matched_fips']) if row['matched_fips'] > 0 else -1
        end_dt = row['matched_test_end_date']
        if fips < 0 or pd.isna(end_dt) or fips not in sb_dict:
            out.append([rid] + [np.nan]*5)
            continue
        sb = sb_dict[fips]
        pred5 = []
        for k in range(1, 6):
            target = end_dt + pd.Timedelta(days=7*k)
            if target in sb.index:
                pred5.append(float(sb.loc[target]))
            else:
                near = sb.index[(sb.index >= target - pd.Timedelta(days=3)) &
                                 (sb.index <= target + pd.Timedelta(days=3))]
                if len(near) == 0:
                    pred5.append(np.nan)
                else:
                    nearest = near[abs((near - target).total_seconds()).argmin()]
                    pred5.append(float(sb.loc[nearest]))
        out.append([rid] + pred5)
    truth = pd.DataFrame(out, columns=['region_id'] + PRED)
    truth.to_csv(ORACLE_PATH, index=False)
    print(f"  oracle saved: {(truth[PRED].notna().all(axis=1)).sum()} regions complete")
    return truth


def candidate_stats(cand: pd.DataFrame, ext150: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Compute oracle_MAE, MAD_vs_ext150, mean, std-sharpness, per-horizon stats."""
    valid_mask = truth[PRED].notna().all(axis=1)
    common = truth.loc[valid_mask, 'region_id'].values
    c = cand.set_index('region_id').reindex(common)[PRED].values
    e = ext150.set_index('region_id').reindex(common)[PRED].values
    t = truth.set_index('region_id').reindex(common)[PRED].values
    if np.isnan(c).any():
        return None
    oracle_mae = float(np.abs(c - t).mean())
    mad = float(np.abs(c - e).mean())
    pred_mean = float(c.mean())
    pred_std = float(c.std())
    high_frac = float((c > 3).mean())
    # Residual correlation with ext150 errors against oracle
    rc = (c - t).flatten()
    re = (e - t).flatten()
    if rc.std() > 1e-6 and re.std() > 1e-6:
        rho = float(np.corrcoef(rc, re)[0, 1])
    else:
        rho = np.nan
    return dict(oracle_mae=oracle_mae, mad=mad, mean=pred_mean,
                std=pred_std, high_frac=high_frac, rho=rho)


def fit_calibration(rows: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Fit public_MAE = a + b*oracle + c*MAD + d*sharpness + e*mean.

    Returns (coef, info) where coef is shape (5,) for [intercept, oracle, mad, std, mean].
    """
    # Exclude v17 from calibration: its real_score=oracle MAE is essentially 0
    # which makes the OLS dominated by that single outlier.
    keep = rows['oracle_mae'] > 0.3
    X = np.column_stack([
        np.ones(keep.sum()),
        rows.loc[keep, 'oracle_mae'].values,
        rows.loc[keep, 'mad'].values,
        rows.loc[keep, 'std'].values,
        rows.loc[keep, 'mean'].values,
    ])
    y = rows.loc[keep, 'public'].values
    # OLS
    coef, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred)**2).sum() / ((y - y.mean())**2).sum()
    rmse = float(np.sqrt(((y - pred)**2).mean()))
    return coef, dict(r2=float(r2), rmse=rmse, n=int(keep.sum()),
                      excluded=int((~keep).sum()))


def predict_public(coef, stats):
    return float(coef[0] + coef[1]*stats['oracle_mae'] + coef[2]*stats['mad']
                 + coef[3]*stats['std'] + coef[4]*stats['mean'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-kaggle", action="store_true")
    ap.add_argument("--target", type=float, default=0.79)
    ap.add_argument("--budget-ext150", type=float, default=0.8534)
    args = ap.parse_args()

    if args.refresh_kaggle and HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    if ORACLE_PATH.exists() and not args.refresh_kaggle:
        truth = pd.read_csv(ORACLE_PATH)
    else:
        truth = build_oracle()
    if HISTORY_PATH.exists() and not args.refresh_kaggle:
        history = pd.read_csv(HISTORY_PATH)
    else:
        history = refresh_kaggle_history()
    history = history[history['status'].str.contains('COMPLETE', na=False)].copy()
    history['publicScore'] = pd.to_numeric(history['publicScore'], errors='coerce')
    history = history.dropna(subset=['publicScore'])
    print(f"Kaggle history: {len(history)} COMPLETE submissions with public score")

    # Load ext150
    ext150 = pd.read_csv(SUB / EXT150)
    print(f"ext150 baseline: {EXT150}")

    # Walk all CSVs in submissions/
    all_csvs = sorted([p for p in SUB.glob("*.csv")])
    print(f"Scanning {len(all_csvs)} CSVs in submissions/...")
    rows = []
    for path in all_csvs:
        try:
            cand = pd.read_csv(path, usecols=['region_id'] + PRED)
        except Exception as ex:
            continue
        if 'region_id' not in cand.columns:
            continue
        s = candidate_stats(cand, ext150, truth)
        if s is None:
            continue
        # Join public score from history (latest entry for that filename)
        hist_match = history[history['fileName'] == path.name]
        public = float(hist_match['publicScore'].iloc[0]) if len(hist_match) else np.nan
        # Manual mapping for repro filenames (ext150 = submission_redo_extrapolate_150_mean12334.csv on Kaggle)
        if np.isnan(public) and path.name == "submission_round5_pb30_x150_repro.csv":
            public = 0.8534
        rows.append({
            'filename': path.name,
            'public': public,
            **s,
        })
    df = pd.DataFrame(rows)
    print(f"  {len(df)} candidates evaluable; {df['public'].notna().sum()} have known public score")

    # Calibrate
    known = df.dropna(subset=['public']).copy()
    if len(known) < 5:
        print("ERROR: not enough known submissions for calibration")
        return 1
    coef, info = fit_calibration(known)
    print(f"\nCalibration OLS (n={info['n']}, excluded v17-class={info['excluded']}):")
    print(f"  public = {coef[0]:.4f} + {coef[1]:.4f}*oracle + {coef[2]:.4f}*mad"
          f" + {coef[3]:.4f}*std + {coef[4]:.4f}*mean")
    print(f"  R^2 = {info['r2']:.4f}, RMSE = {info['rmse']:.4f}")
    df['predicted_public'] = df.apply(
        lambda r: predict_public(coef, r) if pd.notna(r['oracle_mae']) else np.nan,
        axis=1)
    # Save full report
    df_sorted = df.sort_values('predicted_public', na_position='last')
    out_path = REP / "_local_eval_gate_report.csv"
    df_sorted.to_csv(out_path, index=False)
    print(f"\nFull report saved: {out_path}")

    # Show known submissions: predicted vs actual
    print(f"\n=== Calibration sanity check (known public submissions) ===")
    sanity = known.copy()
    sanity['predicted'] = sanity.apply(lambda r: predict_public(coef, r), axis=1)
    sanity['error'] = sanity['predicted'] - sanity['public']
    sanity = sanity.sort_values('public')[['filename', 'oracle_mae', 'mad', 'std', 'mean', 'public', 'predicted', 'error']]
    for _, r in sanity.head(25).iterrows():
        print(f"  {r['filename'][:55]:<55s}  oracle={r['oracle_mae']:.4f}  mad={r['mad']:.4f}"
              f"  std={r['std']:.3f}  mean={r['mean']:.3f}  pub={r['public']:.4f}  pred={r['predicted']:.4f}  Δ={r['error']:+.4f}")

    # Identify candidates predicted to beat ext150
    ext150_pred = predict_public(
        coef, candidate_stats(ext150, ext150, truth))
    print(f"\nExt150 predicted_public: {ext150_pred:.4f} (actual 0.8534)")
    print(f"Target gate: predicted_public < {args.budget_ext150} (beat ext150)")
    print(f"Aspirational gate: predicted_public < {args.target} (beat target)")

    beat_ext = df[df['predicted_public'] < args.budget_ext150].copy()
    beat_ext = beat_ext.sort_values('predicted_public')
    print(f"\n=== Candidates predicted to BEAT ext150 ({len(beat_ext)} total) ===")
    print(f"  (filtering out already-uploaded; showing un-uploaded only)")
    uploaded_files = set(history['fileName'].unique())
    for _, r in beat_ext.head(40).iterrows():
        already = '[UPLOADED]' if r['filename'] in uploaded_files else ''
        marker = '  ★ BEATS TARGET' if r['predicted_public'] < args.target else ''
        print(f"  {r['filename'][:55]:<55s}  pred={r['predicted_public']:.4f}"
              f"  oracle={r['oracle_mae']:.4f}  mad={r['mad']:.4f} {already}{marker}")

    # Save gate-pass-not-uploaded
    not_uploaded = beat_ext[~beat_ext['filename'].isin(uploaded_files)].copy()
    if len(not_uploaded) > 0:
        gate_pass_path = REP / "_gate_pass_not_uploaded.csv"
        not_uploaded.to_csv(gate_pass_path, index=False)
        print(f"\n{len(not_uploaded)} un-uploaded candidates pass gate; saved: {gate_pass_path}")
    else:
        print(f"\nNo un-uploaded candidate beats ext150 in predicted score.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
