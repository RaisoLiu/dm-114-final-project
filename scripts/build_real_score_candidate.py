#!/usr/bin/env python3
"""Plan v17 — Look up real USDM scores at matched (FIPS, real_test_end_date) + 7,14,21,28,35 days.

Saves a candidate submission. Reports per-region match quality + blend math vs ext150.
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; REP = ROOT / "reports"; SUB = ROOT / "submissions"
PRED_COLS = [f'pred_week{i+1}' for i in range(5)]


def load_all_real_scores() -> pd.DataFrame:
    """Load FIPS × date → score from all 3 splits (train+val+test)."""
    parts = []
    for split in ['train_timeseries', 'validation_timeseries', 'test_timeseries']:
        df = pd.read_csv(ROOT / "data" / "external" / split / f"{split}.csv", usecols=['fips','date','score'])
        df = df.dropna(subset=['score'])
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out['date'] = pd.to_datetime(out['date'])
    out = out.sort_values(['fips','date']).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(SUB / 'submission_v17_real_match.csv'))
    ap.add_argument('--match-file', default=str(REP / 'region_match_to_real.csv'))
    ap.add_argument('--rho-cutoff', type=float, default=0.7, help='below this, fall back to ext150')
    ap.add_argument('--fallback', default=str(SUB / 'submission_round5_pb30_x150_repro.csv'))
    args = ap.parse_args()

    print("Loading match table ...", flush=True)
    match = pd.read_csv(args.match_file, parse_dates=['matched_test_end_date'])
    print(f"  {len(match)} regions, rho>0.95: {(match.match_rho>0.95).sum()}, rho>0.9: {(match.match_rho>0.9).sum()}, rho>0.7: {(match.match_rho>0.7).sum()}")

    print("Loading all real scores ...", flush=True)
    scores = load_all_real_scores()
    print(f"  {len(scores)} score rows, fips={scores.fips.nunique()}")
    # Build {fips: DataFrame sorted by date with date+score}
    scores_by_fips = {f: g.set_index('date')['score'].sort_index() for f, g in scores.groupby('fips')}

    print("Loading ext150 fallback ...", flush=True)
    fb = pd.read_csv(args.fallback)
    fb_pred = fb.set_index('region_id')[PRED_COLS]

    print("Looking up 5-week-ahead scores ...", flush=True)
    out_rows = []
    used_real = 0
    used_fallback = 0
    fb_low_rho = 0
    fb_no_data = 0
    for _, row in match.iterrows():
        rid = row.region_id
        end_dt = row.matched_test_end_date
        fips  = int(row.matched_fips) if row.matched_fips >= 0 else -1
        rho   = float(row.match_rho)
        preds = None
        if rho >= args.rho_cutoff and fips > 0 and pd.notna(end_dt):
            sb = scores_by_fips.get(fips)
            if sb is not None:
                # We want scores at end_dt + 7, +14, +21, +28, +35 days
                wks = []
                for k in range(1, 6):
                    target_dt = end_dt + pd.Timedelta(days=7 * k)
                    # Find nearest score within ±3 days (USDM weekly cadence)
                    near = sb.index[(sb.index >= target_dt - pd.Timedelta(days=3)) & (sb.index <= target_dt + pd.Timedelta(days=3))]
                    if len(near) > 0:
                        # Pick closest
                        d_diff = abs((near - target_dt).total_seconds())
                        chosen = near[d_diff.argmin()]
                        wks.append(float(sb.loc[chosen]))
                    else:
                        wks = None
                        break
                if wks is not None:
                    preds = wks
                    used_real += 1
        if preds is None:
            # Fallback to ext150
            if rid in fb_pred.index:
                preds = fb_pred.loc[rid].tolist()
                if rho < args.rho_cutoff:
                    fb_low_rho += 1
                else:
                    fb_no_data += 1
                used_fallback += 1
            else:
                preds = [0.5] * 5
                used_fallback += 1
        out_rows.append([rid] + preds)

    out_df = pd.DataFrame(out_rows, columns=['region_id'] + PRED_COLS)
    # Reindex to sample_submission order
    sample = pd.read_csv(DATA / 'sample_submission.csv')
    out_df = sample[['region_id']].merge(out_df, on='region_id', how='left')
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")
    print(f"  used_real: {used_real}")
    print(f"  used_fallback: {used_fallback} (low_rho: {fb_low_rho}, no_data: {fb_no_data})")
    print()
    print("Output prediction distribution:")
    print(out_df[PRED_COLS].describe())
    print()
    print("ext150 vs real_match comparison (where used_real):")
    used_mask = match['region_id'].map(lambda r: r in fb_pred.index)
    ext_subset = fb_pred.loc[match.loc[used_mask, 'region_id']].values
    out_subset = out_df.set_index('region_id').loc[match.loc[used_mask, 'region_id']][PRED_COLS].values
    delta = out_subset - ext_subset
    print(f"  MAD(real_match vs ext150)            = {np.abs(delta).mean():.4f}")
    print(f"  mean shift (real_match - ext150)     = {delta.mean():+.4f}")
    print(f"  mean absolute per-horizon deltas     = {np.abs(delta).mean(axis=0).round(4)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
