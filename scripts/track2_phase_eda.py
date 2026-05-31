#!/usr/bin/env python3
"""Track 2 — Multi-year phase EDA: check if synth scores have multi-year cycles.

For each of 22 train regions, compute the score time series at weekly anchors,
then FFT to find dominant periods. If any period > 365 days has significant power,
multi-year phase is real and worth modeling. If not, drop Track 2.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    print("Loading train.csv...")
    df = pd.read_csv(ROOT / "data" / "train.csv")
    df['date'] = df['date'].astype(str)
    regions = df['region_id'].unique().tolist()
    print(f"  {len(regions)} regions")

    print("\nPer-region FFT of weekly score series:")
    all_peaks = []
    for rid in regions:
        sub = df[df['region_id'] == rid].sort_values('date').reset_index(drop=True)
        # Weekly anchors = rows where score is not null
        weekly = sub[sub['score'].notna()][['date', 'score']].reset_index(drop=True)
        if len(weekly) < 100:
            continue
        s = weekly['score'].values.astype(np.float32)
        s = s - s.mean()
        # FFT on weekly series (sample period = 7 days)
        n = len(s)
        fft_vals = np.fft.rfft(s)
        freqs = np.fft.rfftfreq(n, d=7.0)  # cycles per day
        # Skip DC (freq[0] = 0)
        power = np.abs(fft_vals)**2
        power[0] = 0
        # Convert freq -> period (days)
        periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-12), np.inf)
        # Top 5 peaks by power
        top_idx = np.argsort(power)[::-1][:5]
        top_periods = periods[top_idx]
        top_power = power[top_idx]
        # Filter multi-year peaks (period > 365 d)
        multi_year_idx = [(p, pw) for p, pw in zip(top_periods, top_power) if 365 < p < 365 * 8]
        all_peaks.append({
            'region': rid,
            'n_weekly': n,
            'top_period': float(top_periods[0]),
            'top_power': float(top_power[0]),
            'multi_year_peaks': multi_year_idx[:3],
        })
        if len(all_peaks) <= 10:
            mp_str = ", ".join(f"({p:.0f}d, pw={pw:.1f})" for p, pw in multi_year_idx[:3])
            print(f"  {rid:>6s}  n={n:<4d}  top_period={top_periods[0]:>6.1f} d (pw={top_power[0]:>8.1f})"
                  f"  multi-yr: {mp_str}")

    # Summarize
    print(f"\n=== Multi-year peak detection across {len(all_peaks)} regions ===")
    has_mp = sum(1 for p in all_peaks if len(p['multi_year_peaks']) > 0)
    print(f"  regions with at least one multi-year peak (365-2920d) in top-5: {has_mp}/{len(all_peaks)}")
    # Aggregate
    common_periods = {}
    for p in all_peaks:
        for period, power in p['multi_year_peaks']:
            bucket = int(round(period / 365)) * 365  # round to nearest year
            common_periods.setdefault(bucket, []).append(power)
    print(f"\nAggregated multi-year peaks (period rounded to nearest 365d):")
    for bucket in sorted(common_periods.keys()):
        n = len(common_periods[bucket])
        mean_pw = np.mean(common_periods[bucket])
        print(f"  ~{bucket} d ({bucket/365:.1f} yr):  {n}/22 regions  mean_power={mean_pw:.1f}")

    # Decision
    print(f"\n=== DECISION ===")
    if has_mp >= 15:
        print("Multi-year peaks ARE present in majority — TRACK 2 IS WORTH PURSUING")
        print("Next: build multi-year phase features for LGBM")
    else:
        print("Multi-year peaks are SCATTERED — drop Track 2, focus on Tracks 1+3.")

    # Save
    out_df = pd.DataFrame([{
        'region': p['region'],
        'n_weekly': p['n_weekly'],
        'top_period': p['top_period'],
        'top_power': p['top_power'],
        'multi_year_peaks_str': str(p['multi_year_peaks']),
    } for p in all_peaks])
    out_path = ROOT / "reports" / "_track2_fft_peaks.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
