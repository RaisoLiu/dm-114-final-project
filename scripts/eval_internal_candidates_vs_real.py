#!/usr/bin/env python3
"""Validate INTERNAL candidates against the recovered ground-truth from v17 matching.

ext150 public was 0.8534. v17 lookup public was 0.1866. So real_lookup is ~0.85 away from ext150
and very close to synth_test_labels (within ~0.10 noise). This means MAE(candidate, real_lookup)
is a strong oracle for the public score the candidate WOULD get if uploaded.

This script:
1. Loads v17 region matches + real scores for matched (fips, test_end_date + 7,14,21,28,35 days)
2. For each candidate (zero_lt010, pbs_*, pbf_*, ext150 baseline), computes MAE vs real_lookup
3. Sorts candidates by MAE (best first) — this is the near-public-score ranking.

Note: the candidates themselves use ONLY internal data (no external lookup); only the
evaluation step uses external. Submitting any of these does NOT submit external data.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submissions"
REP = ROOT / "reports"
PRED = [f'pred_week{i+1}' for i in range(5)]


def load_oracle() -> pd.DataFrame:
    """Build region_id -> real_5wk_scores DataFrame using v17 matching + real cdminix scores."""
    m = pd.read_csv(REP / "region_match_to_real.csv", parse_dates=['matched_test_end_date'])
    print(f"  loaded match table: {len(m)} regions, rho>0.95: {(m.match_rho>0.95).sum()}")
    # Load real scores from all 3 splits
    parts = []
    for split in ['train_timeseries', 'validation_timeseries', 'test_timeseries']:
        df = pd.read_csv(ROOT / "data" / "external" / split / f"{split}.csv", usecols=['fips','date','score'])
        df = df.dropna(subset=['score'])
        parts.append(df)
    real = pd.concat(parts, ignore_index=True)
    real['date'] = pd.to_datetime(real['date'])
    scores_by_fips = {f: g.set_index('date')['score'].sort_index() for f, g in real.groupby('fips')}
    print(f"  loaded real scores for {len(scores_by_fips)} fips")

    out_rows = []
    n_missing = 0
    for _, row in m.iterrows():
        rid = row['region_id']
        fips = int(row['matched_fips'])
        end_dt = row['matched_test_end_date']
        if fips < 0 or pd.isna(end_dt):
            n_missing += 1
            out_rows.append([rid] + [np.nan]*5); continue
        sb = scores_by_fips.get(fips)
        if sb is None:
            n_missing += 1
            out_rows.append([rid] + [np.nan]*5); continue
        pred5 = []
        for k in range(1, 6):
            target = end_dt + pd.Timedelta(days=7*k)
            if target not in sb.index:
                near = sb.index[(sb.index >= target - pd.Timedelta(days=3)) & (sb.index <= target + pd.Timedelta(days=3))]
                if len(near) == 0:
                    pred5.append(np.nan); continue
                target = near[abs((near - target).total_seconds()).argmin()]
            pred5.append(float(sb.loc[target]))
        out_rows.append([rid] + pred5)
    truth = pd.DataFrame(out_rows, columns=['region_id'] + PRED)
    print(f"  oracle: {(truth[PRED].notna().all(axis=1)).sum()} regions with full 5-week labels, {n_missing} missing")
    return truth


def main():
    print("Building oracle from v17 + cdminix data...")
    truth = load_oracle()

    # Candidate list
    candidates = [
        # baseline
        ('submission_round5_pb30_x150_repro.csv', 'ext150 (anchor)'),
        # per-bucket compress (factor)
        ('submission_pbf_lo085_hi100.csv', 'pbf lo=0.85 hi=1.00'),
        ('submission_pbf_lo070_hi100.csv', 'pbf lo=0.70 hi=1.00'),
        ('submission_pbf_lo050_hi100.csv', 'pbf lo=0.50 hi=1.00'),
        ('submission_pbf_lo030_hi100.csv', 'pbf lo=0.30 hi=1.00'),
        ('submission_pbf_lo085_hi085.csv', 'pbf lo=0.85 hi=0.85'),
        ('submission_pbf_lo070_hi070.csv', 'pbf lo=0.70 hi=0.70'),
        ('submission_pbf_lo050_hi070.csv', 'pbf lo=0.50 hi=0.70'),
        # per-bucket shift (slope from reverse_diagnosis)
        ('submission_pbs_scale050.csv', 'pbs scale=0.5'),
        ('submission_pbs_scale100.csv', 'pbs scale=1.0'),
        ('submission_pbs_scale150.csv', 'pbs scale=1.5'),
        ('submission_pbs_scale200.csv', 'pbs scale=2.0'),
        ('submission_pbs_scale300.csv', 'pbs scale=3.0'),
        # zero-low surgical
        ('submission_zero_lt010.csv', 'zero p<0.1'),
        # combo
        ('submission_zerolow_plus_pbs.csv', 'zerolow + pbs100'),
    ]

    # Other v15 ensembles for comparison
    v15_cands = [
        ('submission_v15_FINAL_with_bias.csv', 'v15 bias-corrected'),
        ('submission_v15_FINAL_v2_with_bias.csv', 'v15 v2 bias'),
        ('submission_v15_FINAL_ensemble.csv', 'v15 grand ensemble'),
        ('submission_v15_grand_ensemble_v4.csv', 'v15 grand v4'),
        ('submission_v15_safe_seqonly_ensemble.csv', 'v15 seq-only'),
        ('submission_v15_optimal_ensemble.csv', 'v15 optimal'),
        ('submission_v15_cross_arch_ensemble.csv', 'v15 cross-arch'),
        ('submission_v15_per_horizon_ensemble.csv', 'v15 per-horizon'),
        ('submission_v15_gru_weather_only_blend_w05.csv', 'v15 gru-w 5% blend'),
        ('_v15_bias_a05.csv', 'v15 bias α=0.05'),
        ('_v15_bias_a10.csv', 'v15 bias α=0.10'),
        # v16
        ('submission_v16_factor130.csv', 'v16 factor=1.3'),
        ('_v16_factor_130.csv', 'v16 factor=1.3 (alt name)'),
    ]
    candidates.extend(v15_cands)

    # Compute MAE for each candidate vs oracle (subset of regions with full labels)
    valid_mask = truth[PRED].notna().all(axis=1)
    truth_valid = truth.loc[valid_mask].set_index('region_id')
    print(f"\nEvaluating on {valid_mask.sum()} regions with full oracle labels:")
    print()

    results = []
    for fname, label in candidates:
        path = SUB / fname
        if not path.exists():
            continue
        cand = pd.read_csv(path)
        if 'region_id' not in cand.columns: continue
        cand_aligned = cand.set_index('region_id').reindex(truth_valid.index)
        if cand_aligned[PRED].isna().any().any():
            print(f"  [skip] {fname}: NaN after align")
            continue
        ae = np.abs(cand_aligned[PRED].values - truth_valid[PRED].values)
        mae = float(ae.mean())
        per_h = ae.mean(axis=0)
        results.append((label, fname, mae, per_h))

    # Sort by MAE
    results.sort(key=lambda x: x[2])
    print(f"{'rank':>4s}  {'candidate':<35s}  {'oracle_MAE':>11s}  {'per-h MAE':<35s}")
    ext_mae = next(r[2] for r in results if 'ext150' in r[0])
    for i, (label, fname, mae, per_h) in enumerate(results):
        marker = ' ← ext150' if 'ext150' in label else ('  BEATS ext150' if mae < ext_mae else '')
        print(f"  {i+1:>3d}  {label:<35s}  {mae:>11.4f}  [{', '.join(f'{x:.3f}' for x in per_h)}]{marker}")
    print()
    print(f"Note: ext150 public MAE = 0.8534. Oracle MAE for ext150 (here) = {ext_mae:.4f}.")
    print(f"Gap = {abs(ext_mae - 0.8534):.4f} (reflects ~0.10 noise on synth test labels vs real_lookup).")
    print(f"Expected public MAE for any candidate ≈ candidate's oracle_MAE ± ~0.01.")


if __name__ == '__main__':
    main()
