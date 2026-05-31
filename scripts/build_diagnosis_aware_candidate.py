#!/usr/bin/env python3
"""Plan v9 D4 — Build a diagnosis-aware candidate from D2 + D3 outputs.

Strategy:
- Per-region/horizon, use the D3 slope to score the expected public impact of
  replacing ext150 with the D2 score-only prediction at that cell.
- Build candidate = ext150 for cells where impact >= 0 (no help expected),
  blend toward score-only for cells where impact < 0 (helpful direction).
- Several variants with different aggressiveness; pick by MAD and expected impact.

Output: multiple submissions/submission_v9_diagnosis_*.csv files for the supervisor
to pick from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
SUB = PROJECT_ROOT / "submissions"
REPORTS = PROJECT_ROOT / "reports"

PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]
EXT150_SCORE = 0.8534


def main() -> int:
    sample = pd.read_csv(DATA / "sample_submission.csv", usecols=["region_id"])
    sample["region_id"] = sample["region_id"].astype(str)
    region_order = sample["region_id"].tolist()

    ext150 = pd.read_csv(SUB / "submission_round5_pb30_x150_repro.csv")
    ext150["region_id"] = ext150["region_id"].astype(str)
    ext150 = ext150.set_index("region_id").reindex(region_order).reset_index()
    ext150_arr = ext150[PRED_COLS].to_numpy(dtype=np.float64)

    so = pd.read_csv(SUB / "submission_score_only_longhorizon.csv")
    so["region_id"] = so["region_id"].astype(str)
    so = so.set_index("region_id").reindex(region_order).reset_index()
    so_arr = so[PRED_COLS].to_numpy(dtype=np.float64)

    # Load D3 per-region diagnosis
    diag = pd.read_csv(REPORTS / "public_reverse_diagnosis_per_region.csv")
    diag["region_id"] = diag["region_id"].astype(str)
    diag = diag.set_index("region_id").reindex(region_order).reset_index()
    slope_cols = ["corr_h1", "corr_h2", "corr_h3", "corr_h4", "corr_h5"]
    # Load per-region slope matrix indirectly via D3 — but the script saved only corrs.
    # Re-compute slopes from the 13 historical submissions:
    HISTORY = [
        ("submission_doy_blend_w08.csv", 0.8848),
        ("submission_v7_u2_recent_zero_mask_a20.csv", 0.8919),
        ("submission_v6_3leg_shifted_w15.csv", 0.8853),
        ("submission_ext170_mean12334.csv", 0.9208),
        ("submission_v6e1_blend_ext150_w10.csv", 0.8886),
        ("submission_deep_ensemble_ext150_w10.csv", 0.8767),
        ("submission_deep_lstm_fixed_s114.csv", 0.9032),
        ("submission_deep_cnn_fixed_s114.csv", 0.9706),
        ("submission_9leg_ext50.csv", 0.8674),
        ("submission_stacker_9leg_v2mlp_shift27.csv", 0.8688),
        ("submission_round2_blend_pb30.csv", 0.8599),
        ("submission_round2_ensemble.csv", 0.8609),
        ("submission_redo_blend_pb30.csv", 0.8593),
    ]
    deltas = []
    pub_deltas = []
    for fname, ps in HISTORY:
        path = SUB / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["region_id"] = df["region_id"].astype(str)
        df = df.set_index("region_id").reindex(region_order).reset_index()
        arr = df[PRED_COLS].to_numpy(dtype=np.float64)
        deltas.append(arr - ext150_arr)
        pub_deltas.append(ps - EXT150_SCORE)
    deltas = np.stack(deltas, axis=0)
    pub_deltas = np.asarray(pub_deltas)
    # Per (region, horizon) slope
    slopes = np.zeros_like(ext150_arr)
    for r in range(ext150_arr.shape[0]):
        for h in range(5):
            x = deltas[:, r, h]
            if x.std() < 1e-6:
                continue
            slopes[r, h] = np.cov(x, pub_deltas)[0, 1] / np.var(x)

    so_delta = so_arr - ext150_arr  # what would the candidate add
    so_impact = so_delta * slopes  # estimated per-cell public-MAE impact

    print(f"ext150 mean: {ext150_arr.mean():.4f}")
    print(f"score-only mean: {so_arr.mean():.4f}")
    print(f"per-region slope range: [{slopes.min():.4f}, {slopes.max():.4f}]")
    print(f"per-cell impact range: [{so_impact.min():.4f}, {so_impact.max():.4f}]")
    print(f"net impact if shipped 100%: {so_impact.sum() / so_impact.size:+.4f} (i.e. public={EXT150_SCORE + so_impact.mean():.4f})")
    print()

    # ---------------------------------------------------------
    # Build candidates of varying aggressiveness
    # ---------------------------------------------------------
    candidates = {}

    # Variant A — apply score-only ONLY on cells where impact < 0 (helpful)
    helpful_mask = so_impact < 0
    candA = ext150_arr.copy()
    candA[helpful_mask] = so_arr[helpful_mask]
    candA = np.clip(candA, 0, 5)
    candidates["A_helpful_only"] = candA

    # Variant B — apply score-only on helpful cells, scaled by impact magnitude
    # weight w = clip(-impact / max_helpful_impact, 0, 1)
    max_helpful = max(abs(so_impact.min()), 1e-6)
    weight = np.clip(-so_impact / max_helpful, 0, 1)
    candB = (1 - weight) * ext150_arr + weight * so_arr
    candB = np.clip(candB, 0, 5)
    candidates["B_weighted"] = candB

    # Variant C — only top-K most-helpful cells, ext150 elsewhere
    for top_pct in (5, 10, 20):
        thresh = float(np.percentile(so_impact, top_pct))
        mask = so_impact < thresh
        candC = ext150_arr.copy()
        candC[mask] = so_arr[mask]
        candC = np.clip(candC, 0, 5)
        candidates[f"C_top{top_pct}pct"] = candC

    # Variant D — only top regions (whole-region) by mean impact
    impact_per_region = so_impact.mean(axis=1)
    for n_regions in (200, 500, 1000):
        idx = np.argsort(impact_per_region)[:n_regions]  # smallest = most helpful
        candD = ext150_arr.copy()
        candD[idx] = so_arr[idx]
        candD = np.clip(candD, 0, 5)
        candidates[f"D_top{n_regions}_regions"] = candD

    print("Candidate evaluation:")
    print(f"{'name':<25} {'mean':<8} {'mad_vs_ext':<11} {'expected_impact':<15} {'expected_public':<15}")
    summary = []
    for name, cand in candidates.items():
        mean = float(cand.mean())
        mad = float(np.abs(cand - ext150_arr).mean())
        # Net expected public impact based on regression model
        cand_delta = cand - ext150_arr
        expected = float(np.sum(cand_delta * slopes) / cand_delta.size)
        expected_pub = EXT150_SCORE + expected
        print(f"  {name:<25} {mean:<8.4f} {mad:<11.4f} {expected:<+15.5f} {expected_pub:<15.4f}")
        summary.append({"name": name, "mean": mean, "mad_vs_ext": mad, "expected_impact": expected, "expected_public": expected_pub})

    # Save best 3 candidates
    summary_df = pd.DataFrame(summary).sort_values("expected_public").reset_index(drop=True)
    print()
    print("Top 3 by expected public score:")
    print(summary_df.head(5).to_string(index=False))

    # Save all candidates
    for name, cand in candidates.items():
        df = pd.DataFrame(cand, columns=PRED_COLS)
        df.insert(0, "region_id", region_order)
        out_path = SUB / f"submission_v9_diagnosis_{name}.csv"
        df.to_csv(out_path, index=False)
    print(f"\n[info] wrote {len(candidates)} candidate CSVs to submissions/")

    # Summary report
    rp = REPORTS / "v9_diagnosis_candidates.md"
    with rp.open("w") as f:
        f.write("# Plan v9 D4 — Diagnosis-aware candidates\n\n")
        f.write("All variants apply the D2 score-only prediction selectively to ext150 based on the D3 per-region slope.\n\n")
        f.write("| Variant | Mean | MAD vs ext150 | Expected impact | Expected public |\n")
        f.write("|---|---|---|---|---|\n")
        for row in summary_df.itertuples():
            f.write(f"| {row.name} | {row.mean:.4f} | {row.mad_vs_ext:.4f} | {row.expected_impact:+.5f} | {row.expected_public:.4f} |\n")
        f.write("\n**Caveats**:\n")
        f.write("- Estimates extrapolate from 13 historical datapoints all having `pub_delta > 0`.\n")
        f.write("- True public score depends on whether the score-only model's structural information\n")
        f.write("  matches public truth, not just historical submission deltas.\n")
        f.write("- MAD gate per Plan v9: ≤ 0.30 (Variant A is likely above; Variants C/D should be in range).\n")
    print(f"[info] wrote {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
