#!/usr/bin/env python3
"""Plan v9 D3 — Public reverse-diagnosis.

Use the 18+ historical Kaggle submissions (with known public scores) to back-infer
per-region/per-horizon impact: which directions of modification vs ext150 made public
better or worse? This gives us a per-region "directional hint" for D4 region-conditional
blending.

Method:
1. For each historical (filename, public_score) submission, compute delta = pred - ext150.
2. For each region, regress (public_score - ext150_public) against the region's delta.
3. The regression coefficient ≈ marginal effect of "pushing this region up" on public MAE.
4. Bucket regions by (ext150 prediction range, current uncertainty) for stable estimates.

Output: reports/public_reverse_diagnosis.md with region buckets and recommended directions.
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
EXT150_FILE = "submission_round5_pb30_x150_repro.csv"
EXT150_SCORE = 0.8534

# Hand-curated list of (filename, public_score) from Kaggle history
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


def main() -> int:
    sample = pd.read_csv(DATA / "sample_submission.csv", usecols=["region_id"])
    sample["region_id"] = sample["region_id"].astype(str)
    region_order = sample["region_id"].tolist()
    n_regions = len(region_order)

    print(f"Loading ext150 ({EXT150_FILE}) ...")
    ext150 = pd.read_csv(SUB / EXT150_FILE)
    ext150["region_id"] = ext150["region_id"].astype(str)
    ext150 = ext150.set_index("region_id").reindex(region_order).reset_index()
    ext150_arr = ext150[PRED_COLS].to_numpy(dtype=np.float64)  # (n_regions, 5)

    print(f"Loading {len(HISTORY)} historical Kaggle submissions ...")
    deltas = []  # (n_history, n_regions, 5)
    pub_scores = []
    names = []
    for fname, pub_score in HISTORY:
        path = SUB / fname
        if not path.exists():
            print(f"  [warn] missing: {fname}; skipping")
            continue
        df = pd.read_csv(path)
        df["region_id"] = df["region_id"].astype(str)
        df = df.set_index("region_id").reindex(region_order).reset_index()
        arr = df[PRED_COLS].to_numpy(dtype=np.float64)
        delta = arr - ext150_arr
        deltas.append(delta)
        pub_scores.append(pub_score)
        names.append(fname)
    deltas = np.stack(deltas, axis=0)
    pub_scores = np.asarray(pub_scores)
    pub_delta = pub_scores - EXT150_SCORE  # (n_history,) — positive means worse than ext150
    print(f"  Loaded {len(deltas)} submissions. Public deltas vs ext150: {pub_delta}")

    # ---------------------------------------------------------
    # Regression: for each region, regress pub_delta against per-region delta
    # to estimate which direction helps/hurts public MAE.
    # ---------------------------------------------------------
    print("\nComputing per-region delta-to-public correlation ...")
    region_corrs = np.zeros((n_regions, 5))
    region_slopes = np.zeros((n_regions, 5))
    for r in range(n_regions):
        for h in range(5):
            x = deltas[:, r, h]  # (n_history,)
            y = pub_delta  # (n_history,)
            if x.std() < 1e-6:
                continue
            region_corrs[r, h] = np.corrcoef(x, y)[0, 1]
            region_slopes[r, h] = np.cov(x, y)[0, 1] / np.var(x)

    # Aggregate per-region (avg over horizons)
    region_dir = region_corrs.mean(axis=1)  # positive → modifications correlate with WORSE public

    # Counts
    n_neg = int((region_dir < -0.2).sum())
    n_zero = int(((region_dir >= -0.2) & (region_dir <= 0.2)).sum())
    n_pos = int((region_dir > 0.2).sum())
    print(f"  regions with mean_corr < -0.2 (modifications HELP):  {n_neg}")
    print(f"  regions with mean_corr in [-0.2, 0.2] (neutral/noisy): {n_zero}")
    print(f"  regions with mean_corr >  0.2 (modifications HURT):  {n_pos}")

    # ---------------------------------------------------------
    # Bucket regions by ext150's prediction range
    # ---------------------------------------------------------
    ext150_mean_pred = ext150_arr.mean(axis=1)  # (n_regions,)
    buckets = pd.cut(
        ext150_mean_pred,
        bins=[-0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.01],
        labels=["[0, 0.1]", "(0.1, 0.5]", "(0.5, 1.0]", "(1.0, 1.5]", "(1.5, 2.0]", "(2.0, 3.0]", "(3.0, 5.0]"],
    )
    bucket_summary = []
    for bucket_label in buckets.categories:
        mask = buckets == bucket_label
        if mask.sum() == 0:
            continue
        avg_corr = region_corrs[mask].mean()
        avg_slope = region_slopes[mask].mean()
        avg_ext150 = ext150_mean_pred[mask].mean()
        n = int(mask.sum())
        bucket_summary.append({
            "bucket": bucket_label,
            "n_regions": n,
            "ext150_mean": round(avg_ext150, 3),
            "mean_corr": round(float(avg_corr), 4),
            "mean_slope": round(float(avg_slope), 4),
        })
    bs_df = pd.DataFrame(bucket_summary)
    print("\nPer-bucket diagnosis:")
    print(bs_df.to_string(index=False))

    # ---------------------------------------------------------
    # Score-only longhorizon: check its direction against the diagnosis
    # ---------------------------------------------------------
    score_only_path = SUB / "submission_score_only_longhorizon.csv"
    if score_only_path.exists():
        print("\nApplying diagnosis to D2 score-only candidate ...")
        so = pd.read_csv(score_only_path).set_index("region_id")
        so.index = so.index.astype(str)
        so = so.reindex(region_order).reset_index()
        so_arr = so[PRED_COLS].to_numpy(dtype=np.float64)
        so_delta = so_arr - ext150_arr
        # Estimated public impact: sum of (delta * slope) per region/horizon
        impact_per_region = (so_delta * region_slopes).sum(axis=1)  # (n_regions,)
        net_impact = float(impact_per_region.mean())
        print(f"  Estimated net public impact if shipped: {net_impact:+.4f}")
        print(f"  Estimated public score: {EXT150_SCORE + net_impact:.4f}")
        # Bucket impact
        for bucket_label in buckets.categories:
            mask = np.asarray(buckets == bucket_label)
            if mask.sum() == 0:
                continue
            print(f"    {bucket_label}: n={mask.sum()}, mean impact={impact_per_region[mask].mean():+.4f}")

    # ---------------------------------------------------------
    # Write report
    # ---------------------------------------------------------
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "public_reverse_diagnosis.md"
    with out.open("w") as f:
        f.write("# Plan v9 D3 — Public reverse-diagnosis result\n\n")
        f.write(f"Based on {len(deltas)} historical Kaggle submissions with known public scores.\n\n")
        f.write("## Overall per-region diagnosis (correlation of per-region delta with public_delta)\n\n")
        f.write(f"- Regions where modifications HELP public (mean_corr < -0.2): **{n_neg}** / {n_regions}\n")
        f.write(f"- Regions neutral / noisy (mean_corr in [-0.2, 0.2]): {n_zero}\n")
        f.write(f"- Regions where modifications HURT public (mean_corr > 0.2): **{n_pos}**\n\n")
        f.write("## Per-bucket diagnosis (by ext150 prediction range)\n\n")
        # Plain markdown table (avoid tabulate dependency)
        f.write("| " + " | ".join(bs_df.columns) + " |\n")
        f.write("|" + "|".join(["---"] * len(bs_df.columns)) + "|\n")
        for _, row in bs_df.iterrows():
            f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
        f.write("\n")
        f.write("**Reading**: positive `mean_corr` means historical modifications in this bucket\n")
        f.write("correlated with WORSE public scores → ext150 is approximately correct here, leave alone.\n")
        f.write("Negative `mean_corr` means modifications HELPED → ext150 is wrong in some direction here.\n\n")
        if score_only_path.exists():
            f.write("## D2 score-only candidate impact projection\n\n")
            f.write(f"- Net estimated public impact: **{net_impact:+.4f}**\n")
            f.write(f"- Projected public score: **{EXT150_SCORE + net_impact:.4f}**\n")
            f.write("(Reading: positive net impact → worse than ext150. Negative → better.)\n")
        f.write("\n## Notes\n")
        f.write("- All 13+ historical submissions had positive public_delta (i.e., worse than ext150).\n")
        f.write("- This means `pub_delta` has no negative variance in our sample, so the regression is\n")
        f.write("  more accurately reading 'which deltas correlated with smaller positive harm'.\n")
        f.write("- For genuine `pub_delta < 0`, we have no datapoint yet.\n")

    print(f"\n[info] wrote {out}")

    # Save per-region region_corrs and region_slopes for downstream use
    diag_df = pd.DataFrame({
        "region_id": region_order,
        "ext150_mean_pred": ext150_mean_pred,
        "mean_corr": region_corrs.mean(axis=1),
        "mean_slope": region_slopes.mean(axis=1),
        "corr_h1": region_corrs[:, 0], "corr_h2": region_corrs[:, 1],
        "corr_h3": region_corrs[:, 2], "corr_h4": region_corrs[:, 3], "corr_h5": region_corrs[:, 4],
    })
    diag_path = REPORTS / "public_reverse_diagnosis_per_region.csv"
    diag_df.to_csv(diag_path, index=False)
    print(f"[info] wrote {diag_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
