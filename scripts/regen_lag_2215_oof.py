#!/usr/bin/env python3
"""Regenerate lag-2215 OOF predictions aligned to oof_tensor.csv row_index.

Mirrors train_deep_model.py validation set construction (so row_index aligns):
- region_order from sample_submission.csv
- For each region: anchors = [last_usable - d for d in [728, 735, 742] if last_usable - d >= 91]
  where last_usable = len(group) - 36
- For each (region, anchor, horizon h=1..5):
    target_h = h-th non-null score after anchor (calendar position t_h)
    pred_lag2215(h) = score[t_h - 2215]  (fallback: last observed score at anchor)

Output schema matches deep_cnn_*_validation_predictions.csv so the result joins
into oof_tensor on (row_index, region_id, horizon).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train.csv"
SAMPLE = ROOT / "data" / "sample_submission.csv"
OUT = ROOT / "reports" / "lag_2215_oof_validation_predictions.csv"

REGION_COL = "region_id"
TARGET_COL = "score"
VALID_DELTAS = [728, 735, 742]
LAG_DAYS = 2215
N_HORIZONS = 5


def main() -> None:
    print(f"[lag2215] loading {TRAIN}")
    train = pd.read_csv(TRAIN)
    sample = pd.read_csv(SAMPLE)
    region_order = sample[REGION_COL].astype(str).tolist()

    # Group once for speed
    groups = {str(r): g.reset_index(drop=True) for r, g in train.groupby(REGION_COL, sort=False)}

    rows: list[dict] = []
    anchor_counter = 0
    skipped_short_region = 0
    skipped_no_targets = 0
    fallback_count = 0
    for region in region_order:
        g = groups.get(region)
        if g is None or len(g) == 0:
            skipped_short_region += 1
            continue
        score = pd.to_numeric(g[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)
        score_idx = np.flatnonzero(np.isfinite(score)).astype(np.int32)
        score_values = score[score_idx]
        last_usable = len(g) - 36
        if last_usable < 91:
            skipped_short_region += 1
            continue
        valid_anchors = [int(last_usable - d) for d in VALID_DELTAS if last_usable - d >= 91]
        for anchor in valid_anchors:
            after_mask = score_idx > anchor
            if after_mask.sum() < N_HORIZONS:
                skipped_no_targets += 1
                continue
            after_idx = np.where(after_mask)[0][:N_HORIZONS]
            target_positions = score_idx[after_idx]
            targets = score_values[after_idx]

            # Anchor's most recent observed score (fallback)
            le_mask = score_idx <= anchor
            anchor_last = float(score_values[le_mask][-1]) if le_mask.any() else 0.0

            for h in range(N_HORIZONS):
                t_h = int(target_positions[h])
                lookup_pos = t_h - LAG_DAYS
                # Find the nearest non-null score on or before lookup_pos (same week 6 yrs prior).
                # Scores are weekly (~1/7 days), so lookup_pos itself usually lands on NaN.
                if lookup_pos >= 0:
                    le = score_idx[score_idx <= lookup_pos]
                    if le.size > 0:
                        pred = float(score[le[-1]])
                    else:
                        pred = anchor_last
                        fallback_count += 1
                else:
                    pred = anchor_last
                    fallback_count += 1
                rows.append({
                    "row_index": anchor_counter,
                    "region_id": region,
                    "horizon": h + 1,
                    "y_true": float(targets[h]),
                    "pred_raw": pred,
                    "pred_horizon_calibrated": pred,
                    "pred_final_calibrated": pred,
                })
            anchor_counter += 1

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    mae = float(np.mean(np.abs(df.pred_final_calibrated - df.y_true)))
    print(
        f"[lag2215] wrote {OUT}: anchors={anchor_counter} rows={len(df)} "
        f"skipped_short={skipped_short_region} skipped_no_targets={skipped_no_targets} "
        f"fallback={fallback_count}"
    )
    print(f"[lag2215] OOF MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
