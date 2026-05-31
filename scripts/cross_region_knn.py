#!/usr/bin/env python3
"""Cross-region k-NN: for each region, find k most similar regions by recent weather
pattern, use their observed scores as prediction signal."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
SUB = PROJECT_ROOT / "submissions"
PRED_COLS = [f"pred_week{i+1}" for i in range(5)]
REGION_COL, DATE_COL, TARGET_COL = "region_id", "date", "score"
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--fingerprint-weeks", type=int, default=26)
    p.add_argument("--save-candidate", required=True)
    args = p.parse_args()

    print("Loading...")
    train = pd.read_csv(DATA/"train.csv", usecols=[REGION_COL, DATE_COL, TARGET_COL]+WEATHER_COLS)
    train[REGION_COL] = train[REGION_COL].astype(str)
    train[TARGET_COL] = pd.to_numeric(train[TARGET_COL], errors="coerce")
    test = pd.read_csv(DATA/"test.csv", usecols=[REGION_COL, DATE_COL]+WEATHER_COLS)
    test[REGION_COL] = test[REGION_COL].astype(str)
    sample = pd.read_csv(DATA/"sample_submission.csv", usecols=[REGION_COL])
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    region_order = sample[REGION_COL].tolist()

    # Build per-region recent weather fingerprint (last F weeks of train data)
    print("Building fingerprints...")
    region_arrays = {}
    test_weekly = {}
    region_last_scores = {}
    for r, g in train.groupby(REGION_COL, sort=False):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        dw = g[WEATHER_COLS].to_numpy(dtype=np.float32)
        ds = g[TARGET_COL].to_numpy(dtype=np.float32)
        nw = dw.shape[0]//7
        if nw == 0: continue
        weekly_w = dw[:nw*7].reshape(nw,7,14).mean(1).astype(np.float32)
        weekly_s = np.full(nw, np.nan, dtype=np.float32)
        for w in range(nw):
            sv = ds[w*7:(w+1)*7]
            sv = sv[np.isfinite(sv)]
            if sv.size: weekly_s[w] = float(sv[0])
        region_arrays[str(r)] = {"weather": weekly_w, "score": weekly_s}
        # Last observed scores (weekly indexed)
        last_obs_idx = np.where(np.isfinite(weekly_s))[0]
        if last_obs_idx.size:
            region_last_scores[str(r)] = (last_obs_idx[-1], float(weekly_s[last_obs_idx[-1]]))

    for r, g in test.groupby(REGION_COL, sort=False):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        dw = g[WEATHER_COLS].to_numpy(dtype=np.float32)
        nw = dw.shape[0]//7
        if nw > 0:
            test_weekly[str(r)] = dw[:nw*7].reshape(nw,7,14).mean(1).astype(np.float32)

    # Normalize weather
    all_w = np.concatenate([a["weather"] for a in region_arrays.values()], axis=0)
    wm, ws = all_w.mean(0), all_w.std(0)+1e-6
    for r in region_arrays: region_arrays[r]["weather"] = (region_arrays[r]["weather"]-wm)/ws
    for r in test_weekly: test_weekly[r] = (test_weekly[r]-wm)/ws

    # Build fingerprint = mean of last F weeks of train weather (per region)
    # Also extract last score per region
    F = args.fingerprint_weeks
    print(f"Fingerprint = mean of last {F} weeks of weather per region")
    fingerprints = []
    fp_regions = []
    for r in region_order:
        if r not in region_arrays: continue
        ww = region_arrays[r]["weather"]
        if ww.shape[0] >= F:
            fp = ww[-F:].mean(0)
        else:
            fp = ww.mean(0) if ww.shape[0] > 0 else np.zeros(14)
        fingerprints.append(fp)
        fp_regions.append(r)
    fingerprints = np.stack(fingerprints)
    print(f"  {fingerprints.shape[0]} regions with fingerprints")

    # Pre-compute future-score "vectors" per donor region:
    # For each donor region, what was its score at horizon h relative to its last observation?
    # = mean of last K observed score steps (use as climatology-anchored future)
    # Simplest: for each donor, mean of recent observed scores as "average" future
    donor_avg_scores = np.zeros(len(fp_regions))
    for i, r in enumerate(fp_regions):
        ws_arr = region_arrays[r]["score"]
        valid = ws_arr[np.isfinite(ws_arr)]
        if valid.size:
            donor_avg_scores[i] = float(valid[-26:].mean() if len(valid) >= 26 else valid.mean())
        else:
            donor_avg_scores[i] = 1.2  # default

    # k-NN over fingerprints
    print(f"Computing k-NN with k={args.k}...")
    # Use normalized Euclidean distance
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=args.k + 1, metric="euclidean")  # +1 to exclude self
    nn.fit(fingerprints)
    dists, idxs = nn.kneighbors(fingerprints)

    # For each query region (in region_order), prediction = weighted average of k neighbors' avg scores
    out_rows = []
    fp_region_to_i = {r: i for i, r in enumerate(fp_regions)}
    for r in region_order:
        if r in fp_region_to_i:
            i = fp_region_to_i[r]
            # Exclude self (first neighbor is self) if present
            neighbor_idx = [j for j in idxs[i] if j != i][:args.k]
            if not neighbor_idx:
                neighbor_idx = list(idxs[i][1:args.k+1])
            # Weighted by inverse distance
            d_i = np.array([dists[i][list(idxs[i]).index(j)] for j in neighbor_idx])
            weights = 1.0 / (d_i + 1e-3)
            weights /= weights.sum()
            neighbor_scores = donor_avg_scores[neighbor_idx]
            pred_val = float((weights * neighbor_scores).sum())
            out_rows.append([r] + [pred_val] * 5)
        else:
            out_rows.append([r] + [0.5] * 5)

    out_df = pd.DataFrame(out_rows, columns=[REGION_COL] + PRED_COLS)
    out_df[REGION_COL] = out_df[REGION_COL].astype(str)

    ext = pd.read_csv(SUB/"submission_round5_pb30_x150_repro.csv")
    ext[REGION_COL] = ext[REGION_COL].astype(str)
    ext = ext.set_index(REGION_COL).reindex(region_order).reset_index()
    e = ext[PRED_COLS].to_numpy()
    common = list(set(out_df[REGION_COL]) & set(region_order))
    o = out_df.set_index(REGION_COL).loc[common][PRED_COLS].to_numpy()
    e_idx = [region_order.index(r) for r in common]
    ec = e[e_idx]
    mad_raw = float(np.abs(o - ec).mean())
    rhos = [float(np.corrcoef(o[:,h], ec[:,h])[0,1]) for h in range(5)]
    o_aligned = o.copy()
    for h in range(5):
        if o[:,h].std() > 0.01:
            o_aligned[:,h] = ec[:,h].mean() + (o[:,h]-o[:,h].mean()) * (ec[:,h].std()/o[:,h].std())
    o_aligned = np.clip(o_aligned, 0, 5)
    blend = np.clip(0.05*o_aligned + 0.95*ec, 0, 5)
    mad_blend = float(np.abs(blend - ec).mean())
    print(f"\n  raw MAD={mad_raw:.4f} Pearson={np.mean(rhos):.4f}")
    print(f"  5% blend MAD={mad_blend:.4f}  est@1.13={0.8534+1.13*mad_blend:.4f} est@0.5={0.8534+0.5*mad_blend:.4f}")

    idx_map = {r:i for i,r in enumerate(common)}
    out_full = np.zeros((len(region_order), 5))
    for i, r in enumerate(region_order):
        if r in idx_map: out_full[i] = blend[idx_map[r]]
        else: out_full[i] = e[i]
    df = pd.DataFrame(out_full, columns=PRED_COLS)
    df.insert(0, REGION_COL, region_order)
    df.to_csv(args.save_candidate, index=False)
    print(f"[info] saved {args.save_candidate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
