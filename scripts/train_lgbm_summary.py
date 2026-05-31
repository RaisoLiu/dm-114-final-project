#!/usr/bin/env python3
"""LGBM on weather summary stats — distinct from existing pb30 LGBM in feature set."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

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


def fill_score_nan(s):
    out = s.copy()
    last = 0.0
    for i in range(len(out)):
        if np.isnan(out[i]): out[i] = last
        else: last = out[i]
    return out


def summary_features(ctx_w, ctx_s):
    """K weeks of weather + score → fixed-size feature vector."""
    feats = []
    K = ctx_w.shape[0]
    for win in (4, 12, 26, 52):
        if K >= win:
            w = ctx_w[-win:]; s = ctx_s[-win:]
            feats.append(w.mean(0))
            feats.append(w.std(0))
            feats.append(w.min(0))
            feats.append(w.max(0))
            feats.append(np.array([s.mean(), s.std(), s.min(), s.max(), s[-1]], dtype=np.float32))
        else:
            feats.append(np.zeros(14, dtype=np.float32))
            feats.append(np.zeros(14, dtype=np.float32))
            feats.append(np.zeros(14, dtype=np.float32))
            feats.append(np.zeros(14, dtype=np.float32))
            feats.append(np.zeros(5, dtype=np.float32))
    return np.concatenate(feats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=114)
    p.add_argument("--context-weeks", type=int, default=104)
    p.add_argument("--horizons", type=int, default=5)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--save-candidate", required=True)
    args = p.parse_args()

    train = pd.read_csv(DATA/"train.csv", usecols=[REGION_COL, DATE_COL, TARGET_COL]+WEATHER_COLS)
    train[REGION_COL] = train[REGION_COL].astype(str)
    train[TARGET_COL] = pd.to_numeric(train[TARGET_COL], errors="coerce")
    test = pd.read_csv(DATA/"test.csv", usecols=[REGION_COL, DATE_COL]+WEATHER_COLS)
    test[REGION_COL] = test[REGION_COL].astype(str)
    sample = pd.read_csv(DATA/"sample_submission.csv", usecols=[REGION_COL])
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    region_order = sample[REGION_COL].tolist()

    region_arrays = {}
    test_weekly = {}
    for region, g in train.groupby(REGION_COL, sort=False):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        dw = g[WEATHER_COLS].to_numpy(dtype=np.float32)
        ds = g[TARGET_COL].to_numpy(dtype=np.float32)
        nw = dw.shape[0]//7
        if nw == 0: continue
        weekly_w = dw[:nw*7].reshape(nw,7,14).mean(1).astype(np.float32)
        weekly_s = np.full(nw, np.nan, dtype=np.float32)
        for w in range(nw):
            sv = ds[w*7:(w+1)*7]; sv = sv[np.isfinite(sv)]
            if sv.size: weekly_s[w] = float(sv[0])
        region_arrays[str(region)] = {"weather": weekly_w, "score": weekly_s}
    for region, g in test.groupby(REGION_COL, sort=False):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        dw = g[WEATHER_COLS].to_numpy(dtype=np.float32)
        nw = dw.shape[0]//7
        if nw > 0: test_weekly[str(region)] = dw[:nw*7].reshape(nw,7,14).mean(1).astype(np.float32)

    all_w = np.concatenate([a["weather"] for a in region_arrays.values()], axis=0)
    wm, ws = all_w.mean(0), all_w.std(0)+1e-6
    for r in region_arrays: region_arrays[r]["weather"] = (region_arrays[r]["weather"]-wm)/ws
    for r in test_weekly: test_weekly[r] = (test_weekly[r]-wm)/ws

    # Build training samples
    print("Building training samples...")
    Xs, ys, region_ids = [], [], []
    for r, arrs in region_arrays.items():
        n = arrs["weather"].shape[0]
        if n < args.context_weeks + args.horizons: continue
        for a in range(args.context_weeks - 1, n - args.horizons):
            ctx_w = arrs["weather"][a-args.context_weeks+1:a+1]
            ctx_s = fill_score_nan(arrs["score"][a-args.context_weeks+1:a+1])
            tgt = arrs["score"][a+1:a+1+args.horizons]
            mask = np.isfinite(tgt)
            if not mask.any(): continue
            feats = summary_features(ctx_w, ctx_s)
            Xs.append(feats)
            ys.append(np.where(mask, tgt, np.nan))
            region_ids.append(r)
    X = np.stack(Xs)
    Y = np.stack(ys)
    print(f"  {X.shape[0]:,} samples, {X.shape[1]} features")

    # Train per-horizon
    models = []
    for h in range(args.horizons):
        ymask = np.isfinite(Y[:, h])
        Xh = X[ymask]
        yh = Y[ymask, h]
        print(f"  Horizon {h+1}: {len(yh):,} samples")
        m = lgb.LGBMRegressor(
            num_leaves=args.num_leaves, n_estimators=args.n_estimators,
            learning_rate=args.lr, objective="regression_l1",
            seed=args.seed, verbose=-1,
        )
        m.fit(Xh, yh)
        models.append(m)

    # Inference
    out_rows = []
    for r in region_order:
        if r not in region_arrays:
            out_rows.append([r] + [0.5]*args.horizons)
            continue
        arrs = region_arrays[r]
        tw = arrs["weather"]; ts = arrs["score"]
        test_w = test_weekly.get(r)
        if test_w is not None:
            full_w = np.concatenate([tw, test_w], axis=0)
            last_s = ts[~np.isnan(ts)][-1] if np.isfinite(ts).any() else 0.0
            full_s = np.concatenate([ts, np.full(test_w.shape[0], last_s, dtype=np.float32)])
        else:
            full_w = tw; full_s = ts
        ctx_w = full_w[-args.context_weeks:]
        cs = fill_score_nan(full_s[-args.context_weeks:])
        if ctx_w.shape[0] < args.context_weeks:
            pad = args.context_weeks - ctx_w.shape[0]
            ctx_w = np.pad(ctx_w, ((pad,0),(0,0)), mode="edge")
            cs = np.pad(cs, (pad,0), mode="edge")
        f = summary_features(ctx_w, cs)
        preds = [float(np.clip(m.predict(f.reshape(1,-1))[0], 0, 5)) for m in models]
        out_rows.append([r] + preds)

    out_df = pd.DataFrame(out_rows, columns=[REGION_COL] + PRED_COLS[:args.horizons])
    out_df[REGION_COL] = out_df[REGION_COL].astype(str)

    ext = pd.read_csv(SUB/"submission_round5_pb30_x150_repro.csv")
    ext[REGION_COL] = ext[REGION_COL].astype(str)
    ext = ext.set_index(REGION_COL).reindex(region_order).reset_index()
    e = ext[PRED_COLS[:args.horizons]].to_numpy()
    common = list(set(out_df[REGION_COL]) & set(region_order))
    o = out_df.set_index(REGION_COL).loc[common][PRED_COLS[:args.horizons]].to_numpy()
    e_idx = [region_order.index(r) for r in common]
    ec = e[e_idx]
    mad_raw = float(np.abs(o - ec).mean())
    rhos = [float(np.corrcoef(o[:,h], ec[:,h])[0,1]) for h in range(args.horizons)]
    o_aligned = o.copy()
    for h in range(args.horizons):
        if o[:,h].std() > 0.01:
            o_aligned[:,h] = ec[:,h].mean() + (o[:,h]-o[:,h].mean()) * (ec[:,h].std()/o[:,h].std())
    o_aligned = np.clip(o_aligned, 0, 5)
    blend = np.clip(0.05*o_aligned + 0.95*ec, 0, 5)
    mad_blend = float(np.abs(blend - ec).mean())
    print(f"  raw MAD={mad_raw:.4f} Pearson={np.mean(rhos):.4f}")
    print(f"  5% blend MAD={mad_blend:.4f}  est@1.13={0.8534+1.13*mad_blend:.4f} est@0.5={0.8534+0.5*mad_blend:.4f}")
    idx_map = {r:i for i,r in enumerate(common)}
    out_full = np.zeros((len(region_order), args.horizons))
    for i, r in enumerate(region_order):
        if r in idx_map: out_full[i] = blend[idx_map[r]]
        else: out_full[i] = e[i]
    df = pd.DataFrame(out_full, columns=PRED_COLS[:args.horizons])
    df.insert(0, REGION_COL, region_order)
    df.to_csv(args.save_candidate, index=False)
    print(f"[info] saved {args.save_candidate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
