#!/usr/bin/env python3
"""LGBM with lagged score features only — pure score-history signal, LGBM family."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
SUB = PROJECT_ROOT / "submissions"
PRED_COLS = [f"pred_week{i+1}" for i in range(5)]
REGION_COL, DATE_COL, TARGET_COL = "region_id", "date", "score"


def fill_score_nan(s):
    out = s.copy()
    last = 0.0
    for i in range(len(out)):
        if np.isnan(out[i]): out[i] = last
        else: last = out[i]
    return out


def lag_features(ctx_s):
    """ctx_s is filled. Return [lag1, lag2, lag4, lag8, lag13, lag26, lag52, mean_4, mean_12, mean_26, mean_52, ...stats]"""
    K = len(ctx_s)
    feats = []
    for lag in (1, 2, 3, 4, 6, 8, 13, 26, 52, 78, 104):
        feats.append(ctx_s[-lag] if K >= lag else 0.0)
    for win in (4, 12, 26, 52, 104):
        if K >= win:
            v = ctx_s[-win:]
            feats.extend([v.mean(), v.std(), v.min(), v.max(), v.max()-v.min()])
        else:
            feats.extend([0.0]*5)
    # First difference patterns (recent dynamics)
    if K >= 2:
        diffs = np.diff(ctx_s[-13:])
        feats.extend([diffs.mean(), diffs.std(), float(diffs[-1])])
    else:
        feats.extend([0.0]*3)
    return np.array(feats, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=114)
    p.add_argument("--context-weeks", type=int, default=104)
    p.add_argument("--horizons", type=int, default=5)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--save-candidate", required=True)
    args = p.parse_args()

    train = pd.read_csv(DATA/"train.csv", usecols=[REGION_COL, DATE_COL, TARGET_COL])
    train[REGION_COL] = train[REGION_COL].astype(str)
    train[TARGET_COL] = pd.to_numeric(train[TARGET_COL], errors="coerce")
    sample = pd.read_csv(DATA/"sample_submission.csv", usecols=[REGION_COL])
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    region_order = sample[REGION_COL].tolist()

    region_arrays = {}
    for r, g in train.groupby(REGION_COL, sort=False):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        ds = g[TARGET_COL].to_numpy(dtype=np.float32)
        nw = len(ds)//7
        if nw == 0: continue
        weekly_s = np.full(nw, np.nan, dtype=np.float32)
        for w in range(nw):
            sv = ds[w*7:(w+1)*7]; sv = sv[np.isfinite(sv)]
            if sv.size: weekly_s[w] = float(sv[0])
        region_arrays[str(r)] = weekly_s

    # Build training samples
    print("Building...")
    Xs, ys = [], []
    for r, ws in region_arrays.items():
        n = len(ws)
        if n < args.context_weeks + args.horizons: continue
        for a in range(args.context_weeks - 1, n - args.horizons):
            ctx_s = fill_score_nan(ws[a-args.context_weeks+1:a+1])
            tgt = ws[a+1:a+1+args.horizons]
            mask = np.isfinite(tgt)
            if not mask.any(): continue
            feats = lag_features(ctx_s)
            Xs.append(feats)
            ys.append(np.where(mask, tgt, np.nan))
    X = np.stack(Xs)
    Y = np.stack(ys)
    print(f"  {X.shape[0]:,} samples, {X.shape[1]} features")

    models = []
    for h in range(args.horizons):
        ymask = np.isfinite(Y[:, h])
        m = lgb.LGBMRegressor(
            num_leaves=args.num_leaves, n_estimators=args.n_estimators,
            learning_rate=args.lr, objective="regression_l1",
            seed=args.seed, verbose=-1,
        )
        m.fit(X[ymask], Y[ymask, h])
        models.append(m)
    print("  trained")

    out_rows = []
    for r in region_order:
        if r not in region_arrays:
            out_rows.append([r] + [0.5]*args.horizons)
            continue
        ws = region_arrays[r]
        full_s = ws.copy()
        # For inference, no test scores available
        ctx_s = fill_score_nan(full_s[-args.context_weeks:])
        if len(ctx_s) < args.context_weeks:
            pad = args.context_weeks - len(ctx_s)
            ctx_s = np.pad(ctx_s, (pad, 0), mode="edge")
        f = lag_features(ctx_s)
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
