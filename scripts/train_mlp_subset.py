#!/usr/bin/env python3
"""MLP on weather summary stats with feature subsetting."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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


def summary_features(ctx_w, ctx_s, use_score=True):
    feats = []
    K = ctx_w.shape[0]
    for win in (4, 12, 26, 52):
        if K >= win:
            w = ctx_w[-win:]
            feats.append(w.mean(0))
            feats.append(w.std(0))
            feats.append(w.min(0))
            feats.append(w.max(0))
            if use_score:
                s = ctx_s[-win:]
                feats.append(np.array([s.mean(), s.std(), s.min(), s.max(), s[-1]], dtype=np.float32))
        else:
            feats.append(np.zeros(ctx_w.shape[1], dtype=np.float32))
            feats.append(np.zeros(ctx_w.shape[1], dtype=np.float32))
            feats.append(np.zeros(ctx_w.shape[1], dtype=np.float32))
            feats.append(np.zeros(ctx_w.shape[1], dtype=np.float32))
            if use_score:
                feats.append(np.zeros(5, dtype=np.float32))
    return np.concatenate(feats)


class SummaryDataset(Dataset):
    def __init__(self, region_arrays, context_weeks, horizons, region_to_idx, weather_cols_idx, use_score):
        self.cw, self.H = context_weeks, horizons
        self.region_arrays = region_arrays
        self.region_to_idx = region_to_idx
        self.w_idx = weather_cols_idx
        self.use_score = use_score
        self.samples = []
        for r, arrs in region_arrays.items():
            n = arrs["weather"].shape[0]
            if n >= context_weeks + horizons:
                for a in range(context_weeks - 1, n - horizons):
                    f = arrs["score"][a+1:a+1+horizons]
                    if np.isfinite(f).any(): self.samples.append((r, a))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        r, a = self.samples[idx]
        arrs = self.region_arrays[r]
        K, H = self.cw, self.H
        ctx_w = arrs["weather"][a-K+1:a+1][:, self.w_idx]
        ctx_s = fill_score_nan(arrs["score"][a-K+1:a+1])
        tgt = arrs["score"][a+1:a+1+H]
        mask = np.isfinite(tgt).astype(np.float32)
        tgt = np.nan_to_num(tgt, nan=0.0)
        feats = summary_features(ctx_w, ctx_s, use_score=self.use_score)
        return {
            "feats": torch.from_numpy(feats).float(),
            "tgt": torch.from_numpy(tgt).float(),
            "mask": torch.from_numpy(mask).float(),
            "region_idx": torch.tensor(self.region_to_idx[r], dtype=torch.long),
        }


class SummaryMLP(nn.Module):
    def __init__(self, in_dim, n_regions, hidden=256, n_layers=3, region_emb_dim=32, dropout=0.1, horizons=5):
        super().__init__()
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        layers = []
        prev = in_dim + region_emb_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, horizons))
        self.net = nn.Sequential(*layers)
    def forward(self, feats, ri):
        return self.net(torch.cat([feats, self.region_emb(ri)], dim=-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=114)
    p.add_argument("--context-weeks", type=int, default=104)
    p.add_argument("--horizons", type=int, default=5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--num-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--feature-cols", default="all", help="comma-separated weather col names or 'all' or 'no-score'")
    p.add_argument("--no-score", action="store_true", default=False)
    p.add_argument("--save-candidate", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    use_score = not args.no_score

    if args.feature_cols == "all":
        w_idx = list(range(len(WEATHER_COLS)))
    else:
        col_names = args.feature_cols.split(",")
        w_idx = [WEATHER_COLS.index(c) for c in col_names if c in WEATHER_COLS]

    train = pd.read_csv(DATA/"train.csv", usecols=[REGION_COL, DATE_COL, TARGET_COL]+WEATHER_COLS)
    train[REGION_COL] = train[REGION_COL].astype(str)
    train[TARGET_COL] = pd.to_numeric(train[TARGET_COL], errors="coerce")
    test = pd.read_csv(DATA/"test.csv", usecols=[REGION_COL, DATE_COL]+WEATHER_COLS)
    test[REGION_COL] = test[REGION_COL].astype(str)
    sample = pd.read_csv(DATA/"sample_submission.csv", usecols=[REGION_COL])
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    region_order = sample[REGION_COL].tolist()
    region_to_idx = {r:i for i,r in enumerate(region_order)}

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
            sv = ds[w*7:(w+1)*7]
            sv = sv[np.isfinite(sv)]
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

    ds = SummaryDataset(region_arrays, args.context_weeks, args.horizons, region_to_idx, w_idx, use_score)
    print(f"  {len(ds):,} samples, feature dim = {ds[0]['feats'].shape[0]}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SummaryMLP(ds[0]["feats"].shape[0], len(region_order), hidden=args.hidden, n_layers=args.n_layers, horizons=args.horizons).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_steps)
    scaler = torch.amp.GradScaler("cuda") if device.type=="cuda" else None

    step = 0
    t0 = time.time()
    while step < args.num_steps:
        for batch in loader:
            if step >= args.num_steps: break
            f = batch["feats"].to(device); t = batch["tgt"].to(device); m = batch["mask"].to(device); ri = batch["region_idx"].to(device)
            opt.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=(device.type=="cuda")):
                pred = model(f, ri)
                loss = ((pred - t).abs() * m).sum() / m.sum().clamp_min(1.0)
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
            if step % 500 == 0:
                print(f"  step {step}/{args.num_steps}: loss={float(loss):.4f}")
            step += 1
    print(f"  train: {time.time()-t0:.0f}s")

    model.eval()
    out_rows = []
    with torch.no_grad():
        for bs in range(0, len(region_order), 128):
            br = region_order[bs:bs+128]
            valid = [r for r in br if r in region_arrays]
            if not valid:
                for r in br: out_rows.append([r]+[0.5]*args.horizons)
                continue
            fl = []
            for r in valid:
                tw = region_arrays[r]["weather"]; ts = region_arrays[r]["score"]
                test_w = test_weekly.get(r)
                if test_w is not None:
                    full_w = np.concatenate([tw, test_w], axis=0)
                    last_s = ts[~np.isnan(ts)][-1] if np.isfinite(ts).any() else 0.0
                    full_s = np.concatenate([ts, np.full(test_w.shape[0], last_s, dtype=np.float32)])
                else:
                    full_w = tw; full_s = ts
                ctx_w = full_w[-args.context_weeks:][:, w_idx]
                cs = fill_score_nan(full_s[-args.context_weeks:])
                if ctx_w.shape[0] < args.context_weeks:
                    pad = args.context_weeks - ctx_w.shape[0]
                    ctx_w = np.pad(ctx_w, ((pad,0),(0,0)), mode="edge")
                    cs = np.pad(cs, (pad,0), mode="edge")
                fl.append(summary_features(ctx_w, cs, use_score=use_score))
            f = torch.from_numpy(np.stack(fl)).float().to(device)
            ri = torch.tensor([region_to_idx[r] for r in valid], dtype=torch.long, device=device)
            pred = model(f, ri).cpu().numpy().clip(0, 5)
            for i, r in enumerate(valid): out_rows.append([r]+pred[i].tolist())
            for r in br:
                if r not in valid: out_rows.append([r]+[0.5]*args.horizons)
    out_df = pd.DataFrame(out_rows, columns=[REGION_COL]+PRED_COLS[:args.horizons])
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
