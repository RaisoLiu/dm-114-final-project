#!/usr/bin/env python3
"""Track 3 v2 — CNN + Region Embedding.

Adds a 2,248 × 16-dim region embedding to the CNN encoder so the model
captures per-region drought patterns that pure weather can't.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEATHER_COLS = ['prec', 'surf_pre', 'humidity', 'tmp', 'dp_tmp', 'wb_tmp',
                'tmp_max', 'tmp_min', 'tmp_range', 'surf_tmp',
                'wind', 'wind_max', 'wind_min', 'wind_range']
N_CHANNELS = 14
N_DAYS = 91
N_HORIZONS = 5
PRED_COLS = [f'pred_week{i+1}' for i in range(N_HORIZONS)]


def build_windows(train_df, max_anchors=300):
    train_df = train_df.sort_values(['region_id', 'date']).reset_index(drop=True)
    regions = train_df['region_id'].unique().tolist()
    n_per_region = train_df.groupby('region_id').size().iloc[0]
    w_all = train_df[WEATHER_COLS].values.astype(np.float32).reshape(len(regions), n_per_region, N_CHANNELS)
    s_all = train_df['score'].values.astype(np.float32).reshape(len(regions), n_per_region)
    windows = []; targets = []; region_idx = []
    for r_i in range(len(regions)):
        s = s_all[r_i]
        valid = np.where(~np.isnan(s))[0]
        valid = valid[(valid >= N_DAYS - 1) & (valid <= n_per_region - 1 - 35)]
        if max_anchors and len(valid) > max_anchors:
            valid = valid[::len(valid)//max_anchors][:max_anchors]
        for t in valid:
            future = s[t + 7 * np.arange(1, 6)]
            if np.isnan(future).any(): continue
            win = w_all[r_i, t-N_DAYS+1:t+1, :]
            if np.isnan(win).any(): continue
            windows.append(win.T.astype(np.float32))
            targets.append(future.astype(np.float32))
            region_idx.append(r_i)
    return np.stack(windows), np.stack(targets), np.array(region_idx), regions, w_all, s_all


def build_test_windows(test_df, regions):
    test_df = test_df.sort_values(['region_id', 'date']).reset_index(drop=True)
    n_test_per = test_df.groupby('region_id').size().iloc[0]
    w_test = test_df[WEATHER_COLS].values.astype(np.float32).reshape(len(regions), n_test_per, N_CHANNELS)
    for r in range(len(regions)):
        for c in range(N_CHANNELS):
            col = w_test[r, :, c]
            if np.isnan(col).any():
                col[np.isnan(col)] = np.nanmean(col) if not np.isnan(np.nanmean(col)) else 0.0
    return w_test.transpose(0, 2, 1), regions


class CNNRegEmb(nn.Module):
    def __init__(self, n_regions, hidden=96, reg_emb_dim=16):
        super().__init__()
        self.reg_emb = nn.Embedding(n_regions, reg_emb_dim)
        self.conv1 = nn.Conv1d(N_CHANNELS, hidden, 7, padding=3, dilation=1)
        self.gn1 = nn.GroupNorm(8, hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, 7, padding=6, dilation=2)
        self.gn2 = nn.GroupNorm(8, hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, 7, padding=12, dilation=4)
        self.gn3 = nn.GroupNorm(8, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + reg_emb_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, N_HORIZONS),
        )

    def forward(self, x, region_idx):
        # x: (B, 14, 91), region_idx: (B,)
        h = F.gelu(self.gn1(self.conv1(x)))
        h = F.gelu(self.gn2(self.conv2(h)))
        h = F.gelu(self.gn3(self.conv3(h)))
        h = h.mean(dim=2)  # GAP
        re = self.reg_emb(region_idx)
        z = torch.cat([h, re], dim=1)
        return self.head(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-anchors", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--output", default="submissions/_v18_track3_v2_regemb.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print("Loading data...")
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    test_df = pd.read_csv(ROOT / "data" / "test.csv")
    X, y, ridx, regions, w_all, s_all = build_windows(train_df, max_anchors=args.max_anchors)
    X_te, _ = build_test_windows(test_df, regions)
    print(f"Train anchors: {len(X)}  test regions: {len(regions)}")

    # Standardize
    mean_arr = X.mean(axis=(0, 2))
    std_arr = X.std(axis=(0, 2))
    X_std = (X - mean_arr.reshape(1, N_CHANNELS, 1)) / (std_arr.reshape(1, N_CHANNELS, 1) + 1e-6)
    X_te_std = (X_te - mean_arr.reshape(1, N_CHANNELS, 1)) / (std_arr.reshape(1, N_CHANNELS, 1) + 1e-6)

    # Random 10% holdout
    idx = np.arange(len(X_std)); np.random.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx = idx[:split]; va_idx = idx[split:]
    print(f"Train {len(tr_idx)} val {len(va_idx)}")

    model = CNNRegEmb(len(regions), hidden=args.hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    best_state = None; best_mae = float('inf')
    batch_size = 256
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(tr_idx))
        total = 0; n = 0
        t0 = time.time()
        for i in range(0, len(perm), batch_size):
            ix = tr_idx[perm[i:i+batch_size]]
            xb = torch.from_numpy(X_std[ix]).to(DEVICE)
            rb = torch.from_numpy(ridx[ix]).to(DEVICE)
            yb = torch.from_numpy(y[ix]).to(DEVICE)
            opt.zero_grad()
            pred = model(xb, rb)
            loss = F.l1_loss(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb); n += len(xb)
        model.eval()
        with torch.no_grad():
            preds_v = []; targs_v = []
            for i in range(0, len(va_idx), batch_size):
                ix = va_idx[i:i+batch_size]
                xb = torch.from_numpy(X_std[ix]).to(DEVICE)
                rb = torch.from_numpy(ridx[ix]).to(DEVICE)
                preds_v.append(model(xb, rb).cpu().numpy())
                targs_v.append(y[ix])
            preds_v = np.concatenate(preds_v, axis=0)
            targs_v = np.concatenate(targs_v, axis=0)
            v_mae = float(np.abs(preds_v - targs_v).mean())
        if v_mae < best_mae:
            best_mae = v_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ep {ep+1}/{args.epochs}  train_loss={total/n:.4f}  val_mae={v_mae:.4f}  best={best_mae:.4f}  elapsed={time.time()-t0:.0f}s")
    model.load_state_dict(best_state)

    # Predict test
    print("Test inference...")
    model.eval()
    preds = []
    with torch.no_grad():
        for r in range(len(regions)):
            xb = torch.from_numpy(X_te_std[r:r+1]).to(DEVICE)
            rb = torch.tensor([r], dtype=torch.long, device=DEVICE)
            p = model(xb, rb).cpu().numpy()[0]
            preds.append(p)
    preds = np.clip(np.stack(preds), 0, 5)
    out_df = pd.DataFrame({'region_id': regions})
    for i, c in enumerate(PRED_COLS):
        out_df[c] = preds[:, i]
    out_path = ROOT / args.output
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"  mean={preds.mean():.4f} std={preds.std():.4f}")


if __name__ == "__main__":
    main()
