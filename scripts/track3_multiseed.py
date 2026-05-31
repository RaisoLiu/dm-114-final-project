#!/usr/bin/env python3
"""Train Track 3 with N different seeds, ensemble averaged."""
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
PRED_COLS = [f'pred_week{i+1}' for i in range(5)]


def build_train_windows(train_df, max_anchors=300):
    train_df = train_df.sort_values(['region_id', 'date']).reset_index(drop=True)
    regions = train_df['region_id'].unique().tolist()
    n_per = train_df.groupby('region_id').size().iloc[0]
    w_all = train_df[WEATHER_COLS].values.astype(np.float32).reshape(len(regions), n_per, N_CHANNELS)
    s_all = train_df['score'].values.astype(np.float32).reshape(len(regions), n_per)
    windows = []; targets = []; ridx = []
    for r in range(len(regions)):
        s = s_all[r]; w = w_all[r]
        valid = np.where(~np.isnan(s))[0]
        valid = valid[(valid >= N_DAYS - 1) & (valid <= n_per - 1 - 35)]
        if max_anchors and len(valid) > max_anchors:
            valid = valid[::len(valid)//max_anchors][:max_anchors]
        for t in valid:
            future = s[t + 7 * np.arange(1, 6)]
            if np.isnan(future).any(): continue
            win = w[t-N_DAYS+1:t+1, :]
            if np.isnan(win).any(): continue
            windows.append(win.T)
            targets.append(future)
            ridx.append(r)
    return np.stack(windows).astype(np.float32), np.stack(targets).astype(np.float32), np.array(ridx), regions, w_all, s_all


def build_test(test_df, regions):
    test_df = test_df.sort_values(['region_id', 'date']).reset_index(drop=True)
    nt = test_df.groupby('region_id').size().iloc[0]
    w = test_df[WEATHER_COLS].values.astype(np.float32).reshape(len(regions), nt, N_CHANNELS)
    for r in range(len(regions)):
        for c in range(N_CHANNELS):
            col = w[r, :, c]
            if np.isnan(col).any():
                col[np.isnan(col)] = np.nanmean(col) if not np.isnan(np.nanmean(col)) else 0.0
    return w.transpose(0, 2, 1)


class CNNHead(nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        self.c1 = nn.Conv1d(N_CHANNELS, hidden, 7, padding=3, dilation=1)
        self.g1 = nn.GroupNorm(8, hidden)
        self.c2 = nn.Conv1d(hidden, hidden, 7, padding=6, dilation=2)
        self.g2 = nn.GroupNorm(8, hidden)
        self.c3 = nn.Conv1d(hidden, hidden, 7, padding=12, dilation=4)
        self.g3 = nn.GroupNorm(8, hidden)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        h = F.gelu(self.g1(self.c1(x)))
        h = F.gelu(self.g2(self.c2(h)))
        h = F.gelu(self.g3(self.c3(h)))
        return self.head(h)


def train_one(seed, X_std, y, ridx, X_te_std, epochs=8, hidden=96):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CNNHead(hidden=hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    idx = np.arange(len(X_std)); np.random.shuffle(idx)
    split = int(0.9*len(idx)); tr=idx[:split]; va=idx[split:]
    best_state = None; best_mae = 1e9
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        total = 0; n = 0
        for i in range(0, len(perm), 256):
            ix = tr[perm[i:i+256]]
            xb = torch.from_numpy(X_std[ix]).to(DEVICE)
            yb = torch.from_numpy(y[ix]).to(DEVICE)
            opt.zero_grad()
            loss = F.l1_loss(model(xb), yb)
            loss.backward(); opt.step()
            total += loss.item()*len(xb); n += len(xb)
        model.eval()
        with torch.no_grad():
            vp = []; vt = []
            for i in range(0, len(va), 256):
                ix = va[i:i+256]
                xb = torch.from_numpy(X_std[ix]).to(DEVICE)
                vp.append(model(xb).cpu().numpy()); vt.append(y[ix])
            vp = np.concatenate(vp); vt = np.concatenate(vt)
            v_mae = float(np.abs(vp - vt).mean())
        if v_mae < best_mae:
            best_mae = v_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te_std), 128):
            xb = torch.from_numpy(X_te_std[i:i+128]).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds), best_mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs='+', default=[42, 1234, 7777, 31337])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--max-anchors", type=int, default=300)
    ap.add_argument("--output", default="submissions/_v18_track3_multiseed.csv")
    args = ap.parse_args()

    print("Loading...")
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    test_df = pd.read_csv(ROOT / "data" / "test.csv")
    X, y, ridx, regions, w_all, s_all = build_train_windows(train_df, max_anchors=args.max_anchors)
    X_te = build_test(test_df, regions)
    print(f"Train {len(X)} test {len(regions)}")

    mean = X.mean(axis=(0, 2)).reshape(1, N_CHANNELS, 1)
    std = X.std(axis=(0, 2)).reshape(1, N_CHANNELS, 1) + 1e-6
    X_std = (X - mean) / std
    X_te_std = (X_te - mean) / std

    all_preds = []
    for seed in args.seeds:
        t0 = time.time()
        print(f"\nSeed {seed}:")
        preds, val_mae = train_one(seed, X_std, y, ridx, X_te_std,
                                     epochs=args.epochs, hidden=args.hidden)
        all_preds.append(preds)
        print(f"  val_mae={val_mae:.4f}  pred_mean={preds.mean():.3f}  std={preds.std():.3f}  elapsed={time.time()-t0:.0f}s")

    # Average ensemble
    avg = np.mean(all_preds, axis=0)
    avg = np.clip(avg, 0, 5)
    out = pd.DataFrame({'region_id': regions})
    for i, c in enumerate(PRED_COLS):
        out[c] = avg[:, i]
    out_path = ROOT / args.output
    out.to_csv(out_path, index=False)
    print(f"\nMultiseed ensemble saved: {out_path}")
    print(f"  mean={avg.mean():.4f}  std={avg.std():.4f}")


if __name__ == "__main__":
    main()
