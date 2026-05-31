#!/usr/bin/env python3
"""Track 3 with MSE loss (instead of L1) — biases toward MEAN not MEDIAN.
May give different errors → orthogonal candidate."""
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


def build_train(train_df, max_anchors=300):
    train_df = train_df.sort_values(['region_id', 'date']).reset_index(drop=True)
    regions = train_df['region_id'].unique().tolist()
    n_per = train_df.groupby('region_id').size().iloc[0]
    w = train_df[WEATHER_COLS].values.astype(np.float32).reshape(len(regions), n_per, N_CHANNELS)
    s = train_df['score'].values.astype(np.float32).reshape(len(regions), n_per)
    windows = []; targets = []; ridx = []
    for r in range(len(regions)):
        valid = np.where(~np.isnan(s[r]))[0]
        valid = valid[(valid >= N_DAYS - 1) & (valid <= n_per - 1 - 35)]
        if max_anchors and len(valid) > max_anchors:
            valid = valid[::len(valid)//max_anchors][:max_anchors]
        for t in valid:
            future = s[r, t + 7 * np.arange(1, 6)]
            if np.isnan(future).any(): continue
            win = w[r, t-N_DAYS+1:t+1, :]
            if np.isnan(win).any(): continue
            windows.append(win.T); targets.append(future); ridx.append(r)
    return np.stack(windows).astype(np.float32), np.stack(targets).astype(np.float32), np.array(ridx), regions


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


class CNNMSE(nn.Module):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-anchors", type=int, default=300)
    ap.add_argument("--loss", choices=['mse', 'huber', 'wmse'], default='mse')
    ap.add_argument("--output", default="submissions/_v18_track3_mse.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"Loss: {args.loss}")
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    test_df = pd.read_csv(ROOT / "data" / "test.csv")
    X, y, ridx, regions = build_train(train_df, max_anchors=args.max_anchors)
    X_te = build_test(test_df, regions)

    mean = X.mean(axis=(0, 2)).reshape(1, N_CHANNELS, 1)
    std = X.std(axis=(0, 2)).reshape(1, N_CHANNELS, 1) + 1e-6
    X_std = (X - mean) / std
    X_te_std = (X_te - mean) / std

    idx = np.arange(len(X_std)); np.random.shuffle(idx)
    split = int(0.9*len(idx)); tr=idx[:split]; va=idx[split:]
    model = CNNMSE(hidden=args.hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best_state = None; best_mae = 1e9
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        perm = np.random.permutation(len(tr))
        total = 0; n = 0
        for i in range(0, len(perm), 256):
            ix = tr[perm[i:i+256]]
            xb = torch.from_numpy(X_std[ix]).to(DEVICE)
            yb = torch.from_numpy(y[ix]).to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            if args.loss == 'mse':
                loss = F.mse_loss(pred, yb)
            elif args.loss == 'huber':
                loss = F.smooth_l1_loss(pred, yb, beta=1.0)
            elif args.loss == 'wmse':
                # Weighted: weight = (1 + score) to emphasize high-score
                w = 1.0 + yb
                loss = ((pred - yb) ** 2 * w).mean()
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
        print(f"  ep {ep+1}/{args.epochs}  train_loss={total/n:.4f}  val_mae={v_mae:.4f}  best={best_mae:.4f}  elapsed={time.time()-t0:.0f}s")
    model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te_std), 128):
            xb = torch.from_numpy(X_te_std[i:i+128]).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    preds = np.clip(np.concatenate(preds), 0, 5)
    out = pd.DataFrame({'region_id': regions})
    for i, c in enumerate(PRED_COLS):
        out[c] = preds[:, i]
    out.to_csv(ROOT / args.output, index=False)
    print(f"Saved: {ROOT / args.output}")
    print(f"  mean={preds.mean():.4f}  std={preds.std():.4f}")


if __name__ == "__main__":
    main()
