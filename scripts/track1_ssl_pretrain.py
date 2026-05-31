#!/usr/bin/env python3
"""Track 1 — SSL Weather Pretraining (Time-MAE style).

Pretrain a Transformer encoder on 91-day weather windows using masked
reconstruction. Then fine-tune a small score head on the 22 synth labeled anchors.

Architecture:
  Input: (B, 14, 91) -> linear projection (14 -> d_model) -> transpose to (B, 91, d_model)
  + positional encoding -> 8-layer Transformer encoder
  Heads:
    - Reconstruction: (B, 91, d_model) -> Linear(d_model, 14) -> reconstruct masked cells
    - Score: mean-pool 91 time steps -> MLP(d_model, 128, 5)

Two-stage training:
  Stage A (pretrain): masked reconstruction on ALL 12.3M daily-window slices
                       (each region's 5,480-day series, sliding 91-day windows stride=7)
                       mask 30% of cells, predict masked
  Stage B (finetune): supervised score regression on labeled anchors
                       (frozen backbone + new score head, or LoRA)

Estimated time: pretrain ~3-5h on 4090-class GPU; finetune ~30 min; total <5h.
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
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEATHER_COLS = ['prec', 'surf_pre', 'humidity', 'tmp', 'dp_tmp', 'wb_tmp',
                'tmp_max', 'tmp_min', 'tmp_range', 'surf_tmp',
                'wind', 'wind_max', 'wind_min', 'wind_range']
N_CHANNELS = 14
N_DAYS = 91
PRED_COLS = [f'pred_week{i+1}' for i in range(5)]


class WeatherWindowDataset(Dataset):
    """Sliding windows over all 2,248 regions' daily weather."""
    def __init__(self, w_all: np.ndarray, stride: int = 7, stats=None):
        # w_all: (R, T, C)
        self.w_all = w_all
        self.R, self.T, self.C = w_all.shape
        self.stride = stride
        # Pre-compute (region, t) anchor list
        anchors = []
        for r in range(self.R):
            for t in range(N_DAYS - 1, self.T, stride):
                anchors.append((r, t))
        self.anchors = anchors
        if stats is None:
            stats = {
                'mean': w_all.reshape(-1, self.C).mean(axis=0),
                'std': w_all.reshape(-1, self.C).std(axis=0),
            }
        self.stats = stats

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, i):
        r, t = self.anchors[i]
        w = self.w_all[r, t - N_DAYS + 1: t + 1, :]  # (91, 14)
        w_std = (w - self.stats['mean']) / (self.stats['std'] + 1e-6)
        return torch.from_numpy(w_std.T.astype(np.float32))  # (14, 91)


class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, L, D)
        return x + self.pe[:, :x.size(1), :]


class SSLTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.proj = nn.Linear(N_CHANNELS, d_model)
        self.pos = PosEnc(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                            dim_feedforward=d_model * 4,
                                            dropout=0.1, batch_first=True,
                                            activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.recon_head = nn.Linear(d_model, N_CHANNELS)
        self.score_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5),
        )
        self.d_model = d_model

    def encode(self, x):
        # x: (B, 14, 91) -> (B, 91, 14) -> (B, 91, d_model)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        return x  # (B, 91, d_model)

    def recon(self, x):
        z = self.encode(x)  # (B, 91, d_model)
        return self.recon_head(z).transpose(1, 2)  # (B, 14, 91)

    def predict_score(self, x):
        z = self.encode(x).transpose(1, 2)  # (B, d_model, 91)
        return self.score_head(z)  # (B, 5)


def masked_loss(model, x, mask_ratio=0.30):
    """Apply random (channel, time) cell mask and compute L1 loss on masked cells."""
    B, C, T = x.shape
    mask = torch.rand(B, C, T, device=x.device) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0.0
    pred = model.recon(x_masked)  # (B, 14, 91)
    return F.l1_loss(pred[mask], x[mask])


def pretrain(model, w_all: np.ndarray, epochs=10, batch_size=512, lr=3e-4, stride=7):
    """Stage A: masked reconstruction pretrain."""
    ds = WeatherWindowDataset(w_all, stride=stride)
    print(f"  pretrain dataset: {len(ds)} windows")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        total = 0; n = 0; t0 = time.time()
        for xb in dl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            loss = masked_loss(model, xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb); n += len(xb)
        elapsed = time.time() - t0
        print(f"  pretrain ep {ep+1}/{epochs}  loss={total/n:.4f}  elapsed={elapsed:.0f}s")
    return ds.stats


def fit_score_head(model, X_tr, y_tr, region_idx, stats, epochs=15, batch_size=256, lr=3e-4):
    """Stage B: finetune score head on labeled anchors."""
    print(f"  fit score head on {len(X_tr)} anchors, val 10% random split")
    np.random.seed(0)
    idx = np.arange(len(X_tr))
    np.random.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx = idx[:split]; va_idx = idx[split:]
    X_tr_t = torch.from_numpy(X_tr[tr_idx]).float()
    y_tr_t = torch.from_numpy(y_tr[tr_idx]).float()
    X_va_t = torch.from_numpy(X_tr[va_idx]).float()
    y_va_t = torch.from_numpy(y_tr[va_idx]).float()

    # Standardize
    mean_t = torch.from_numpy(stats['mean']).float().view(1, N_CHANNELS, 1)
    std_t = torch.from_numpy(stats['std']).float().view(1, N_CHANNELS, 1) + 1e-6
    X_tr_t = (X_tr_t - mean_t) / std_t
    X_va_t = (X_va_t - mean_t) / std_t

    # Allow whole model fine-tuning
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None; best_mae = float('inf')

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        total = 0; n = 0
        for i in range(0, len(perm), batch_size):
            ix = perm[i:i+batch_size]
            xb = X_tr_t[ix].to(DEVICE); yb = y_tr_t[ix].to(DEVICE)
            opt.zero_grad()
            pred = model.predict_score(xb)
            # Multi-task: keep aux loss alive (small weight)
            aux_loss = masked_loss(model, xb, mask_ratio=0.15)
            loss = F.l1_loss(pred, yb) + 0.1 * aux_loss
            loss.backward()
            opt.step()
            total += loss.item() * len(xb); n += len(xb)
        # val
        model.eval()
        with torch.no_grad():
            v_preds = []
            for i in range(0, len(X_va_t), batch_size):
                xb = X_va_t[i:i+batch_size].to(DEVICE)
                v_preds.append(model.predict_score(xb).cpu().numpy())
            v_preds = np.concatenate(v_preds, axis=0)
            v_mae = float(np.abs(v_preds - y_va_t.numpy()).mean())
        if v_mae < best_mae:
            best_mae = v_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"    finetune ep {ep+1}/{epochs}  train_loss={total/n:.4f}  val_mae={v_mae:.4f}  best={best_mae:.4f}")
    model.load_state_dict(best_state)
    return best_mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--pretrain-epochs", type=int, default=8)
    ap.add_argument("--finetune-epochs", type=int, default=15)
    ap.add_argument("--max-anchors", type=int, default=400)
    ap.add_argument("--pretrain-stride", type=int, default=14)
    ap.add_argument("--output", default="submissions/_v18_track1_ssl.csv")
    args = ap.parse_args()

    # Load data
    print("Loading train.csv...")
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    test_df = pd.read_csv(ROOT / "data" / "test.csv")

    # Reshape
    train_df = train_df.sort_values(['region_id', 'date']).reset_index(drop=True)
    regions = train_df['region_id'].unique().tolist()
    n_per_region = train_df.groupby('region_id').size().iloc[0]
    w_all = train_df[WEATHER_COLS].values.astype(np.float32).reshape(len(regions), n_per_region, N_CHANNELS)
    s_all = train_df['score'].values.astype(np.float32).reshape(len(regions), n_per_region)
    print(f"  train: {len(regions)} regions × {n_per_region} days × {N_CHANNELS} channels")

    # Build labeled anchors
    print("Building labeled anchors...")
    windows = []
    targets = []
    region_idx = []
    for r_i in range(len(regions)):
        s = s_all[r_i]
        valid_anchors = np.where(~np.isnan(s))[0]
        valid_anchors = valid_anchors[(valid_anchors >= N_DAYS - 1) & (valid_anchors <= n_per_region - 1 - 35)]
        if args.max_anchors and len(valid_anchors) > args.max_anchors:
            step = len(valid_anchors) // args.max_anchors
            valid_anchors = valid_anchors[::step][:args.max_anchors]
        for t in valid_anchors:
            future = s[t + 7 * np.arange(1, 6)]
            if np.isnan(future).any():
                continue
            win = w_all[r_i, t - N_DAYS + 1: t + 1, :]
            if np.isnan(win).any():
                continue
            windows.append(win.T)
            targets.append(future)
            region_idx.append(r_i)
    X_tr = np.stack(windows).astype(np.float32)
    y_tr = np.stack(targets).astype(np.float32)
    region_idx = np.array(region_idx, dtype=np.int64)
    print(f"  labeled anchors: {len(X_tr)}")

    # Init model
    print(f"\nInit SSL Transformer: d_model={args.d_model} layers={args.num_layers}")
    model = SSLTransformer(d_model=args.d_model, num_layers=args.num_layers).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.2f}M")

    # Stage A: pretrain
    print(f"\n=== Stage A: masked reconstruction pretrain ===")
    stats = pretrain(model, w_all, epochs=args.pretrain_epochs, stride=args.pretrain_stride)
    ck = ROOT / "checkpoints" / "track1_pretrain.pt"
    ck.parent.mkdir(exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'stats': stats, 'args': vars(args)}, ck)
    print(f"  pretrain checkpoint saved: {ck}")

    # Stage B: finetune
    print(f"\n=== Stage B: finetune score head ===")
    val_mae = fit_score_head(model, X_tr, y_tr, region_idx, stats,
                              epochs=args.finetune_epochs)
    ck2 = ROOT / "checkpoints" / "track1_finetuned.pt"
    torch.save({'state_dict': model.state_dict(), 'stats': stats, 'val_mae': val_mae}, ck2)

    # Test inference
    print(f"\n=== Test inference ===")
    test_df = test_df.sort_values(['region_id', 'date']).reset_index(drop=True)
    test_regions = test_df['region_id'].unique().tolist()
    test_days_per_region = test_df.groupby('region_id').size().iloc[0]
    w_test = test_df[WEATHER_COLS].values.astype(np.float32).reshape(len(test_regions), test_days_per_region, N_CHANNELS)
    # NaN imputation
    for r in range(len(test_regions)):
        for c in range(N_CHANNELS):
            col = w_test[r, :, c]
            if np.isnan(col).any():
                mean = np.nanmean(col)
                col[np.isnan(col)] = mean if not np.isnan(mean) else 0.0
                w_test[r, :, c] = col
    X_te = w_test.transpose(0, 2, 1)  # (2248, 14, 91)
    mean_arr = stats['mean'].reshape(1, N_CHANNELS, 1)
    std_arr = stats['std'].reshape(1, N_CHANNELS, 1)
    X_te_std = (X_te - mean_arr) / (std_arr + 1e-6)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te_std), 128):
            xb = torch.from_numpy(X_te_std[i:i+128]).float().to(DEVICE)
            preds.append(model.predict_score(xb).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    preds = np.clip(preds, 0, 5)
    out_df = pd.DataFrame({'region_id': test_regions})
    for i, col in enumerate(PRED_COLS):
        out_df[col] = preds[:, i]
    out_path = ROOT / args.output
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  preds mean={preds.mean():.4f} std={preds.std():.4f}")


if __name__ == "__main__":
    main()
