#!/usr/bin/env python3
"""Track 3 — Test-Time Adaptation (TTT) on 2,248 zero-shot test regions.

Architecture:
  91-day weather window (14 channels) -> CNN encoder -> dual heads:
    - main: 5-horizon score prediction (L1 loss on synth train)
    - aux:  masked weather reconstruction (self-supervised, no labels)

Training:
  22 synth regions × 782 weekly anchors -> 17,204 samples
  multi-task loss = score_loss + lambda * aux_loss
  LORO validation

TTT inference:
  For each test region: fine-tune aux head + last 2 encoder layers
  on aux loss using that region's 91-day window only (50 steps).
  Then predict 5 horizons with adapted model.

Anti-v17-leak rules:
  - No region_id as feature
  - No date/year/DOY as feature
  - No score lookup of any kind
  - Input: only the 91-day weather window (14 × 91 tensor) + soil-free
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
N_HORIZONS = 5
PRED_COLS = [f'pred_week{i+1}' for i in range(N_HORIZONS)]


def standardize_features(arr: np.ndarray, stats: dict) -> np.ndarray:
    """Standardize using per-channel mean/std stats."""
    mean = stats['mean'].reshape(1, N_CHANNELS, 1)
    std = stats['std'].reshape(1, N_CHANNELS, 1)
    return (arr - mean) / (std + 1e-6)


def build_train_windows(train_df: pd.DataFrame, max_anchors_per_region: int | None = None):
    """Vectorized window builder: O(N) total via groupby."""
    print(f"Building train windows from {len(train_df)} rows...")
    train_df = train_df.copy()
    # Use ordinal day = row order within each region (pre-sorted by date)
    train_df = train_df.sort_values(['region_id', 'date']).reset_index(drop=True)
    regions = train_df['region_id'].unique().tolist()
    print(f"  {len(regions)} regions")
    n_per_region = train_df.groupby('region_id').size()
    assert n_per_region.nunique() == 1, "expected equal rows per region"
    days_per_region = int(n_per_region.iloc[0])
    print(f"  days per region: {days_per_region}")

    # Reshape into (region, days, channel)
    w_all = train_df[WEATHER_COLS].values.astype(np.float32)
    s_all = train_df['score'].values.astype(np.float32)
    w_all = w_all.reshape(len(regions), days_per_region, N_CHANNELS)  # (R, T, 14)
    s_all = s_all.reshape(len(regions), days_per_region)  # (R, T)

    # Compute anchor positions: any t where s[r,t] is not NaN and t in [N_DAYS-1, T - 5*7 -1]
    # Also require future +7,+14,..+35 to be not-NaN
    windows_list = []
    targets_list = []
    region_idx_list = []
    for r_i in range(len(regions)):
        s = s_all[r_i]
        anchor_mask = ~np.isnan(s)
        # Valid t range
        valid = np.where(anchor_mask)[0]
        valid = valid[(valid >= N_DAYS - 1) & (valid <= days_per_region - 1 - 5*7)]
        if max_anchors_per_region:
            # subsample to keep training size manageable
            if len(valid) > max_anchors_per_region:
                step = len(valid) // max_anchors_per_region
                valid = valid[::step][:max_anchors_per_region]
        # Check future labels not NaN
        for t in valid:
            future = s[t + 7 * np.arange(1, 6)]
            if np.isnan(future).any():
                continue
            win = w_all[r_i, t-N_DAYS+1:t+1, :]  # (91, 14)
            if np.isnan(win).any():
                continue
            windows_list.append(win.T.astype(np.float32))  # (14, 91)
            targets_list.append(future.astype(np.float32))
            region_idx_list.append(r_i)
    print(f"  built {len(windows_list)} anchor windows")
    X = np.stack(windows_list)  # (N, 14, 91)
    y = np.stack(targets_list)  # (N, 5)
    region_idx = np.array(region_idx_list, dtype=np.int64)
    return X, y, region_idx, regions


def build_test_windows(test_df: pd.DataFrame):
    """Vectorized test window builder."""
    print(f"Building test windows from {len(test_df)} rows...")
    test_df = test_df.sort_values(['region_id', 'date']).reset_index(drop=True)
    regions = test_df['region_id'].unique().tolist()
    print(f"  {len(regions)} test regions")
    w_all = test_df[WEATHER_COLS].values.astype(np.float32)
    w_all = w_all.reshape(len(regions), N_DAYS, N_CHANNELS)  # (R, 91, 14)
    # Impute NaN channel by channel
    for r in range(len(regions)):
        for c in range(N_CHANNELS):
            col = w_all[r, :, c]
            if np.isnan(col).any():
                mean = np.nanmean(col)
                col[np.isnan(col)] = mean if not np.isnan(mean) else 0.0
                w_all[r, :, c] = col
    # Transpose to (R, 14, 91)
    X = w_all.transpose(0, 2, 1)
    return X, regions


class CNNEncoder(nn.Module):
    """1-D CNN that produces a sequence representation."""
    def __init__(self, in_channels=N_CHANNELS, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3, dilation=1)
        self.gn1 = nn.GroupNorm(8, hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=7, padding=6, dilation=2)
        self.gn2 = nn.GroupNorm(8, hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=7, padding=12, dilation=4)
        self.gn3 = nn.GroupNorm(8, hidden)
        self.hidden = hidden

    def forward(self, x):
        # x: (B, 14, 91)
        h = F.gelu(self.gn1(self.conv1(x)))
        h = F.gelu(self.gn2(self.conv2(h)))
        h = F.gelu(self.gn3(self.conv3(h)))
        return h  # (B, hidden, 91)


class DualHeadModel(nn.Module):
    """Encoder + dual heads (score regression + weather reconstruction)."""
    def __init__(self, hidden=64):
        super().__init__()
        self.encoder = CNNEncoder(hidden=hidden)
        # Score head: pool encoder output, MLP -> 5 horizons
        self.score_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, N_HORIZONS),
        )
        # Aux head: reconstruct masked weather cells
        self.aux_head = nn.Conv1d(hidden, N_CHANNELS, kernel_size=1)

    def forward(self, x, return_aux=False):
        h = self.encoder(x)
        score = self.score_head(h)
        if return_aux:
            aux = self.aux_head(h)
            return score, aux
        return score


def masked_aux_loss(model, x, mask_ratio=0.15):
    """Self-supervised masked weather reconstruction loss."""
    # Random mask over (channel, day) cells
    B, C, T = x.shape
    mask = torch.rand(B, C, T, device=x.device) < mask_ratio  # True = masked
    x_masked = x.clone()
    # Set masked cells to 0 (after standardization, this is the channel mean)
    x_masked[mask] = 0.0
    _, aux = model(x_masked, return_aux=True)
    # L1 loss on masked positions only
    loss = F.l1_loss(aux[mask], x[mask])
    return loss


class WindowDataset(Dataset):
    def __init__(self, X, y, region_idx=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None
        self.region_idx = torch.from_numpy(region_idx).long() if region_idx is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.y is not None:
            return self.X[i], self.y[i]
        return self.X[i]


def train_base_model(X, y, region_idx, val_region_idx=None, epochs=20, batch_size=256,
                     lr=3e-4, aux_lambda=0.3, hidden=64, verbose=True):
    """Train dual-head model. If val_region_idx is set, hold out that region."""
    if val_region_idx is not None:
        train_mask = region_idx != val_region_idx
        val_mask = region_idx == val_region_idx
    else:
        # All training
        n = len(X)
        idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(idx)
        split = int(0.9 * n)
        train_mask = np.zeros(n, dtype=bool); train_mask[idx[:split]] = True
        val_mask = ~train_mask

    X_tr = X[train_mask]; y_tr = y[train_mask]
    X_va = X[val_mask]; y_va = y[val_mask]
    if verbose:
        print(f"  train {len(X_tr)} val {len(X_va)} samples")

    model = DualHeadModel(hidden=hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    ds_tr = WindowDataset(X_tr, y_tr)
    ds_va = WindowDataset(X_va, y_va)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    best_val_mae = float('inf')
    best_state = None
    for ep in range(epochs):
        model.train()
        train_loss = 0.0; n = 0
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            score_pred, _ = model(xb, return_aux=True)
            score_loss = F.l1_loss(score_pred, yb)
            aux_loss = masked_aux_loss(model, xb)
            loss = score_loss + aux_lambda * aux_loss
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb); n += len(xb)
        train_loss /= n
        # Val
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_targets = []
            for xb, yb in dl_va:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                p = model(xb)
                val_preds.append(p.cpu().numpy())
                val_targets.append(yb.cpu().numpy())
            val_preds = np.concatenate(val_preds, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)
            val_mae = float(np.abs(val_preds - val_targets).mean())
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  ep {ep+1:>2d}  train_loss={train_loss:.4f}  val_mae={val_mae:.4f}  best={best_val_mae:.4f}")
    model.load_state_dict(best_state)
    return model, best_val_mae


def loro_validate(X, y, region_idx, regions, epochs=15, hidden=64):
    """Leave-one-region-out validation across 22 regions."""
    print(f"\n=== LORO validation across {len(regions)} regions ===")
    results = []
    for vi in range(len(regions)):
        rid = regions[vi]
        print(f"  LORO[{vi+1}/{len(regions)}] hold out {rid}")
        # Re-seed each fold for reproducibility
        torch.manual_seed(42 + vi); np.random.seed(42 + vi)
        model, val_mae = train_base_model(X, y, region_idx, val_region_idx=vi,
                                            epochs=epochs, hidden=hidden, verbose=False)
        results.append({'region': rid, 'val_mae': val_mae})
        print(f"    -> val_mae={val_mae:.4f}")
    df = pd.DataFrame(results)
    print(f"\nLORO summary: mean={df['val_mae'].mean():.4f}  median={df['val_mae'].median():.4f}"
          f"  min={df['val_mae'].min():.4f}  max={df['val_mae'].max():.4f}")
    return df


def ttt_inference(model, X_test, regions, ttt_steps=50, ttt_lr=1e-4,
                  mask_ratio=0.15, prior_lambda=0.1, verbose=False):
    """For each test region, fine-tune aux+last layers, then predict scores."""
    print(f"\n=== TTT inference on {len(regions)} test regions ===")
    # Save base weights for L2-anchor regularization
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    predictions = np.zeros((len(regions), N_HORIZONS), dtype=np.float32)
    t0 = time.time()
    for i, rid in enumerate(regions):
        # Reload base weights
        model.load_state_dict(base_state)
        # Set only last conv + aux head trainable
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.encoder.conv3.parameters():
            p.requires_grad_(True)
        for p in model.aux_head.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=ttt_lr, weight_decay=1e-3
        )
        x = torch.from_numpy(X_test[i]).unsqueeze(0).to(DEVICE)  # (1, 14, 91)
        model.train()
        for step in range(ttt_steps):
            opt.zero_grad()
            loss = masked_aux_loss(model, x, mask_ratio=mask_ratio)
            # Add L2 anchor to base weights
            anchor_loss = 0.0
            for k, p in model.named_parameters():
                if p.requires_grad:
                    anchor_loss = anchor_loss + ((p - base_state[k].to(DEVICE))**2).sum()
            loss = loss + prior_lambda * 0.001 * anchor_loss
            loss.backward()
            opt.step()
        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]  # (5,)
        predictions[i] = pred
        if (i+1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (len(regions) - i - 1) / (i + 1)
            if verbose:
                print(f"  region {i+1}/{len(regions)} elapsed={elapsed:.0f}s eta={eta:.0f}s")
    print(f"  done in {time.time() - t0:.1f}s")
    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=['loro', 'fit_and_predict', 'fast', 'ttt_only'], default='fast')
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ttt-steps", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--max-anchors", type=int, default=200)
    ap.add_argument("--output", default="submissions/submission_v18_track3_ttt.csv")
    ap.add_argument("--load-checkpoint", default=None, help="path to .pt to skip training")
    args = ap.parse_args()

    print("Loading train.csv...")
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    print(f"  train: {len(train_df)} rows")
    print("Loading test.csv...")
    test_df = pd.read_csv(ROOT / "data" / "test.csv")
    print(f"  test: {len(test_df)} rows")

    # Build train windows (subsample anchors per region for tractability)
    X_tr, y_tr, region_idx, regions = build_train_windows(train_df, max_anchors_per_region=args.max_anchors)
    print(f"Train tensor: X{X_tr.shape}  y{y_tr.shape}  regions={len(regions)}")

    # Compute standardization stats per channel
    stats = {
        'mean': X_tr.mean(axis=(0, 2)),  # per channel
        'std': X_tr.std(axis=(0, 2)),
    }
    print(f"channel means: {[f'{x:.2f}' for x in stats['mean']]}")
    X_tr_std = standardize_features(X_tr, stats)
    print(f"  standardized X_tr: min={X_tr_std.min():.2f} max={X_tr_std.max():.2f} mean={X_tr_std.mean():.4f}")

    # Build test windows
    X_te, test_regions = build_test_windows(test_df)
    print(f"Test tensor: X{X_te.shape}  regions={len(test_regions)}")
    X_te_std = standardize_features(X_te, stats)

    if args.mode == 'loro':
        loro_df = loro_validate(X_tr_std, y_tr, region_idx, regions,
                                 epochs=args.epochs, hidden=args.hidden)
        loro_df.to_csv(ROOT / "reports" / "_track3_loro.csv", index=False)
        return 0

    if args.load_checkpoint and Path(args.load_checkpoint).exists():
        print(f"\nLoading base checkpoint from {args.load_checkpoint}")
        ck = torch.load(args.load_checkpoint, map_location=DEVICE, weights_only=False)
        # Need to know hidden size from checkpoint
        # Try to infer from state dict
        weight_shape = ck['state_dict']['encoder.conv1.weight'].shape
        hidden = weight_shape[0]
        print(f"  inferred hidden={hidden}")
        model = DualHeadModel(hidden=hidden).to(DEVICE)
        model.load_state_dict(ck['state_dict'])
        # Use saved stats for standardization consistency
        stats = ck['stats']
        X_te_std = standardize_features(X_te, stats)
        val_mae = ck.get('val_mae', 0.0)
    else:
        # Mode: fit on all + predict
        print(f"\n=== Training base model on ALL {len(regions)} regions (epochs={args.epochs}) ===")
        torch.manual_seed(42); np.random.seed(42)
        # 10% random holdout for early stop
        model, val_mae = train_base_model(X_tr_std, y_tr, region_idx, val_region_idx=None,
                                           epochs=args.epochs, hidden=args.hidden, verbose=True)
        print(f"\nBase model val (random holdout) MAE: {val_mae:.4f}")
        # Save base model
        ck = ROOT / "checkpoints" / "track3_base.pt"
        ck.parent.mkdir(exist_ok=True)
        torch.save({'state_dict': model.state_dict(), 'stats': stats, 'val_mae': val_mae}, ck)
        print(f"Saved base checkpoint: {ck}")

    # TTT inference on 2248 test regions
    if args.mode == 'fast':
        # No TTT, just predict
        print("\n=== Fast inference (no TTT) ===")
        model.eval()
        with torch.no_grad():
            preds_list = []
            for i in range(0, len(X_te_std), 128):
                batch = torch.from_numpy(X_te_std[i:i+128]).to(DEVICE)
                p = model(batch).cpu().numpy()
                preds_list.append(p)
            test_preds = np.concatenate(preds_list, axis=0)
    else:
        test_preds = ttt_inference(model, X_te_std, test_regions,
                                     ttt_steps=args.ttt_steps, verbose=True)

    # Clip to [0, 5]
    test_preds = np.clip(test_preds, 0.0, 5.0)
    # Build CSV
    out_df = pd.DataFrame({'region_id': test_regions})
    for i, col in enumerate(PRED_COLS):
        out_df[col] = test_preds[:, i]
    out_path = ROOT / args.output
    out_path.parent.mkdir(exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nCandidate saved: {out_path}")
    print(f"  preds mean={test_preds.mean():.4f} std={test_preds.std():.4f}")
    print(f"  preds per horizon mean: {test_preds.mean(axis=0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
