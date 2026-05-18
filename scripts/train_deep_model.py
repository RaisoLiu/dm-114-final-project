#!/usr/bin/env python3
"""GPU deep models for drought score prediction (Plan v5).

Three architectures via --arch:
- cnn:   1D-CNN encoder + region embedding + 5 heads
- lstm:  Bi-LSTM encoder + region embedding + 5 heads
- trans: Transformer encoder + region embedding + 5 heads

Outputs (gap_model-compatible OOF schema):
  - submissions/submission_deep_<arch>_s<seed>.csv
  - reports/deep_<arch>_s<seed>_validation_predictions.csv
  - reports/deep_<arch>_s<seed>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold

from drought.features import DATE_COL, REGION_COL, TARGET_COL, date_dayofyear, date_ordinal

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
N_HORIZONS = 5
N_WEATHER = len(WEATHER_COLS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", required=True)
    p.add_argument("--report-output", required=True)
    p.add_argument("--validation-pred-output", default=None)
    p.add_argument("--arch", choices=["cnn", "lstm", "trans"], required=True)
    p.add_argument("--gap-mode", choices=["blackout91", "public_like", "zero"], default="blackout91")
    p.add_argument("--seed", type=int, default=114)
    p.add_argument("--valid-deltas", default="728,735,742")
    p.add_argument("--train-samples-per-region", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--emb-dim", type=int, default=32)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", default=True, help="Mixed precision on CUDA")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--bounded-output", action="store_true", default=False,
                   help="Output = 5*sigmoid(raw). Prevents H1 saturation OOD seen in s114/s271828.")
    p.add_argument("--drop-cutoff-age", action="store_true", default=False,
                   help="Replace scalar[3] (anchor-cutoff) with 0 — suspected OOD-shifting feature.")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Linear LR warmup epochs before cosine. Recommended for transformer (e.g. 10).")
    return p.parse_args()


def score_targets(score_idx: np.ndarray, score_values: np.ndarray, anchor: int, horizons: int = N_HORIZONS):
    after_mask = score_idx > anchor
    if after_mask.sum() < horizons:
        return None
    after_idx = np.where(after_mask)[0][:horizons]
    return score_values[after_idx]


def score_cutoff_for_mode(anchor: int, gap: int, mode: str) -> int:
    if mode == "zero":
        return anchor
    if mode == "blackout91":
        return max(0, anchor - 91)
    return max(0, anchor - gap)


def build_dataset(args, train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, rng: np.random.Generator):
    region_order = sample[REGION_COL].astype(str).tolist()
    region_to_code = {r: i for i, r in enumerate(region_order)}

    train_meta = train.groupby(REGION_COL, sort=False).agg(train_end=(DATE_COL, "last"))
    test_meta = test.groupby(REGION_COL, sort=False).agg(test_end=(DATE_COL, "last"))
    train_end_ord = train_meta["train_end"].map(date_ordinal)
    test_end_ord = test_meta["test_end"].map(date_ordinal)
    public_like_gap_by_region = (test_end_ord - train_end_ord + 6).astype(int).to_dict()
    public_like_gaps = np.array(list(public_like_gap_by_region.values()), dtype=np.int32)

    norm_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for region, group in train.groupby(REGION_COL, sort=False):
        w = group[WEATHER_COLS].to_numpy(dtype=np.float32)
        norm_stats[str(region)] = (w.mean(axis=0), np.maximum(w.std(axis=0), 1e-6))

    valid_deltas = [int(d.strip()) for d in args.valid_deltas.split(",") if d.strip()]

    valid_rows = []
    train_rows = []

    for region_id, group in train.groupby(REGION_COL, sort=False):
        region = str(region_id)
        group = group.reset_index(drop=True)
        weather = group[WEATHER_COLS].to_numpy(dtype=np.float32)
        mean, std = norm_stats[region]
        score = pd.to_numeric(group[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)
        score_idx = np.flatnonzero(np.isfinite(score)).astype(np.int32)
        score_values = score[score_idx].astype(np.float32)
        last_usable = len(group) - 36
        if last_usable < 91:
            continue
        region_idx = region_to_code.get(region, len(region_to_code))
        gap = int(public_like_gap_by_region.get(region, 0))
        valid_anchors = [int(last_usable - d) for d in valid_deltas if last_usable - d >= 91]
        valid_anchor_set = set(valid_anchors)

        def make_row(anchor: int, sampled_gap: int):
            cutoff = score_cutoff_for_mode(anchor, sampled_gap, args.gap_mode)
            wnd = (weather[anchor - 90:anchor + 1] - mean) / std
            end_date = str(group.loc[anchor, DATE_COL])
            doy = date_dayofyear(end_date)
            scalars = [
                float(sampled_gap),
                float(np.sin(2 * np.pi * doy / 366.0)),
                float(np.cos(2 * np.pi * doy / 366.0)),
                0.0 if args.drop_cutoff_age else float(anchor - cutoff),
            ]
            return wnd.T, region_idx, scalars  # (14, 91)

        for anchor in valid_anchors:
            targets = score_targets(score_idx, score_values, anchor)
            if targets is None:
                continue
            wnd, ri, scalars = make_row(anchor, gap)
            valid_rows.append((wnd, ri, scalars, targets, region, anchor))

        candidate_anchors = np.arange(365, max(366, last_usable - 420), 7, dtype=np.int32)
        if candidate_anchors.size == 0:
            candidate_anchors = np.arange(91, last_usable + 1, 7, dtype=np.int32)
        sample_size = min(args.train_samples_per_region, candidate_anchors.size)
        anchors = rng.choice(candidate_anchors, size=sample_size, replace=False)
        for anchor in anchors:
            anchor = int(anchor)
            if anchor in valid_anchor_set:
                continue
            targets = score_targets(score_idx, score_values, anchor)
            if targets is None:
                continue
            sampled_gap = int(rng.choice(public_like_gaps))
            wnd, ri, scalars = make_row(anchor, sampled_gap)
            train_rows.append((wnd, ri, scalars, targets, region, anchor))

    # Test set
    test_groups = {str(region): group.reset_index(drop=True) for region, group in test.groupby(REGION_COL, sort=False)}
    train_groups = {str(region): group.reset_index(drop=True) for region, group in train.groupby(REGION_COL, sort=False)}
    test_rows = []
    for region in region_order:
        train_g = train_groups[region]
        test_g = test_groups[region]
        score = pd.to_numeric(train_g[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)
        score_idx = np.flatnonzero(np.isfinite(score)).astype(np.int32)
        mean, std = norm_stats[region]
        weather = test_g[WEATHER_COLS].to_numpy(dtype=np.float32)[-91:]
        wnd = (weather - mean) / std
        end_date = str(test_g[DATE_COL].iloc[-1])
        doy = date_dayofyear(end_date)
        anchor_idx = len(train_g) - 1 + int(test_end_ord[region] - train_end_ord[region])
        cutoff = int(score_idx[-1]) if score_idx.size else 0
        gap = int(public_like_gap_by_region[region])
        scalars = [float(gap), float(np.sin(2 * np.pi * doy / 366.0)), float(np.cos(2 * np.pi * doy / 366.0)),
                   0.0 if args.drop_cutoff_age else float(anchor_idx - cutoff)]
        test_rows.append((wnd.T, region_to_code[region], scalars, region))

    return {
        "valid": valid_rows,
        "train": train_rows,
        "test": test_rows,
        "region_order": region_order,
        "n_regions": len(region_to_code),
    }


def pack(rows, with_y=True):
    X = np.stack([r[0] for r in rows]).astype(np.float32)  # (N, 14, 91)
    R = np.array([r[1] for r in rows], dtype=np.int64)
    S = np.stack([r[2] for r in rows]).astype(np.float32)  # (N, 4)
    if with_y:
        Y = np.stack([r[3] for r in rows]).astype(np.float32)
        return X, R, S, Y
    return X, R, S


# === Architectures ===
class DeepCNN(nn.Module):
    def __init__(self, n_regions: int, emb_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(N_WEATHER, 64, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.region_emb = nn.Embedding(n_regions, emb_dim)
        d = 128 * 8 + emb_dim + 4
        self.head = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.horizons = nn.ModuleList([nn.Linear(256, 1) for _ in range(N_HORIZONS)])

    def forward(self, x, r, s):
        z = self.enc(x)
        e = self.region_emb(r)
        h = torch.cat([z, e, s], dim=1)
        h = self.head(h)
        return torch.cat([head(h) for head in self.horizons], dim=1)


class DeepLSTM(nn.Module):
    def __init__(self, n_regions: int, emb_dim: int = 32, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.region_emb = nn.Embedding(n_regions, emb_dim)
        self.in_proj = nn.Linear(N_WEATHER, 64)
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        d = 2 * hidden * 2 + emb_dim + 4  # mean+max pool over time, both directions
        self.head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.horizons = nn.ModuleList([nn.Linear(256, 1) for _ in range(N_HORIZONS)])

    def forward(self, x, r, s):
        # x: (B, 14, 91) -> (B, 91, 14)
        xt = x.transpose(1, 2)
        xt = self.in_proj(xt)
        out, _ = self.lstm(xt)  # (B, 91, 2*hidden)
        mp = out.mean(dim=1)
        mxp = out.max(dim=1).values
        pooled = torch.cat([mp, mxp], dim=1)
        e = self.region_emb(r)
        h = torch.cat([pooled, e, s], dim=1)
        h = self.head(h)
        return torch.cat([head(h) for head in self.horizons], dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DeepTrans(nn.Module):
    """Transformer encoder with CLS token + pre-LN.

    Larger than the original (d_model 128, dim_feedforward 512) and uses a learnable
    [CLS] token for pooling. The original (d=64, ff=128, mean-pool) gave val MAE 0.86
    — model couldn't differentiate regions because the 64-dim pooled bottleneck collapsed
    weather-token representations.
    """
    def __init__(self, n_regions: int, emb_dim: int = 32, d_model: int = 128, n_heads: int = 8, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.region_emb = nn.Embedding(n_regions, emb_dim)
        self.in_proj = nn.Linear(N_WEATHER, d_model)
        self.pos = PositionalEncoding(d_model, max_len=128)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        d = d_model + emb_dim + 4
        self.head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.horizons = nn.ModuleList([nn.Linear(256, 1) for _ in range(N_HORIZONS)])

    def forward(self, x, r, s):
        # x: (B, 14, 91) -> (B, 91, d_model)
        xt = x.transpose(1, 2)
        xt = self.in_proj(xt)
        xt = self.pos(xt)
        B = xt.shape[0]
        cls = self.cls.expand(B, -1, -1)
        xt = torch.cat([cls, xt], dim=1)  # (B, 92, d_model)
        z = self.encoder(xt)
        pooled = z[:, 0]  # CLS output
        e = self.region_emb(r)
        h = torch.cat([pooled, e, s], dim=1)
        h = self.head(h)
        return torch.cat([head(h) for head in self.horizons], dim=1)


class BoundedOutputWrapper(nn.Module):
    """Wraps a base model so output = 5 * sigmoid(raw). Caps predictions in (0, 5)."""
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x, r, s):
        raw = self.base(x, r, s)
        return 5.0 * torch.sigmoid(raw)


def build_model(arch: str, n_regions: int, args):
    if arch == "cnn":
        base = DeepCNN(n_regions, emb_dim=args.emb_dim, dropout=args.dropout)
    elif arch == "lstm":
        base = DeepLSTM(n_regions, emb_dim=args.emb_dim, dropout=args.dropout)
    elif arch == "trans":
        base = DeepTrans(n_regions, emb_dim=args.emb_dim, dropout=args.dropout)
    else:
        raise ValueError(f"unknown arch {arch}")
    if getattr(args, "bounded_output", False):
        return BoundedOutputWrapper(base)
    return base


def train_one_fold(X_tr, R_tr, S_tr, Y_tr, X_va, R_va, S_va, Y_va, n_regions, args, device):
    model = build_model(args.arch, n_regions, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.warmup_epochs > 0 and args.warmup_epochs < args.epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - args.warmup_epochs)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.L1Loss()
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ds_tr = TensorDataset(
        torch.from_numpy(X_tr), torch.from_numpy(R_tr), torch.from_numpy(S_tr), torch.from_numpy(Y_tr)
    )
    ds_va = TensorDataset(
        torch.from_numpy(X_va), torch.from_numpy(R_va), torch.from_numpy(S_va), torch.from_numpy(Y_va)
    )
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    best_va_mae = float("inf")
    best_state = None
    patience_counter = 0
    patience = 15
    for epoch in range(args.epochs):
        model.train()
        for xb, rb, sb, yb in dl_tr:
            xb, rb, sb, yb = xb.to(device), rb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred = model(xb, rb, sb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(xb, rb, sb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        sched.step()
        model.eval()
        va_mae = 0.0
        n = 0
        with torch.no_grad():
            for xb, rb, sb, yb in dl_va:
                xb, rb, sb, yb = xb.to(device), rb.to(device), sb.to(device), yb.to(device)
                pred = model(xb, rb, sb)
                va_mae += float(torch.abs(pred - yb).sum().item())
                n += yb.numel()
        va_mae /= max(1, n)
        if va_mae < best_va_mae - 1e-5:
            best_va_mae = va_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_va_mae


def predict(model, X, R, S, args, device):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(R), torch.from_numpy(S))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, pin_memory=device.type == "cuda")
    out = []
    model.eval()
    with torch.no_grad():
        for xb, rb, sb in dl:
            xb, rb, sb = xb.to(device), rb.to(device), sb.to(device)
            if args.amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    p = model(xb, rb, sb)
            else:
                p = model(xb, rb, sb)
            out.append(p.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[info] arch={args.arch} seed={args.seed} device={device}")
    if device.type == "cuda":
        print(f"[info] CUDA device: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_capability(0)})")

    data_dir = Path(args.data_dir)
    print("Loading data ...")
    train = pd.read_csv(data_dir / "train.csv", usecols=[REGION_COL, DATE_COL, TARGET_COL, *WEATHER_COLS])
    test = pd.read_csv(data_dir / "test.csv", usecols=[REGION_COL, DATE_COL, *WEATHER_COLS])
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    print("Building dataset (anchors + tensors) ...")
    data = build_dataset(args, train, test, sample, rng)
    print(f"[info] valid rows: {len(data['valid'])}; train rows: {len(data['train'])}; test rows: {len(data['test'])}")

    all_rows = data["train"] + data["valid"]
    X_all, R_all, S_all, Y_all = pack(all_rows, with_y=True)
    regions_all = np.array([r[4] for r in all_rows])

    t0 = time.time()
    gkf = GroupKFold(n_splits=args.n_folds)
    oof_pred = np.zeros((len(all_rows), N_HORIZONS), dtype=np.float32)
    oof_y = np.zeros((len(all_rows), N_HORIZONS), dtype=np.float32)
    fold_maes = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, Y_all, groups=regions_all)):
        print(f"[fold {fold + 1}/{args.n_folds}] train {len(tr_idx)} valid {len(va_idx)}")
        fold_t0 = time.time()
        model, va_mae = train_one_fold(
            X_all[tr_idx], R_all[tr_idx], S_all[tr_idx], Y_all[tr_idx],
            X_all[va_idx], R_all[va_idx], S_all[va_idx], Y_all[va_idx],
            n_regions=data["n_regions"], args=args, device=device,
        )
        oof_pred[va_idx] = predict(model, X_all[va_idx], R_all[va_idx], S_all[va_idx], args, device)
        oof_y[va_idx] = Y_all[va_idx]
        fold_maes.append(va_mae)
        print(f"  fold {fold + 1} val MAE: {va_mae:.4f}  ({time.time() - fold_t0:.1f}s)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof_mae = float(np.abs(oof_pred - oof_y).mean())
    print(f"[info] OOF MAE: {oof_mae:.4f}; fold MAEs: {[f'{m:.4f}' for m in fold_maes]}")

    # Refit-all using a tiny validation slice (just for early stopping)
    print("Refit on all data for test predictions ...")
    refit_va_size = max(64, int(0.05 * len(all_rows)))
    refit_perm = np.random.RandomState(args.seed).permutation(len(all_rows))
    refit_va = refit_perm[:refit_va_size]
    refit_tr = refit_perm[refit_va_size:]
    final_model, _ = train_one_fold(
        X_all[refit_tr], R_all[refit_tr], S_all[refit_tr], Y_all[refit_tr],
        X_all[refit_va], R_all[refit_va], S_all[refit_va], Y_all[refit_va],
        n_regions=data["n_regions"], args=args, device=device,
    )
    X_te, R_te, S_te = pack(data["test"], with_y=False)
    test_pred = predict(final_model, X_te, R_te, S_te, args, device)
    test_pred = np.clip(test_pred, 0.0, 5.0)

    region_order = data["region_order"]
    sub_df = pd.DataFrame(test_pred, columns=[f"pred_week{i + 1}" for i in range(N_HORIZONS)])
    sub_df.insert(0, REGION_COL, region_order)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_path, index=False)
    print(f"[info] wrote {out_path} (mean {sub_df.iloc[:, 1:].values.mean():.4f})")

    if args.validation_pred_output:
        val_rows = []
        valid_offset = len(data["train"])
        for i in range(len(data["valid"])):
            global_i = valid_offset + i
            region = data["valid"][i][4]
            for h in range(N_HORIZONS):
                val_rows.append({
                    "row_index": i,
                    "region_id": region,
                    "horizon": h + 1,
                    "y_true": float(Y_all[global_i, h]),
                    "pred_raw": float(oof_pred[global_i, h]),
                    "pred_horizon_calibrated": float(oof_pred[global_i, h]),
                    "pred_final_calibrated": float(oof_pred[global_i, h]),
                })
        pd.DataFrame(val_rows).to_csv(args.validation_pred_output, index=False)
        print(f"[info] wrote validation predictions: {args.validation_pred_output}")

    elapsed = time.time() - t0
    rep = {
        "arch": args.arch,
        "gap_mode": args.gap_mode,
        "seed": args.seed,
        "n_train": len(data["train"]),
        "n_valid": len(data["valid"]),
        "fold_maes": fold_maes,
        "oof_mae": oof_mae,
        "validation_mae": oof_mae,
        "test_prediction_mean": float(test_pred.mean()),
        "test_prediction_min": float(test_pred.min()),
        "test_prediction_max": float(test_pred.max()),
        "elapsed_seconds": elapsed,
        "device": str(device),
        "args": vars(args),
    }
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.report_output).open("w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(f"[info] wrote report: {args.report_output} (elapsed {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
