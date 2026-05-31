#!/usr/bin/env python3
"""Plan v15 — Broad architecture sweep WITHOUT uploading.

For each variant, train and compute Pearson + MAD vs ext150. Estimate expected
5% blend public MAE via empirical slope law.

Variants:
- arch: GRU / LSTM / RNN
- no-weather: drop weather input
- no-score: drop score input
- residual-on-persistence: target = score - last_observed_score
- bigger / smaller model

Usage: train each, compute metrics, output sweep_results.csv table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
SUB = PROJECT_ROOT / "submissions"
REPORTS = PROJECT_ROOT / "reports"
PRED_COLS = [f"pred_week{i + 1}" for i in range(5)]

REGION_COL = "region_id"
DATE_COL = "date"
TARGET_COL = "score"
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
N_WEATHER = len(WEATHER_COLS)
N_CAL = 4  # sin/cos of week-of-year + sin/cos of half-year (long cycle)


def date_to_ord(s: str) -> int:
    y, m, d = [int(p) for p in s.split("-")]
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return y * 365 + sum(days_in_month[:m - 1]) + d


def calendar_features(start_ord: int, n_weeks: int) -> np.ndarray:
    days = start_ord + np.arange(n_weeks) * 7
    doy = days % 365
    w1 = 2 * np.pi * doy / 365.0
    w2 = 2 * np.pi * doy / 182.5
    out = np.stack([np.sin(w1), np.cos(w1), np.sin(w2), np.cos(w2)], axis=-1)
    return out.astype(np.float32)


def aggregate_to_weekly(daily_w, daily_s, daily_ord=None):
    n_days = daily_w.shape[0]
    n_weeks = n_days // 7
    weekly_w = daily_w[:n_weeks * 7].reshape(n_weeks, 7, daily_w.shape[1]).mean(axis=1)
    weekly_s = np.full(n_weeks, np.nan, dtype=np.float32)
    for w in range(n_weeks):
        s_week = daily_s[w * 7:(w + 1) * 7]
        valid = s_week[np.isfinite(s_week)]
        if valid.size > 0:
            weekly_s[w] = float(valid[0])
    if daily_ord is not None:
        start_ord = int(daily_ord[0])
        weekly_cal = calendar_features(start_ord, n_weeks)
        return weekly_w.astype(np.float32), weekly_s.astype(np.float32), weekly_cal
    return weekly_w.astype(np.float32), weekly_s.astype(np.float32)


def fill_score_nan(s: np.ndarray) -> np.ndarray:
    out = s.copy()
    last_valid = 0.0
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last_valid
        else:
            last_valid = out[i]
    return out


class WeeklyDataset(Dataset):
    def __init__(self, region_arrays, context_weeks, horizons, region_to_idx,
                 use_weather=True, use_score=True, residual_on_persistence=False,
                 use_calendar=False):
        self.context_weeks = context_weeks
        self.horizons = horizons
        self.region_arrays = region_arrays
        self.region_to_idx = region_to_idx
        self.use_weather = use_weather
        self.use_score = use_score
        self.residual = residual_on_persistence
        self.use_calendar = use_calendar
        self.samples = []
        for region, arrs in region_arrays.items():
            n_weeks = arrs["weather"].shape[0]
            if n_weeks >= context_weeks + horizons:
                last_anchor = n_weeks - horizons - 1
                for anchor in range(context_weeks - 1, last_anchor + 1):
                    future = arrs["score"][anchor + 1:anchor + 1 + horizons]
                    if np.isfinite(future).any():
                        self.samples.append((region, anchor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        region, anchor = self.samples[idx]
        arrs = self.region_arrays[region]
        K = self.context_weeks
        H = self.horizons
        ctx_w = arrs["weather"][anchor - K + 1:anchor + 1]  # (K, 14)
        ctx_s = arrs["score"][anchor - K + 1:anchor + 1]  # (K,)
        tgt_s = arrs["score"][anchor + 1:anchor + 1 + H]  # (H,)
        mask = np.isfinite(tgt_s).astype(np.float32)
        tgt_s = np.nan_to_num(tgt_s, nan=0.0)
        ctx_s = fill_score_nan(ctx_s)
        last_score = float(ctx_s[-1]) if len(ctx_s) > 0 else 0.0
        if self.residual:
            tgt_s = tgt_s - last_score  # predict deviation
        if self.use_calendar:
            ctx_cal = arrs["calendar"][anchor - K + 1:anchor + 1]
        else:
            ctx_cal = np.zeros((K, N_CAL), dtype=np.float32)
        return {
            "ctx_w": torch.from_numpy(ctx_w).float(),
            "ctx_s": torch.from_numpy(ctx_s).float(),
            "ctx_cal": torch.from_numpy(ctx_cal).float(),
            "tgt_s": torch.from_numpy(tgt_s).float(),
            "mask": torch.from_numpy(mask).float(),
            "last_score": torch.tensor(last_score, dtype=torch.float32),
            "region_idx": torch.tensor(self.region_to_idx[region], dtype=torch.long),
        }


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):  # x: (B, C, L)
        out = self.conv1(x)[:, :, :-self.pad] if self.pad > 0 else self.conv1(x)
        out = torch.relu(out)
        out = self.drop(out)
        out = self.conv2(out)[:, :, :-self.pad] if self.pad > 0 else self.conv2(out)
        skip = x if self.skip is None else self.skip(x)
        return torch.relu(out + skip)


class TCNEncoder(nn.Module):
    def __init__(self, in_ch, channels=(64, 128, 128), kernel=3, dropout=0.1):
        super().__init__()
        layers = []
        prev = in_ch
        for i, c in enumerate(channels):
            layers.append(TCNBlock(prev, c, kernel=kernel, dilation=2 ** i, dropout=dropout))
            prev = c
        self.net = nn.Sequential(*layers)
        self.out_dim = channels[-1]

    def forward(self, x):  # x: (B, L, C) → (B, C', L)
        x = x.transpose(1, 2)
        z = self.net(x)
        return z[:, :, -1]  # last timestep features (B, C')


class TransformerEncoder(nn.Module):
    def __init__(self, in_ch, d_model=128, n_heads=4, n_layers=2, dropout=0.1, max_len=512):
        super().__init__()
        self.input_proj = nn.Linear(in_ch, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                            dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_dim = d_model

    def forward(self, x):  # (B, L, C)
        z = self.input_proj(x) + self.pos[:, :x.shape[1]]
        z = self.enc(z)
        return z[:, -1]


class SeqEncDec(nn.Module):
    def __init__(self, n_regions, hidden=256, n_layers=2, dropout=0.1, region_emb_dim=32,
                 horizons=5, arch="GRU", use_weather=True, use_score=True, use_calendar=False):
        super().__init__()
        self.use_weather = use_weather
        self.use_score = use_score
        self.use_calendar = use_calendar
        input_dim = (N_WEATHER if use_weather else 0) + (1 if use_score else 0) + (N_CAL if use_calendar else 0)
        if input_dim == 0:
            raise ValueError("must use at least one input feature")
        self.arch = arch
        if arch in ("GRU", "LSTM", "RNN"):
            encoder_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}[arch]
            self.encoder = encoder_cls(
                input_size=input_dim, hidden_size=hidden, num_layers=n_layers,
                batch_first=True, bidirectional=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            state_dim = hidden * 2 * n_layers
        elif arch == "TCN":
            channels = tuple([hidden] * n_layers)
            self.encoder = TCNEncoder(input_dim, channels=channels, dropout=dropout)
            state_dim = self.encoder.out_dim
        elif arch == "TRANS":
            self.encoder = TransformerEncoder(input_dim, d_model=hidden, n_heads=4,
                                              n_layers=n_layers, dropout=dropout)
            state_dim = self.encoder.out_dim
        else:
            raise ValueError(f"unknown arch {arch}")
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        self.head = nn.Sequential(
            nn.Linear(state_dim + region_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, horizons),
        )

    def forward(self, ctx_w, ctx_s, ctx_cal, region_idx):
        parts = []
        if self.use_weather:
            parts.append(ctx_w)
        if self.use_score:
            parts.append(ctx_s.unsqueeze(-1))
        if self.use_calendar:
            parts.append(ctx_cal)
        x = torch.cat(parts, dim=-1)
        if self.arch == "LSTM":
            _, (h_n, _) = self.encoder(x)
            B = x.shape[0]
            state = h_n.transpose(0, 1).reshape(B, -1)
        elif self.arch in ("GRU", "RNN"):
            _, h_n = self.encoder(x)
            B = x.shape[0]
            state = h_n.transpose(0, 1).reshape(B, -1)
        else:  # TCN or TRANS
            state = self.encoder(x)
        r_emb = self.region_emb(region_idx)
        combined = torch.cat([state, r_emb], dim=-1)
        return self.head(combined)


def train_and_evaluate(args, region_arrays, test_weekly, region_order, region_to_idx, ext150_arr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WeeklyDataset(region_arrays, args.context_weeks, args.horizons, region_to_idx,
                             use_weather=args.use_weather, use_score=args.use_score,
                             residual_on_persistence=args.residual_on_persistence,
                             use_calendar=args.use_calendar)
    print(f"  {len(dataset):,} samples")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    model = SeqEncDec(
        n_regions=len(region_order), hidden=args.hidden, n_layers=args.n_layers,
        dropout=args.dropout, region_emb_dim=args.region_emb_dim, horizons=args.horizons,
        arch=args.arch, use_weather=args.use_weather, use_score=args.use_score,
        use_calendar=args.use_calendar,
    ).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    step = 0
    t0 = time.time()
    while step < args.num_steps:
        for batch in loader:
            if step >= args.num_steps:
                break
            ctx_w = batch["ctx_w"].to(device)
            ctx_s = batch["ctx_s"].to(device)
            ctx_cal = batch["ctx_cal"].to(device)
            tgt_s = batch["tgt_s"].to(device)
            mask = batch["mask"].to(device)
            region_idx = batch["region_idx"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                pred = model(ctx_w, ctx_s, ctx_cal, region_idx)
                diff = (pred - tgt_s).abs() * mask
                loss = diff.sum() / mask.sum().clamp_min(1.0)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            if step % 500 == 0:
                print(f"  step {step}/{args.num_steps}: loss={float(loss.detach()):.4f}")
            step += 1
    print(f"  train time: {time.time() - t0:.0f}s")

    # Inference
    model.eval()
    out_rows = []
    pred_cols_local = [f"pred_week{i + 1}" for i in range(args.horizons)]
    with torch.no_grad():
        for batch_start in range(0, len(region_order), 64):
            batch_regions = region_order[batch_start:batch_start + 64]
            valid_regions = [r for r in batch_regions if r in region_arrays]
            if not valid_regions:
                for r in batch_regions:
                    out_rows.append([r] + [0.5] * args.horizons)
                continue
            B = len(valid_regions)
            ctx_w_b = torch.zeros(B, args.context_weeks, N_WEATHER, device=device)
            ctx_s_b = torch.zeros(B, args.context_weeks, device=device)
            ctx_cal_b = torch.zeros(B, args.context_weeks, N_CAL, device=device)
            last_score_b = torch.zeros(B, device=device)
            region_idx = torch.tensor([region_to_idx[r] for r in valid_regions], dtype=torch.long, device=device)
            for i, r in enumerate(valid_regions):
                tw = region_arrays[r]["weather"]
                ts = region_arrays[r]["score"]
                test_w = test_weekly.get(r)
                if test_w is not None:
                    full_w = np.concatenate([tw, test_w], axis=0)
                    last_s = ts[~np.isnan(ts)][-1] if np.isfinite(ts).any() else 0.0
                    test_s = np.full(test_w.shape[0], last_s, dtype=np.float32)
                    full_s = np.concatenate([ts, test_s])
                else:
                    full_w = tw
                    full_s = ts
                ctx_w = full_w[-args.context_weeks:]
                cs = fill_score_nan(full_s[-args.context_weeks:])
                if args.use_calendar and "calendar" in region_arrays[r]:
                    tcal = region_arrays[r]["calendar"]
                    n_train = tcal.shape[0]
                    if test_w is not None:
                        start_after_train = int(region_arrays[r]["start_ord"]) + n_train * 7
                        test_cal = calendar_features(start_after_train, test_w.shape[0])
                        full_cal = np.concatenate([tcal, test_cal], axis=0)
                    else:
                        full_cal = tcal
                    cc = full_cal[-args.context_weeks:]
                    if cc.shape[0] < args.context_weeks:
                        pad = args.context_weeks - cc.shape[0]
                        cc = np.pad(cc, ((pad, 0), (0, 0)), mode="edge")
                else:
                    cc = np.zeros((args.context_weeks, N_CAL), dtype=np.float32)
                if ctx_w.shape[0] < args.context_weeks:
                    pad = args.context_weeks - ctx_w.shape[0]
                    ctx_w = np.pad(ctx_w, ((pad, 0), (0, 0)), mode="edge")
                    cs = np.pad(cs, (pad, 0), mode="edge")
                ctx_w_b[i] = torch.from_numpy(ctx_w).float().to(device)
                ctx_s_b[i] = torch.from_numpy(cs).float().to(device)
                ctx_cal_b[i] = torch.from_numpy(cc).float().to(device)
                last_score_b[i] = float(cs[-1])
            pred = model(ctx_w_b, ctx_s_b, ctx_cal_b, region_idx)
            if args.residual_on_persistence:
                pred = pred + last_score_b.unsqueeze(-1)
            pred = torch.clamp(pred, 0.0, 5.0).cpu().numpy()
            for i, r in enumerate(valid_regions):
                out_rows.append([r] + pred[i].tolist())
            for r in batch_regions:
                if r not in valid_regions:
                    out_rows.append([r] + [0.5] * args.horizons)

    out_df = pd.DataFrame(out_rows, columns=[REGION_COL] + pred_cols_local)
    out_df[REGION_COL] = out_df[REGION_COL].astype(str)

    # Metrics
    common_regions = list(set(out_df[REGION_COL]) & set(region_order))
    o = out_df.set_index(REGION_COL).loc[common_regions][pred_cols_local].to_numpy(dtype=np.float64)
    e_idx = [region_order.index(r) for r in common_regions]
    e = ext150_arr[e_idx]
    mad_raw = float(np.abs(o - e).mean())
    rhos = [float(np.corrcoef(o[:, h], e[:, h])[0, 1]) for h in range(args.horizons)]
    overall_mean = float(o.mean())
    # mean+std align
    o_aligned = o.copy()
    for h in range(args.horizons):
        if o[:, h].std() > 0.01:
            o_aligned[:, h] = e[:, h].mean() + (o[:, h] - o[:, h].mean()) * (e[:, h].std() / o[:, h].std())
    o_aligned = np.clip(o_aligned, 0, 5)
    mad_aligned = float(np.abs(o_aligned - e).mean())
    # 5% blend
    blend = 0.05 * o_aligned + 0.95 * e
    blend = np.clip(blend, 0, 5)
    mad_blend = float(np.abs(blend - e).mean())
    return {
        "raw_mean": overall_mean,
        "raw_mad": mad_raw,
        "pearson_per_horizon": rhos,
        "pearson_mean": float(np.mean(rhos)),
        "aligned_mad": mad_aligned,
        "blend_5pct_mad": mad_blend,
        "expected_public_at_slope_1.13": 0.8534 + 1.13 * mad_blend,
        "expected_public_at_slope_0.5": 0.8534 + 0.5 * mad_blend,
        "_blend_array": blend,
        "_common_regions": common_regions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, help="label for this variant")
    parser.add_argument("--arch", default="GRU", choices=["GRU", "LSTM", "RNN", "TCN", "TRANS"])
    parser.add_argument("--context-weeks", type=int, default=104)
    parser.add_argument("--horizons", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--region-emb-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--use-weather", action="store_true", default=True)
    parser.add_argument("--no-weather", dest="use_weather", action="store_false")
    parser.add_argument("--use-score", action="store_true", default=True)
    parser.add_argument("--no-score", dest="use_score", action="store_false")
    parser.add_argument("--residual-on-persistence", action="store_true", default=False)
    parser.add_argument("--use-calendar", action="store_true", default=False, help="add DOY sin/cos features")
    parser.add_argument("--save-candidate", default=None, help="optional path to save 5% blend candidate")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data once
    train = pd.read_csv(DATA / "train.csv", usecols=[REGION_COL, DATE_COL, TARGET_COL] + WEATHER_COLS)
    train[REGION_COL] = train[REGION_COL].astype(str)
    train[TARGET_COL] = pd.to_numeric(train[TARGET_COL], errors="coerce")
    test = pd.read_csv(DATA / "test.csv", usecols=[REGION_COL, DATE_COL] + WEATHER_COLS)
    test[REGION_COL] = test[REGION_COL].astype(str)
    sample = pd.read_csv(DATA / "sample_submission.csv", usecols=[REGION_COL])
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    region_order = sample[REGION_COL].tolist()
    region_to_idx = {r: i for i, r in enumerate(region_order)}

    region_arrays = {}
    test_weekly = {}
    for region, g in train.groupby(REGION_COL, sort=False):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        daily_w = g[WEATHER_COLS].to_numpy(dtype=np.float32)
        daily_s = g[TARGET_COL].to_numpy(dtype=np.float32)
        weekly_w, weekly_s = aggregate_to_weekly(daily_w, daily_s)
        start_ord = date_to_ord(g[DATE_COL].iloc[0])
        n_weeks = weekly_w.shape[0]
        weekly_cal = calendar_features(start_ord, n_weeks)
        region_arrays[str(region)] = {
            "weather": weekly_w, "score": weekly_s,
            "calendar": weekly_cal, "start_ord": start_ord,
        }
    for region, g in test.groupby(REGION_COL, sort=False):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        daily_w = g[WEATHER_COLS].to_numpy(dtype=np.float32)
        n_weeks = daily_w.shape[0] // 7
        if n_weeks > 0:
            weekly_w = daily_w[:n_weeks * 7].reshape(n_weeks, 7, N_WEATHER).mean(axis=1)
            test_weekly[str(region)] = weekly_w.astype(np.float32)

    all_w = np.concatenate([a["weather"] for a in region_arrays.values()], axis=0)
    w_mean = all_w.mean(axis=0)
    w_std = all_w.std(axis=0) + 1e-6
    for r in region_arrays:
        region_arrays[r]["weather"] = (region_arrays[r]["weather"] - w_mean) / w_std
    for r in test_weekly:
        test_weekly[r] = (test_weekly[r] - w_mean) / w_std

    ext150 = pd.read_csv(SUB / "submission_round5_pb30_x150_repro.csv")
    ext150[REGION_COL] = ext150[REGION_COL].astype(str)
    ext150 = ext150.set_index(REGION_COL).reindex(region_order).reset_index()
    ext150_arr = ext150[PRED_COLS].to_numpy(dtype=np.float64)

    print(f"\n=== Variant: {args.variant} (arch={args.arch}, weather={args.use_weather}, score={args.use_score}, residual={args.residual_on_persistence}) ===")
    metrics = train_and_evaluate(args, region_arrays, test_weekly, region_order, region_to_idx, ext150_arr)
    blend = metrics.pop("_blend_array")
    common_regions = metrics.pop("_common_regions")
    print(json.dumps(metrics, indent=2))
    # Save candidate CSV if requested
    if args.save_candidate:
        # Reindex to full region_order
        idx_map = {r: i for i, r in enumerate(common_regions)}
        out = np.zeros((len(region_order), args.horizons))
        for i, r in enumerate(region_order):
            if r in idx_map:
                out[i] = blend[idx_map[r]]
            else:
                out[i] = ext150_arr[i]
        df = pd.DataFrame(out, columns=PRED_COLS[:args.horizons])
        df.insert(0, REGION_COL, region_order)
        Path(args.save_candidate).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_candidate, index=False)
        print(f"[info] wrote candidate {args.save_candidate}")
    # Append to a JSON log
    log_path = REPORTS / "v15_sweep_results.json"
    log = []
    if log_path.exists():
        try:
            log = json.loads(log_path.read_text())
        except Exception:
            log = []
    log.append({"variant": args.variant, "args": vars(args), **metrics})
    log_path.write_text(json.dumps(log, indent=2))
    print(f"[info] appended to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
