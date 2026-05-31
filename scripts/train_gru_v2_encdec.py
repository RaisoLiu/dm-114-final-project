#!/usr/bin/env python3
"""Plan v14 — GRU v2 weekly encoder-decoder, score-only, direct 5-horizon prediction.

Architectural improvements over v1 autoregressive:
1. Weekly cadence (not daily) — 7x fewer timesteps, no NaN issue, no rollout drift
2. Encoder-decoder — encoder sees full history (incl test weather), decoder predicts 5 horizons directly
3. Score-only — drop weather prediction, weather is just past covariate (input)
4. Bidirectional encoder — sees full context
5. Direct 5-horizon parallel head (no rollout)

Training: anchor randomly in train, encoder over previous K weeks, decoder predicts next 5 weeks.
Inference: encoder over full train weekly history + 13 test weeks (91 days), decoder predicts 5 weeks.
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


def date_to_ord(s: str) -> int:
    y, m, d = [int(p) for p in s.split("-")]
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return y * 365 + sum(days_in_month[:m - 1]) + d


def aggregate_to_weekly(daily_arr: np.ndarray, score_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given daily weather (n_days, 14) and daily score (n_days,, mostly NaN), return weekly aggregates.
    Weekly weather = mean per 7-day window. Weekly score = the observed score within that week (or NaN if none).
    """
    n_days = daily_arr.shape[0]
    n_weeks = n_days // 7
    weekly_w = daily_arr[:n_weeks * 7].reshape(n_weeks, 7, daily_arr.shape[1]).mean(axis=1)
    weekly_s = np.full(n_weeks, np.nan, dtype=np.float32)
    for w in range(n_weeks):
        s_week = score_arr[w * 7:(w + 1) * 7]
        valid = s_week[np.isfinite(s_week)]
        if valid.size > 0:
            weekly_s[w] = float(valid[0])
    return weekly_w.astype(np.float32), weekly_s.astype(np.float32)


# ---------- Data ----------
class WeeklySeqDataset(Dataset):
    """Each sample: encoder context (K weeks) + 5-week future targets.
    Anchor is the LAST week of encoder context (i.e., decoder predicts weeks anchor+1..anchor+5)."""
    def __init__(self, region_arrays: dict, context_weeks: int, horizons: int, region_to_idx: dict):
        self.context_weeks = context_weeks
        self.horizons = horizons
        self.region_arrays = region_arrays
        self.region_to_idx = region_to_idx
        self.samples = []
        for region, arrs in region_arrays.items():
            n_weeks = arrs["weather"].shape[0]
            need = context_weeks + horizons
            if n_weeks >= need:
                # Each valid anchor must have at least `horizons` non-NaN future scores
                # Sample every `stride` weeks to reduce sample count
                stride = 1
                last_anchor = n_weeks - horizons - 1
                for anchor in range(context_weeks - 1, last_anchor + 1, stride):
                    # check that targets are not all NaN
                    future_scores = arrs["score"][anchor + 1:anchor + 1 + horizons]
                    if np.isfinite(future_scores).any():
                        self.samples.append((region, anchor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        region, anchor = self.samples[idx]
        arrs = self.region_arrays[region]
        K = self.context_weeks
        H = self.horizons
        # Encoder context: weeks [anchor - K + 1, anchor]
        ctx_w = arrs["weather"][anchor - K + 1:anchor + 1]  # (K, 14)
        ctx_s = arrs["score"][anchor - K + 1:anchor + 1]  # (K,)
        # Targets: weeks [anchor + 1, anchor + H]
        tgt_s = arrs["score"][anchor + 1:anchor + 1 + H]  # (H,)
        # Mask: which target horizons have observed score (rest contribute 0 loss)
        mask = np.isfinite(tgt_s).astype(np.float32)
        tgt_s = np.nan_to_num(tgt_s, nan=0.0)
        # NaN-fill context score with forward-fill then backward-fill within context (or 0 if region just started)
        ctx_s = ctx_s.copy()
        last_valid = 0.0
        for i in range(len(ctx_s)):
            if np.isnan(ctx_s[i]):
                ctx_s[i] = last_valid
            else:
                last_valid = ctx_s[i]
        return {
            "ctx_w": torch.from_numpy(ctx_w).float(),
            "ctx_s": torch.from_numpy(ctx_s).float(),
            "tgt_s": torch.from_numpy(tgt_s).float(),
            "mask": torch.from_numpy(mask).float(),
            "region_idx": torch.tensor(self.region_to_idx[region], dtype=torch.long),
        }


# ---------- Model ----------
class GRUEncDec(nn.Module):
    def __init__(self, n_regions: int, hidden: int = 256, n_layers: int = 2, dropout: float = 0.1,
                 region_emb_dim: int = 32, horizons: int = 5):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.horizons = horizons
        # Encoder: bidirectional GRU
        self.encoder = nn.GRU(
            input_size=N_WEATHER + 1,  # 14 weather + 1 score
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        # After encoding, take final hidden states from both directions + region emb → 5 horizon heads
        # Output dim of encoder final: hidden * 2 * n_layers
        encoder_state_dim = hidden * 2 * n_layers
        self.head = nn.Sequential(
            nn.Linear(encoder_state_dim + region_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, horizons),
        )

    def forward(self, ctx_w: torch.Tensor, ctx_s: torch.Tensor, region_idx: torch.Tensor) -> torch.Tensor:
        """ctx_w: (B, K, 14), ctx_s: (B, K). Returns (B, horizons)."""
        x = torch.cat([ctx_w, ctx_s.unsqueeze(-1)], dim=-1)  # (B, K, 15)
        _, h_n = self.encoder(x)  # h_n: (n_layers * 2, B, hidden)
        # h_n shape: (n_layers * 2, B, hidden) — reshape to (B, n_layers * 2 * hidden)
        B = ctx_w.shape[0]
        encoder_state = h_n.transpose(0, 1).reshape(B, -1)  # (B, n_layers * 2 * hidden)
        r_emb = self.region_emb(region_idx)  # (B, region_emb_dim)
        combined = torch.cat([encoder_state, r_emb], dim=-1)
        pred = self.head(combined)  # (B, horizons)
        return pred


# ---------- Training ----------
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, max_steps: int, step_offset: int = 0):
    model.train()
    step = step_offset
    losses = []
    for batch in loader:
        if step >= max_steps:
            break
        ctx_w = batch["ctx_w"].to(device, non_blocking=True)
        ctx_s = batch["ctx_s"].to(device, non_blocking=True)
        tgt_s = batch["tgt_s"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        region_idx = batch["region_idx"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            pred = model(ctx_w, ctx_s, region_idx)  # (B, H)
            # Masked L1 loss
            diff = (pred - tgt_s).abs() * mask
            loss = diff.sum() / mask.sum().clamp_min(1.0)

        if scaler is not None:
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

        losses.append(float(loss.detach()))
        if step % 100 == 0:
            print(f"  step {step}/{max_steps}: loss={float(loss.detach()):.4f}")
        step += 1
    return losses, step


# ---------- Inference ----------
@torch.no_grad()
def predict_test(model, region_arrays: dict, region_to_idx: dict, region_order: list,
                 test_weekly: dict, context_weeks: int, device, batch_size: int = 64):
    """For each region: encoder context = last (context_weeks) weeks of (train_history + test_weeks).
    Decoder predicts pred_week_1..5.
    """
    model.eval()
    out_rows = []
    for batch_start in range(0, len(region_order), batch_size):
        batch_regions = region_order[batch_start:batch_start + batch_size]
        valid_regions = [r for r in batch_regions if r in region_arrays]
        if not valid_regions:
            for r in batch_regions:
                out_rows.append([r] + [0.5] * 5)
            continue

        B = len(valid_regions)
        ctx_w_batch = torch.zeros(B, context_weeks, N_WEATHER, device=device)
        ctx_s_batch = torch.zeros(B, context_weeks, device=device)
        region_idx = torch.tensor([region_to_idx[r] for r in valid_regions], dtype=torch.long, device=device)

        for i, r in enumerate(valid_regions):
            train_w = region_arrays[r]["weather"]  # (n_weeks, 14)
            train_s = region_arrays[r]["score"]
            # Append test weather (13 weekly aggregates of test 91 days)
            test_w = test_weekly.get(r)
            if test_w is not None:
                full_w = np.concatenate([train_w, test_w], axis=0)
                # No test score, pad with NaN-fill (last train score)
                last_s = train_s[~np.isnan(train_s)][-1] if np.isfinite(train_s).any() else 0.0
                test_s = np.full(test_w.shape[0], last_s, dtype=np.float32)
                full_s = np.concatenate([train_s, test_s])
            else:
                full_w = train_w
                full_s = train_s
            # Take last `context_weeks` weeks
            ctx_w = full_w[-context_weeks:]
            ctx_s = full_s[-context_weeks:]
            # NaN-fill ctx_s
            cs = ctx_s.copy()
            last_valid = 0.0
            for j in range(len(cs)):
                if np.isnan(cs[j]):
                    cs[j] = last_valid
                else:
                    last_valid = cs[j]
            # Pad if too short
            if ctx_w.shape[0] < context_weeks:
                pad = context_weeks - ctx_w.shape[0]
                ctx_w = np.pad(ctx_w, ((pad, 0), (0, 0)), mode="edge")
                cs = np.pad(cs, (pad, 0), mode="edge")
            ctx_w_batch[i] = torch.from_numpy(ctx_w).float().to(device)
            ctx_s_batch[i] = torch.from_numpy(cs).float().to(device)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            pred = model(ctx_w_batch, ctx_s_batch, region_idx)
        pred = torch.clamp(pred, 0.0, 5.0).cpu().numpy()
        for i, r in enumerate(valid_regions):
            out_rows.append([r] + pred[i].tolist())
        for r in batch_regions:
            if r not in valid_regions:
                out_rows.append([r] + [0.5] * 5)
    return out_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-weeks", type=int, default=104, help="encoder context length in weeks")
    parser.add_argument("--horizons", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--region-emb-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-csv", default=str(SUB / "submission_gru_v2_encdec.csv"))
    parser.add_argument("--report-json", default=str(REPORTS / "gru_v2_encdec.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("Loading data ...")
    train = pd.read_csv(DATA / "train.csv", usecols=[REGION_COL, DATE_COL, TARGET_COL] + WEATHER_COLS)
    train[REGION_COL] = train[REGION_COL].astype(str)
    train[TARGET_COL] = pd.to_numeric(train[TARGET_COL], errors="coerce")
    test = pd.read_csv(DATA / "test.csv", usecols=[REGION_COL, DATE_COL] + WEATHER_COLS)
    test[REGION_COL] = test[REGION_COL].astype(str)
    sample = pd.read_csv(DATA / "sample_submission.csv", usecols=[REGION_COL])
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    region_order = sample[REGION_COL].tolist()
    region_to_idx = {r: i for i, r in enumerate(region_order)}

    print("Aggregating to weekly per region ...")
    region_arrays: dict[str, dict[str, np.ndarray]] = {}
    test_weekly: dict[str, np.ndarray] = {}
    for region, g in train.groupby(REGION_COL, sort=False):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        daily_w = g[WEATHER_COLS].to_numpy(dtype=np.float32)
        daily_s = g[TARGET_COL].to_numpy(dtype=np.float32)
        weekly_w, weekly_s = aggregate_to_weekly(daily_w, daily_s)
        region_arrays[str(region)] = {"weather": weekly_w, "score": weekly_s}
    for region, g in test.groupby(REGION_COL, sort=False):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        daily_w = g[WEATHER_COLS].to_numpy(dtype=np.float32)
        n_weeks = daily_w.shape[0] // 7
        if n_weeks > 0:
            weekly_w = daily_w[:n_weeks * 7].reshape(n_weeks, 7, N_WEATHER).mean(axis=1)
            test_weekly[str(region)] = weekly_w.astype(np.float32)

    # Global weather normalization
    all_w = np.concatenate([a["weather"] for a in region_arrays.values()], axis=0)
    w_mean = all_w.mean(axis=0)
    w_std = all_w.std(axis=0) + 1e-6
    print(f"  weather means: {w_mean.round(2)}")
    print(f"  weather stds: {w_std.round(2)}")
    for r in region_arrays:
        region_arrays[r]["weather"] = (region_arrays[r]["weather"] - w_mean) / w_std
    for r in test_weekly:
        test_weekly[r] = (test_weekly[r] - w_mean) / w_std

    print("Building dataset ...")
    dataset = WeeklySeqDataset(region_arrays, args.context_weeks, args.horizons, region_to_idx)
    print(f"  {len(dataset):,} samples")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    print("Building model ...")
    model = GRUEncDec(
        n_regions=len(region_order),
        hidden=args.hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
        region_emb_dim=args.region_emb_dim,
        horizons=args.horizons,
    ).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"Training {args.num_steps} steps ...")
    t0 = time.time()
    step = 0
    while step < args.num_steps:
        _, step = train_one_epoch(model, loader, optimizer, scheduler, scaler, device, args.num_steps, step_offset=step)
        print(f"  epoch end at step {step}/{args.num_steps}")
    print(f"  training took {time.time() - t0:.0f}s")

    print("Predicting test ...")
    out_rows = predict_test(model, region_arrays, region_to_idx, region_order, test_weekly,
                            args.context_weeks, device, batch_size=64)
    out_df = pd.DataFrame(out_rows, columns=[REGION_COL] + PRED_COLS)
    out_df[REGION_COL] = out_df[REGION_COL].astype(str)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[info] wrote {out_path}")

    # Sanity
    print("\nSanity checks ...")
    print(f"  per-horizon mean: {[round(out_df[c].mean(), 4) for c in PRED_COLS]}")
    print(f"  overall mean: {out_df[PRED_COLS].values.mean():.4f}")
    print(f"  range: [{out_df[PRED_COLS].values.min():.4f}, {out_df[PRED_COLS].values.max():.4f}]")
    ext150 = pd.read_csv(SUB / "submission_round5_pb30_x150_repro.csv")
    ext150[REGION_COL] = ext150[REGION_COL].astype(str)
    common = list(set(out_df[REGION_COL]) & set(ext150[REGION_COL]))
    e = ext150.set_index(REGION_COL).loc[common][PRED_COLS].to_numpy(dtype=np.float64)
    o = out_df.set_index(REGION_COL).loc[common][PRED_COLS].to_numpy(dtype=np.float64)
    mad = float(np.abs(o - e).mean())
    rhos = [float(np.corrcoef(o[:, h], e[:, h])[0, 1]) for h in range(5)]
    print(f"  MAD vs ext150: {mad:.4f}")
    print(f"  Pearson per horizon: {[round(r, 4) for r in rhos]}")

    report = {
        "args": vars(args),
        "per_horizon_mean": [float(out_df[c].mean()) for c in PRED_COLS],
        "overall_mean": float(out_df[PRED_COLS].values.mean()),
        "mad_vs_ext150": mad,
        "pearson_vs_ext150": rhos,
    }
    rp = Path(args.report_json)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[info] wrote {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
