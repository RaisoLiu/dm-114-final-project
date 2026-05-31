#!/usr/bin/env python3
"""Plan v13 — GRU Autoregressive Joint Weather+Score Model.

Architecture:
- Shared GRU(input_dim=16, hidden=256, layers=2) + region_embedding(2248→32)
- Per timestep t: input = [weather_t (14), prev_score (1), score_flag (1)] = 16-D
- Output: [weather_{t+1}_pred (14), score_t_pred (1)] = 15-D
- Step 0: weather_0 = real, prev_score = 0, flag = 0
- Step t > 0: weather_t = model's own weather_t pred (from step t-1), prev_score = model's score_{t-1} pred

Training:
- Random anchors per region; rollout T=60 timesteps fully autoregressive (no teacher forcing)
- Loss = MAE(score_pred, score_filled) + λ * MAE(weather_pred, weather_next)
- AdamW + CosineAnnealingLR + AMP on GB10

Inference:
- Initialize hidden state via warm-up over last 30 train days (with real data)
- Anchor at train_end_date; roll forward gap_days + 35 days fully autoregressively
- Extract scores at days gap+7, +14, +21, +28, +35
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


def fill_score_bucket(scores: np.ndarray) -> np.ndarray:
    """For each 7-day bucket starting from first non-NaN, broadcast that bucket's score to all 7 days."""
    n = scores.size
    out = np.full(n, np.nan, dtype=np.float32)
    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return out
    first_idx = int(np.argmax(finite_mask))
    for bucket_start in range(first_idx, n, 7):
        bucket_end = min(bucket_start + 7, n)
        bucket_scores = scores[bucket_start:bucket_end]
        valid = bucket_scores[np.isfinite(bucket_scores)]
        if valid.size > 0:
            out[bucket_start:bucket_end] = float(valid[0])
    # Backward fill before first score
    if first_idx > 0:
        out[:first_idx] = float(scores[first_idx])
    return out


# ---------- Data ----------
class SequenceDataset(Dataset):
    def __init__(self, region_arrays: dict, rollout_len: int, warmup: int, region_to_idx: dict):
        self.rollout_len = rollout_len
        self.warmup = warmup
        self.region_arrays = region_arrays
        self.region_to_idx = region_to_idx
        self.samples: list[tuple[str, int]] = []
        for region, arrs in region_arrays.items():
            n = arrs["weather"].shape[0]
            need = warmup + rollout_len + 1  # need weather_next at last step
            if n >= need:
                # anchors from `warmup` to `n - rollout_len - 1` (inclusive)
                last_anchor = n - rollout_len - 1
                for anchor in range(warmup, last_anchor + 1, 7):  # step 7 to reduce sample count
                    self.samples.append((region, anchor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        region, anchor = self.samples[idx]
        arrs = self.region_arrays[region]
        T = self.rollout_len
        # weather_t for t = 0..T-1: weather[anchor : anchor+T]
        weather_seq = arrs["weather"][anchor:anchor + T]  # (T, 14)
        # weather_{t+1} target: weather[anchor+1 : anchor+T+1]
        weather_next = arrs["weather"][anchor + 1:anchor + T + 1]  # (T, 14)
        # score_t target: score[anchor : anchor+T]
        score_seq = arrs["score"][anchor:anchor + T]  # (T,)
        # Warmup window for hidden state init: weather and score before anchor
        warm_start = anchor - self.warmup
        warm_weather = arrs["weather"][warm_start:anchor]  # (warmup, 14)
        warm_score = arrs["score"][warm_start:anchor]  # (warmup,)
        return {
            "weather": torch.from_numpy(weather_seq).float(),
            "weather_next": torch.from_numpy(weather_next).float(),
            "score": torch.from_numpy(score_seq).float(),
            "warm_weather": torch.from_numpy(warm_weather).float(),
            "warm_score": torch.from_numpy(warm_score).float(),
            "region_idx": torch.tensor(self.region_to_idx[region], dtype=torch.long),
        }


# ---------- Model ----------
class GRUAutoreg(nn.Module):
    def __init__(self, n_regions: int, hidden: int = 256, n_layers: int = 2, dropout: float = 0.1, region_emb_dim: int = 32):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        # Input: weather (14) + prev_score (1) + score_flag (1) = 16
        self.gru = nn.GRU(input_size=16, hidden_size=hidden, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        self.head = nn.Linear(hidden + region_emb_dim, N_WEATHER + 1)

    def init_hidden(self, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(self.n_layers, batch_size, self.hidden, device=device)

    def step(self, weather_t: torch.Tensor, prev_score: torch.Tensor, score_flag: torch.Tensor,
             h: torch.Tensor, region_idx: torch.Tensor):
        """Single timestep. Inputs shape (B, *), returns (weather_next_pred (B,14), score_t_pred (B,), new h)."""
        x = torch.cat([weather_t, prev_score, score_flag], dim=-1).unsqueeze(1)  # (B, 1, 16)
        out, h = self.gru(x, h)  # out: (B, 1, hidden)
        r_emb = self.region_emb(region_idx).unsqueeze(1)  # (B, 1, region_emb)
        combined = torch.cat([out, r_emb], dim=-1)  # (B, 1, hidden + region_emb)
        pred = self.head(combined).squeeze(1)  # (B, 15)
        weather_next_pred = pred[:, :N_WEATHER]
        score_t_pred = pred[:, N_WEATHER]
        return weather_next_pred, score_t_pred, h

    def warmup_hidden(self, warm_weather: torch.Tensor, warm_score: torch.Tensor, region_idx: torch.Tensor) -> torch.Tensor:
        """Run model over warmup window with REAL weather+score (teacher-forced) to init hidden state.
        warm_weather: (B, W, 14), warm_score: (B, W). Returns h: (n_layers, B, hidden)."""
        B, W, _ = warm_weather.shape
        device = warm_weather.device
        h = self.init_hidden(B, device)
        # We feed REAL data here, no autoregression (just to seed the hidden state)
        for w in range(W):
            weather_t = warm_weather[:, w]
            prev_score = warm_score[:, w].unsqueeze(-1)
            flag = torch.ones(B, 1, device=device)
            x = torch.cat([weather_t, prev_score, flag], dim=-1).unsqueeze(1)
            _, h = self.gru(x, h)
        return h


# ---------- Training ----------
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, lambda_weather: float, max_steps: int, step_offset: int = 0):
    model.train()
    losses_score = []
    losses_weather = []
    total_loss_running = []
    step = step_offset
    for batch in loader:
        if step >= max_steps:
            break
        weather = batch["weather"].to(device, non_blocking=True)  # (B, T, 14)
        weather_next = batch["weather_next"].to(device, non_blocking=True)  # (B, T, 14)
        score_target = batch["score"].to(device, non_blocking=True)  # (B, T)
        warm_w = batch["warm_weather"].to(device, non_blocking=True)
        warm_s = batch["warm_score"].to(device, non_blocking=True)
        region_idx = batch["region_idx"].to(device, non_blocking=True)
        B, T, _ = weather.shape

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            # Warmup hidden state
            h = model.warmup_hidden(warm_w, warm_s, region_idx)

            # Autoregressive rollout
            score_loss_accum = torch.zeros((), device=device)
            weather_loss_accum = torch.zeros((), device=device)
            n_score_steps = 0
            n_weather_steps = 0

            # Step 0: use real weather_0
            prev_weather = weather[:, 0]
            prev_score = torch.zeros(B, 1, device=device)
            score_flag = torch.zeros(B, 1, device=device)

            for t in range(T):
                weather_next_pred, score_t_pred, h = model.step(prev_weather, prev_score, score_flag, h, region_idx)
                # Score loss at step t
                score_loss_accum = score_loss_accum + (score_t_pred - score_target[:, t]).abs().mean()
                n_score_steps += 1
                # Weather loss: weather_next_pred should match weather_next[:, t] (= weather[:, t+1])
                if t < T - 1 or weather_next.shape[1] > t:
                    weather_loss_accum = weather_loss_accum + (weather_next_pred - weather_next[:, t]).abs().mean()
                    n_weather_steps += 1
                # Feed forward
                prev_weather = weather_next_pred  # autoregressive
                prev_score = score_t_pred.unsqueeze(-1)
                score_flag = torch.ones(B, 1, device=device)

            score_loss = score_loss_accum / max(1, n_score_steps)
            weather_loss = weather_loss_accum / max(1, n_weather_steps)
            loss = score_loss + lambda_weather * weather_loss

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

        losses_score.append(float(score_loss.detach()))
        losses_weather.append(float(weather_loss.detach()))
        total_loss_running.append(float(loss.detach()))
        if step % 100 == 0:
            print(f"  step {step}/{max_steps}: loss={float(loss.detach()):.4f} (score {float(score_loss.detach()):.4f}, weather {float(weather_loss.detach()):.4f})")
        step += 1
    return losses_score, losses_weather, step


# ---------- Inference ----------
@torch.no_grad()
def predict_test(model, region_arrays: dict, region_to_idx: dict, region_order: list, test_end_dates: dict,
                 train_end_dates: dict, warmup: int, device, batch_size: int = 64):
    """For each region: warmup hidden state, then roll forward (gap + 35) days autoregressively.
    Extract score predictions at days gap+7, +14, +21, +28, +35."""
    model.eval()
    out_rows = []
    # Group regions by their rollout_len for batching efficiency
    # rollout_len = gap_days + 35 (rough); we'll just batch a fixed max length per batch
    pred_per_region = {}

    for batch_start in range(0, len(region_order), batch_size):
        batch_regions = region_order[batch_start:batch_start + batch_size]
        # Filter regions that have data
        valid_regions = [r for r in batch_regions if r in region_arrays and r in train_end_dates and r in test_end_dates]
        if not valid_regions:
            for r in batch_regions:
                pred_per_region[r] = [0.5] * 5
            continue

        # Per-region T (rollout length)
        Ts = []
        gap_offsets = []
        for r in valid_regions:
            gap_days = date_to_ord(test_end_dates[r]) - date_to_ord(train_end_dates[r])
            T = gap_days + 35 + 7  # need a few buffer days
            Ts.append(T)
            gap_offsets.append(gap_days)
        T_max = max(Ts)

        B = len(valid_regions)
        # Build warmup tensors
        warm_w = torch.zeros(B, warmup, N_WEATHER, device=device)
        warm_s = torch.zeros(B, warmup, device=device)
        weather_0 = torch.zeros(B, N_WEATHER, device=device)
        for i, r in enumerate(valid_regions):
            arrs = region_arrays[r]
            n = arrs["weather"].shape[0]
            if n < warmup + 1:
                # fallback: use whatever is available
                w = arrs["weather"][:warmup]
                s = arrs["score"][:warmup]
                pad_w = max(0, warmup - w.shape[0])
                if pad_w > 0:
                    w = np.pad(w, ((pad_w, 0), (0, 0)), mode="edge")
                    s = np.pad(s, (pad_w, 0), mode="edge")
                warm_w[i] = torch.from_numpy(w).float().to(device)
                warm_s[i] = torch.from_numpy(s).float().to(device)
                weather_0[i] = torch.from_numpy(arrs["weather"][-1]).float().to(device)
            else:
                warm_w[i] = torch.from_numpy(arrs["weather"][-(warmup + 1):-1]).float().to(device)
                warm_s[i] = torch.from_numpy(arrs["score"][-(warmup + 1):-1]).float().to(device)
                weather_0[i] = torch.from_numpy(arrs["weather"][-1]).float().to(device)
        region_idx = torch.tensor([region_to_idx[r] for r in valid_regions], dtype=torch.long, device=device)

        # Warmup
        h = model.warmup_hidden(warm_w, warm_s, region_idx)

        # Rollout
        prev_weather = weather_0
        prev_score = torch.zeros(B, 1, device=device)
        score_flag = torch.zeros(B, 1, device=device)
        score_traj = torch.zeros(B, T_max, device=device)
        for t in range(T_max):
            weather_next_pred, score_t_pred, h = model.step(prev_weather, prev_score, score_flag, h, region_idx)
            score_traj[:, t] = score_t_pred
            prev_weather = weather_next_pred
            prev_score = score_t_pred.unsqueeze(-1)
            score_flag = torch.ones(B, 1, device=device)

        # Extract per-region predictions at the right offsets
        for i, r in enumerate(valid_regions):
            gap_days = gap_offsets[i]
            preds = []
            for h_w in range(1, 6):
                days_ahead = gap_days + 7 * h_w
                # idx within score_traj: day-1 in 0-indexed array
                # The model's step 0 produces score_0, which is for "today" (day = train_end + 1 in absolute terms).
                # Actually, day=0 in our rollout corresponds to "the day after train_end_date" (since weather_0 = last train weather, but we predict score for the day after).
                # Wait — re-read the design. weather_0 = last train weather (i.e., train_end_date's weather). score_0 = "score for train_end_date" (current).
                # So score_traj[i, t] is the score for day (train_end_date + t)? Or (train_end_date + t + 1)?
                # In our autoregressive setup: at step 0, input is weather_0 (real), output is score_0 (current day). So score_traj[0] is for day=train_end_date_day.
                # Wait but we want score at test_end + 7*h_w = train_end + gap + 7*h_w. Index t = gap + 7*h_w.
                idx = days_ahead
                idx = max(0, min(idx, T_max - 1))
                preds.append(float(torch.clamp(score_traj[i, idx], 0.0, 5.0).item()))
            pred_per_region[r] = preds

        # Regions not in valid_regions get fallback
        for r in batch_regions:
            if r not in pred_per_region:
                pred_per_region[r] = [0.5] * 5

    for r in region_order:
        out_rows.append([r] + pred_per_region.get(r, [0.5] * 5))
    return out_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-len", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--region-emb-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-weather", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-csv", default=str(SUB / "submission_gru_autoregressive.csv"))
    parser.add_argument("--report-json", default=str(REPORTS / "gru_autoregressive.json"))
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
    print(f"  {len(train):,} rows, {train[REGION_COL].nunique()} regions")

    test = pd.read_csv(DATA / "test.csv", usecols=[REGION_COL, DATE_COL])
    test[REGION_COL] = test[REGION_COL].astype(str)
    test_end_dates = test.groupby(REGION_COL, sort=False)[DATE_COL].last().to_dict()

    sample = pd.read_csv(DATA / "sample_submission.csv", usecols=[REGION_COL])
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    region_order = sample[REGION_COL].tolist()
    region_to_idx = {r: i for i, r in enumerate(region_order)}

    print("Preparing per-region arrays + NaN bucket-fill ...")
    region_arrays: dict[str, dict[str, np.ndarray]] = {}
    train_end_dates: dict[str, str] = {}
    weather_means: dict[str, np.ndarray] = {}
    weather_stds: dict[str, np.ndarray] = {}
    for region, g in train.groupby(REGION_COL, sort=False):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        weather = g[WEATHER_COLS].to_numpy(dtype=np.float32)
        score_raw = g[TARGET_COL].to_numpy(dtype=np.float32)
        score_filled = fill_score_bucket(score_raw)
        # If any NaN remains (shouldn't), replace with global mean
        if np.isnan(score_filled).any():
            score_filled = np.nan_to_num(score_filled, nan=float(np.nanmean(score_filled)))
        region_arrays[str(region)] = {"weather": weather, "score": score_filled}
        train_end_dates[str(region)] = str(g[DATE_COL].iloc[-1])

    # Global weather normalization (computed across all regions)
    all_w = np.concatenate([a["weather"] for a in region_arrays.values()], axis=0)
    w_mean = all_w.mean(axis=0)
    w_std = all_w.std(axis=0) + 1e-6
    print(f"  weather means: {w_mean.round(2)}")
    print(f"  weather stds: {w_std.round(2)}")
    for r in region_arrays:
        region_arrays[r]["weather"] = (region_arrays[r]["weather"] - w_mean) / w_std

    print("Building dataset ...")
    dataset = SequenceDataset(region_arrays, args.rollout_len, args.warmup, region_to_idx)
    print(f"  {len(dataset)} training samples")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    print("Building model ...")
    n_regions = len(region_order)
    model = GRUAutoreg(n_regions, hidden=args.hidden, n_layers=args.n_layers,
                       dropout=args.dropout, region_emb_dim=args.region_emb_dim).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"Training {args.num_steps} steps ...")
    t0 = time.time()
    step = 0
    while step < args.num_steps:
        _, _, step = train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                                      args.lambda_weather, args.num_steps, step_offset=step)
        print(f"  epoch done at step {step}/{args.num_steps}")
    print(f"  training took {time.time() - t0:.0f}s")

    print("Predicting test ...")
    out_rows = predict_test(model, region_arrays, region_to_idx, region_order, test_end_dates,
                            train_end_dates, args.warmup, device, batch_size=64)
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
        "n_regions": int(len(out_df)),
        "per_horizon_mean": [float(out_df[c].mean()) for c in PRED_COLS],
        "overall_mean": float(out_df[PRED_COLS].values.mean()),
        "min": float(out_df[PRED_COLS].values.min()),
        "max": float(out_df[PRED_COLS].values.max()),
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
