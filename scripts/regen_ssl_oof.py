#!/usr/bin/env python3
"""Regenerate Track 1 SSL Transformer OOF predictions aligned to oof_tensor.csv row_index.

Loads track1_finetuned.pt and runs inference over the same 6,744 (region, anchor)
validation tuples that train_deep_model.py uses (anchors = last_usable - {728,735,742}).
Standardizes inputs with the GLOBAL (over-all-regions) weather mean/std that the
SSL pretrain/finetune used.

Output schema matches deep_cnn_*_validation_predictions.csv so it joins on
(row_index, region_id, horizon) into oof_tensor.
"""
from __future__ import annotations
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TRAIN = ROOT / "data" / "train.csv"
SAMPLE = ROOT / "data" / "sample_submission.csv"
CKPT = ROOT / "checkpoints" / "track1_finetuned.pt"
OUT = ROOT / "reports" / "track1_ssl_oof_validation_predictions.csv"

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
N_CHANNELS = 14
N_DAYS = 91
N_HORIZONS = 5
VALID_DELTAS = [728, 735, 742]
REGION_COL = "region_id"
DATE_COL = "date"
TARGET_COL = "score"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model definition copied verbatim from scripts/track1_ssl_pretrain.py
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class SSLTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.proj = nn.Linear(N_CHANNELS, d_model)
        self.pos = PosEnc(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, activation="gelu",
        )
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

    def encode(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        return x

    def predict_score(self, x):
        z = self.encode(x).transpose(1, 2)
        return self.score_head(z)


def score_targets(score_idx: np.ndarray, score_values: np.ndarray, anchor: int) -> np.ndarray | None:
    after_mask = score_idx > anchor
    if after_mask.sum() < N_HORIZONS:
        return None
    after_idx = np.where(after_mask)[0][:N_HORIZONS]
    return score_values[after_idx]


def main() -> None:
    print(f"[ssl] device={DEVICE}", flush=True)
    print(f"[ssl] loading {TRAIN}", flush=True)
    train = pd.read_csv(TRAIN)
    sample = pd.read_csv(SAMPLE)
    region_order = sample[REGION_COL].astype(str).tolist()

    # Build per-region weather arrays and global stats
    groups = {str(r): g.reset_index(drop=True) for r, g in train.groupby(REGION_COL, sort=False)}
    region_weathers: dict[str, np.ndarray] = {}
    region_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for region in region_order:
        g = groups.get(region)
        if g is None or len(g) == 0:
            continue
        w = g[WEATHER_COLS].to_numpy(dtype=np.float32)
        s = pd.to_numeric(g[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)
        s_idx = np.flatnonzero(np.isfinite(s)).astype(np.int32)
        s_vals = s[s_idx]
        region_weathers[region] = w
        region_scores[region] = (s_idx, s_vals)

    # Global mean/std across all-region-all-time weather (matches WeatherWindowDataset stats)
    all_w = np.concatenate([w for w in region_weathers.values()], axis=0)  # (R*T, 14)
    mean = all_w.mean(axis=0)
    std = all_w.std(axis=0) + 1e-6
    print(f"[ssl] global stats: mean.shape={mean.shape} std.shape={std.shape}", flush=True)

    # Enumerate the 6,744 validation (region, anchor) tuples in the same order as train_deep_model.py
    val_tuples: list[tuple[str, int, np.ndarray]] = []  # (region, anchor, y_true(5))
    for region in region_order:
        g = groups.get(region)
        if g is None or len(g) == 0:
            continue
        s_idx, s_vals = region_scores[region]
        last_usable = len(g) - 36
        if last_usable < 91:
            continue
        valid_anchors = [int(last_usable - d) for d in VALID_DELTAS if last_usable - d >= 91]
        for anchor in valid_anchors:
            tgt = score_targets(s_idx, s_vals, anchor)
            if tgt is None:
                continue
            val_tuples.append((region, anchor, tgt))
    print(f"[ssl] {len(val_tuples)} validation anchors", flush=True)

    # Load checkpoint
    print(f"[ssl] loading {CKPT}", flush=True)
    model = SSLTransformer().to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ssl] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[ssl]   missing keys (first 5): {missing[:5]}", flush=True)
    if unexpected:
        print(f"[ssl]   unexpected keys (first 5): {unexpected[:5]}", flush=True)
    model.eval()

    # Batched inference
    BATCH = 256
    rows: list[dict] = []
    t0 = time.time()
    with torch.no_grad():
        for batch_start in range(0, len(val_tuples), BATCH):
            batch = val_tuples[batch_start : batch_start + BATCH]
            X = np.zeros((len(batch), N_CHANNELS, N_DAYS), dtype=np.float32)
            for k, (region, anchor, _) in enumerate(batch):
                w = region_weathers[region][anchor - N_DAYS + 1 : anchor + 1, :]  # (91, 14)
                w_std = (w - mean) / std
                X[k] = w_std.T  # (14, 91)
            xb = torch.from_numpy(X).to(DEVICE)
            yhat = model.predict_score(xb).cpu().numpy()  # (B, 5)
            for k, (region, anchor, ytrue) in enumerate(batch):
                row_index = batch_start + k
                for h in range(N_HORIZONS):
                    pred = float(yhat[k, h])
                    rows.append({
                        "row_index": row_index,
                        "region_id": region,
                        "horizon": h + 1,
                        "y_true": float(ytrue[h]),
                        "pred_raw": pred,
                        "pred_horizon_calibrated": pred,
                        "pred_final_calibrated": pred,
                    })
            if batch_start % (BATCH * 4) == 0:
                elapsed = time.time() - t0
                eta = (elapsed / max(1, batch_start + len(batch))) * (len(val_tuples) - batch_start - len(batch))
                print(f"[ssl] {batch_start + len(batch)}/{len(val_tuples)} elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    mae = float(np.mean(np.abs(df.pred_final_calibrated - df.y_true)))
    print(f"[ssl] wrote {OUT}: rows={len(df)} OOF MAE={mae:.4f} total_time={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
