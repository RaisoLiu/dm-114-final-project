#!/usr/bin/env python3
"""Adversarial validation for Plan v6.

Train a binary LightGBM to distinguish *test anchors* (positive: the actual
`last_usable + public_like_gap` rows that Kaggle scores) from candidate
*training anchors* sampled across each region's history. Then score every
weekly candidate training anchor by `p_test`. The training anchors with the
highest scores are most "test-like" and become the new training pool /
validation slice for Plan v6 (replacing the misaligned `valid_deltas
{728,735,742}`).

Output: ``reports/adversarial_scores.parquet`` with columns
``region_id, anchor_index, delta_from_last_usable, p_test``.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/adversarial_validation.py

Compact feature set (no use of `make_row` to avoid leaking cutoff_age,
no recency-from-last_usable to avoid trivial separator):
- region_code (ordinal) — distinguishes regions but is *the same* between any
  train-vs-test pair, so doesn't trivially separate the two classes.
- DOY sin/cos of end_date — test anchors have a specific DOY band that
  training anchors only sometimes hit.
- 14 weather variables × {mean, std, last_7_mean, last_28_mean} = 56 features.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drought.features import DATE_COL, REGION_COL, date_dayofyear, date_ordinal  # noqa: E402

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
N_WEATHER = len(WEATHER_COLS)

# Number of randomly-sampled negative training anchors per region.
# Uniform sampling across the candidate-anchor range to avoid teaching the
# classifier to recognize specific deltas. With 2248 regions × 80 = ~180k
# negatives and 2248 positives, class imbalance is handled via scale_pos_weight.
NEG_SAMPLES_PER_REGION = 80


def weather_window_features(window: np.ndarray) -> np.ndarray:
    """window shape (>=91, N_WEATHER) -> length 4*N_WEATHER vector.

    Features: per-variable mean, std, last-7-day mean, last-28-day mean.
    """
    if window.shape[0] < 91:
        # Pad with last row if too short (shouldn't normally happen)
        pad = np.tile(window[-1:], (91 - window.shape[0], 1))
        window = np.concatenate([pad, window], axis=0)
    last91 = window[-91:]
    last7 = window[-7:]
    last28 = window[-28:]
    return np.concatenate([
        last91.mean(axis=0),
        last91.std(axis=0),
        last7.mean(axis=0),
        last28.mean(axis=0),
    ], dtype=np.float32)


def doy_features(end_date: str) -> tuple[float, float]:
    doy = date_dayofyear(end_date)
    angle = 2.0 * math.pi * doy / 366.0
    return math.sin(angle), math.cos(angle)


def build_anchor_row(weather: np.ndarray, end_date: str, region_code: int) -> list[float]:
    feats: list[float] = [float(region_code)]
    feats.extend(doy_features(end_date))
    feats.extend(weather_window_features(weather).tolist())
    return feats


def feature_columns() -> list[str]:
    cols = ["region_code", "doy_sin", "doy_cos"]
    for stat in ("mean", "std", "last7_mean", "last28_mean"):
        for w in WEATHER_COLS:
            cols.append(f"{w}_{stat}")
    return cols


def main() -> int:
    data_dir = PROJECT_ROOT / "data"
    train = pd.read_csv(data_dir / "train.csv", usecols=[REGION_COL, DATE_COL, *WEATHER_COLS, "score"])
    test = pd.read_csv(data_dir / "test.csv", usecols=[REGION_COL, DATE_COL, *WEATHER_COLS])
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    region_order = sample[REGION_COL].astype(str).tolist()
    region_to_code = {r: i for i, r in enumerate(region_order)}
    print(f"[info] train rows {len(train):,}, test rows {len(test):,}, regions {len(region_order)}")

    # Pre-group
    train_groups = {str(r): g.reset_index(drop=True) for r, g in train.groupby(REGION_COL, sort=False)}
    test_groups = {str(r): g.reset_index(drop=True) for r, g in test.groupby(REGION_COL, sort=False)}

    # Per-region metadata: last_usable_anchor, public_like_gap
    train_end_ord = train.groupby(REGION_COL, sort=False)[DATE_COL].agg("last").map(date_ordinal)
    test_end_ord = test.groupby(REGION_COL, sort=False)[DATE_COL].agg("last").map(date_ordinal)
    public_like_gap_by_region = (test_end_ord - train_end_ord + 6).astype(int).to_dict()

    rng = np.random.default_rng(20260516)

    print("[step 1] Build positive (test anchors) + negative (sampled training anchors) rows ...")
    pos_rows: list[list[float]] = []
    neg_rows: list[list[float]] = []
    neg_meta: list[tuple[str, int, int]] = []  # (region, anchor_index, delta_from_last_usable)
    candidate_rows: list[list[float]] = []
    candidate_meta: list[tuple[str, int, int]] = []  # (region, anchor_index, delta)

    for region in region_order:
        train_g = train_groups[region]
        test_g = test_groups[region]
        weather_train = train_g[WEATHER_COLS].to_numpy(dtype=np.float32)
        weather_test = test_g[WEATHER_COLS].to_numpy(dtype=np.float32)
        last_usable_anchor = len(train_g) - 36
        if last_usable_anchor < 91:
            continue
        region_code = region_to_code[region]

        # Positive: the actual test anchor (end of test history, 91-day window)
        test_window = weather_test[-91:]
        test_end_date = str(test_g[DATE_COL].iloc[-1])
        pos_rows.append(build_anchor_row(test_window, test_end_date, region_code))

        # Negative samples: uniform random anchors across the training history
        max_anchor = last_usable_anchor
        if max_anchor > 91:
            n_samples = min(NEG_SAMPLES_PER_REGION, max_anchor - 91)
            sampled_anchors = rng.choice(np.arange(91, max_anchor + 1), size=n_samples, replace=False)
            for anchor in sampled_anchors:
                anchor = int(anchor)
                window = weather_train[anchor - 90: anchor + 1]
                end_date = str(train_g.loc[anchor, DATE_COL])
                delta = last_usable_anchor - anchor
                neg_rows.append(build_anchor_row(window, end_date, region_code))
                neg_meta.append((region, anchor, delta))

        # Candidates to score: every 7 days from 91 to last_usable_anchor
        for anchor in range(91, last_usable_anchor + 1, 7):
            window = weather_train[anchor - 90: anchor + 1]
            end_date = str(train_g.loc[anchor, DATE_COL])
            delta = last_usable_anchor - anchor
            candidate_rows.append(build_anchor_row(window, end_date, region_code))
            candidate_meta.append((region, anchor, delta))

    print(f"  positives: {len(pos_rows):,}; negatives: {len(neg_rows):,}; candidates to score: {len(candidate_rows):,}")

    cols = feature_columns()
    X_pos = np.array(pos_rows, dtype=np.float32)
    X_neg = np.array(neg_rows, dtype=np.float32)
    X_candidates = np.array(candidate_rows, dtype=np.float32)
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    print(f"  X.shape={X.shape}, y mean={y.mean():.4f}")

    print("[step 2] Train binary LightGBM (5-fold CV for OOF AUC) ...")
    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:
        print(f"[error] lightgbm not installed: {e}")
        return 1

    kf = KFold(n_splits=5, shuffle=True, random_state=20260516)
    oof_pred = np.zeros(len(y))
    final_models = []
    fold_aucs = []
    scale_pos_weight = float((y == 0).sum()) / float((y == 1).sum())
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X)):
        m = LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.30,
            scale_pos_weight=scale_pos_weight,
            random_state=20260516 + fold_idx,
            n_jobs=-1,
            verbosity=-1,
        )
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[va_idx])[:, 1]
        oof_pred[va_idx] = p
        auc = roc_auc_score(y[va_idx], p)
        fold_aucs.append(auc)
        final_models.append(m)
        print(f"  fold {fold_idx + 1}/5  AUC={auc:.4f}")
    print(f"[info] OOF AUC: {roc_auc_score(y, oof_pred):.4f} (mean fold {np.mean(fold_aucs):.4f})")

    if roc_auc_score(y, oof_pred) < 0.9:
        print("[warn] AUC < 0.9 — classifier is weak; verify feature construction")
    else:
        print("[info] AUC ≥ 0.9 — train vs test cleanly separable on these features")

    print("[step 3] Score every candidate training anchor (mean over 5 models) ...")
    p_cand = np.mean([m.predict_proba(X_candidates)[:, 1] for m in final_models], axis=0)

    df = pd.DataFrame({
        "region_id": [r for r, _, _ in candidate_meta],
        "anchor_index": [a for _, a, _ in candidate_meta],
        "delta_from_last_usable": [d for _, _, d in candidate_meta],
        "p_test": p_cand,
    })

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "adversarial_scores.parquet"
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        # Fallback to CSV if pyarrow missing
        out_path = out_dir / "adversarial_scores.csv"
        df.to_csv(out_path, index=False)
    print(f"[info] wrote {out_path}  ({len(df):,} rows)")

    print("\n[step 4] Sanity check: distribution of p_test by delta ...")
    for d_bin in [(0, 14), (14, 60), (60, 180), (180, 365), (365, 730), (730, 99999)]:
        mask = (df["delta_from_last_usable"] >= d_bin[0]) & (df["delta_from_last_usable"] < d_bin[1])
        if mask.any():
            sub = df.loc[mask, "p_test"]
            print(f"  delta in [{d_bin[0]:5d}, {d_bin[1]:5d}): n={mask.sum():6d}  p_test mean={sub.mean():.4f}  p50={sub.median():.4f}  p95={sub.quantile(0.95):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
