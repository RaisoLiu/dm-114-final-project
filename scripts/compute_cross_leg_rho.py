#!/usr/bin/env python3
"""Recompute per-region residual correlation ρ between non-GBDT legs and the GBDT anchor.

For each non-GBDT leg L in {lag-2215, CNN, SSL}:
  1. Per region, compute residuals r_anchor = y - anchor_pred and r_L = y - L_pred (over 5 horizons × per-anchor).
  2. Pearson ρ between r_anchor and r_L within each region.
  3. Region-resampled bootstrap (B=1000) of mean ρ across regions, report mean + 95% CI.

This replaces the previously non-recomputable ρ = 0.107 ± 0.027 claim in the abstract.

Output: reports/cross_leg_rho_bootstrap.json
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = REPORTS / "cross_leg_rho_bootstrap.json"

B_BOOT = 1000
SEED = 114


def load_aligned() -> pd.DataFrame:
    oof = pd.read_csv(REPORTS / "oof_tensor.csv")
    oof["region_id"] = oof["region_id"].astype(str)
    gbdt_cols = [c for c in oof.columns if c.startswith("pred_b91_") or c.startswith("pred_pl_")]
    oof["gbdt_anchor"] = oof[gbdt_cols].mean(axis=1)
    cols = ["row_index", "region_id", "horizon", "y_true", "gbdt_anchor"]
    base = oof[cols].copy()

    lag = pd.read_csv(REPORTS / "lag_2215_oof_validation_predictions.csv",
                      usecols=["row_index", "region_id", "horizon", "pred_final_calibrated"])
    lag["region_id"] = lag["region_id"].astype(str)
    lag = lag.rename(columns={"pred_final_calibrated": "lag_2215"})
    base = base.merge(lag, on=["row_index", "region_id", "horizon"], how="inner")

    cnn_files = sorted(REPORTS.glob("deep_cnn_*_validation_predictions.csv"))
    cnn_acc = None
    for path in cnn_files:
        d = pd.read_csv(path, usecols=["row_index", "region_id", "horizon", "pred_final_calibrated"])
        d["region_id"] = d["region_id"].astype(str)
        d = d.rename(columns={"pred_final_calibrated": path.stem})
        cnn_acc = d if cnn_acc is None else cnn_acc.merge(d, on=["row_index", "region_id", "horizon"], how="inner")
    cnn_cols = [c for c in cnn_acc.columns if c not in ("row_index", "region_id", "horizon")]
    cnn_acc["cnn_mean"] = cnn_acc[cnn_cols].mean(axis=1)
    base = base.merge(cnn_acc[["row_index", "region_id", "horizon", "cnn_mean"]],
                      on=["row_index", "region_id", "horizon"], how="inner")

    ssl = pd.read_csv(REPORTS / "track1_ssl_oof_validation_predictions.csv",
                       usecols=["row_index", "region_id", "horizon", "pred_final_calibrated"])
    ssl["region_id"] = ssl["region_id"].astype(str)
    ssl = ssl.rename(columns={"pred_final_calibrated": "ssl"})
    base = base.merge(ssl, on=["row_index", "region_id", "horizon"], how="inner")
    return base


def per_region_rho(df: pd.DataFrame, leg_col: str) -> dict[str, float]:
    """Pearson correlation between r_anchor and r_leg, per region, over all anchor×horizon rows."""
    r_anchor = (df["y_true"] - df["gbdt_anchor"]).to_numpy(dtype=np.float64)
    r_leg = (df["y_true"] - df[leg_col]).to_numpy(dtype=np.float64)
    region = df["region_id"].to_numpy()
    out: dict[str, float] = {}
    for r in pd.unique(region):
        mask = (region == r)
        x = r_anchor[mask]; y = r_leg[mask]
        if x.size < 3 or x.std() < 1e-9 or y.std() < 1e-9:
            continue
        rho = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(rho):
            out[r] = rho
    return out


def bootstrap_mean_rho(rho_per_region: dict[str, float], B: int = B_BOOT, seed: int = SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    keys = list(rho_per_region.keys())
    vals = np.array([rho_per_region[k] for k in keys], dtype=np.float64)
    n = vals.size
    boots = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = vals[idx].mean()
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "n_regions": int(n),
        "B": int(B),
    }


def main() -> None:
    base = load_aligned()
    print(f"[rho] aligned rows: {len(base)}; regions: {base.region_id.nunique()}", flush=True)
    result: dict[str, dict] = {}
    for leg in ["lag_2215", "cnn_mean", "ssl"]:
        per = per_region_rho(base, leg)
        boot = bootstrap_mean_rho(per)
        result[leg] = boot
        print(
            f"[rho] {leg}: mean ρ = {boot['mean']:.4f}  std = {boot['std']:.4f}  "
            f"95% CI = [{boot['ci95_low']:.4f}, {boot['ci95_high']:.4f}]  n={boot['n_regions']}",
            flush=True,
        )
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[rho] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
