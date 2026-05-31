#!/usr/bin/env python3
"""Plan v12 Track B1 — Build per-region climate fingerprints.

For each region, compute a 28-D climate signature:
- 12 monthly mean temperatures (°C)
- 12 monthly mean precipitation (mm/day)
- annual mean T, annual P total, surface_pressure (proxy for elevation), wind speed

Save to reports/climate_fingerprints.csv for downstream matching (B2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
REPORTS = PROJECT_ROOT / "reports"


def main() -> int:
    print("Loading train.csv ...")
    cols = ["region_id", "date", "tmp", "tmp_max", "tmp_min", "prec", "humidity", "surf_pre", "wind", "dp_tmp"]
    t = pd.read_csv(DATA / "train.csv", usecols=cols)
    t["region_id"] = t["region_id"].astype(str)
    t["year"] = t["date"].str.slice(0, 4).astype(int)
    # Robust month parsing: split on '-' and take 2nd field (handles 4 or 5-digit years)
    t["month"] = t["date"].str.split("-").str[1].astype(int)
    print(f"  {len(t):,} rows, {t['region_id'].nunique()} regions")

    print("Computing monthly climatology per region ...")
    # Monthly aggregates per region
    monthly = t.groupby(["region_id", "month"]).agg(
        tmp_mean=("tmp", "mean"),
        prec_mean=("prec", "mean"),
        prec_total=("prec", "sum"),
        humidity_mean=("humidity", "mean"),
        dp_tmp_mean=("dp_tmp", "mean"),
        n_days=("tmp", "count"),
    ).reset_index()
    # Pivot to one row per region with 12 columns each for T and P
    tmp_pivot = monthly.pivot(index="region_id", columns="month", values="tmp_mean").add_prefix("tmp_m")
    prec_pivot = monthly.pivot(index="region_id", columns="month", values="prec_mean").add_prefix("prec_m")
    humid_pivot = monthly.pivot(index="region_id", columns="month", values="humidity_mean").add_prefix("humid_m")
    fp = pd.concat([tmp_pivot, prec_pivot, humid_pivot], axis=1)

    # Annual aggregates
    annual = t.groupby("region_id").agg(
        tmp_mean_annual=("tmp", "mean"),
        tmp_min_all=("tmp", "min"),
        tmp_max_all=("tmp", "max"),
        prec_mean_annual=("prec", "mean"),
        humidity_mean_annual=("humidity", "mean"),
        surf_pre_mean=("surf_pre", "mean"),
        wind_mean=("wind", "mean"),
        dp_tmp_mean=("dp_tmp", "mean"),
        n_total=("tmp", "count"),
    )
    annual["tmp_amplitude"] = annual["tmp_max_all"] - annual["tmp_min_all"]
    annual["prec_annual_mm"] = annual["prec_mean_annual"] * 365.25
    # Rough elevation from surface pressure (sea-level pressure ~ 101.325 kPa, scale height ~ 8400m)
    # P(h) ≈ P0 * exp(-h / 8400), so h ≈ -8400 * ln(P / 101.325)
    annual["elevation_m"] = (-8400.0 * np.log(annual["surf_pre_mean"] / 101.325)).round().astype(int)
    # Hemisphere indicator: warmest month is Jan-Feb (S hemisphere) or Jul-Aug (N hemisphere)?
    warmest_month = monthly.loc[monthly.groupby("region_id")["tmp_mean"].idxmax(), ["region_id", "month"]].set_index("region_id")
    warmest_month.columns = ["warmest_month"]
    annual = annual.merge(warmest_month, left_index=True, right_index=True, how="left")
    annual["hemisphere"] = np.where(annual["warmest_month"].between(4, 9), "N", "S")

    fp = fp.merge(annual, left_index=True, right_index=True)

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "climate_fingerprints.csv"
    fp.to_csv(out)
    print(f"[info] wrote {out} ({len(fp)} regions, {len(fp.columns)} features)")

    # Summary
    print()
    print("Climate diversity summary:")
    print(fp[["tmp_mean_annual", "tmp_amplitude", "prec_annual_mm", "elevation_m", "hemisphere", "warmest_month"]].describe(include="all"))
    print()
    print("Hemisphere distribution:")
    print(fp["hemisphere"].value_counts())
    return 0


if __name__ == "__main__":
    sys.exit(main())
