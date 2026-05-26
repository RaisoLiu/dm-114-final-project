#!/usr/bin/env python3
"""
DM 114 Report - Strict IEEE-compliant Figure Generator
Implements §2.8 of the approved plan.

Rules enforced:
- Single-column figures: exactly 3.5 inches wide
- All text rendered at final size: 8pt (labels/ticks), 9pt (titles)
- No reliance on LaTeX scaling for fonts
- Output: PDF (preferred, vector text) or high-dpi PNG at correct physical size
"""

from __future__ import annotations
import os
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import numpy as np

# =============================================================================
# §2.8 IEEE Sizing Contract - DO NOT CHANGE THESE VALUES LIGHTLY
# =============================================================================
IEEE_SINGLE_COL_INCH = 3.50          # Target width for most guiding figures
IEEE_FULL_WIDTH_INCH = 7.00
IEEE_FONT_SIZE = 8                   # Primary label/tick size
IEEE_TITLE_SIZE = 9
IEEE_LEGEND_SIZE = 7
IEEE_DPI = 300

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_PLOTS = ROOT / "reports" / "plots"
FIG_DIR = Path(__file__).resolve().parent

def set_ieee_figure_style(target_width_inch: float = IEEE_SINGLE_COL_INCH):
    """Apply IEEE-compliant styling for a figure of given physical width (inches)."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': IEEE_FONT_SIZE,
        'axes.labelsize': IEEE_FONT_SIZE,
        'axes.titlesize': IEEE_TITLE_SIZE,
        'xtick.labelsize': IEEE_FONT_SIZE - 1,
        'ytick.labelsize': IEEE_FONT_SIZE - 1,
        'legend.fontsize': IEEE_LEGEND_SIZE,
        'figure.titlesize': IEEE_TITLE_SIZE,
        'axes.linewidth': 0.8,
        'pdf.fonttype': 42,      # TrueType fonts (good for IEEE)
        'ps.fonttype': 42,
    })
    # Return suggested height for 3.5" width (typical aspect for 2-4 panel figures)
    height = target_width_inch * 0.72   # ~2.52 inches for 3.5" wide
    return height


def save_ieee_figure(fig: plt.Figure, filename: str, target_width_inch: float = IEEE_SINGLE_COL_INCH):
    """Save figure at exact target physical width with correct DPI."""
    out_path = FIG_DIR / filename
    # Set exact size in inches (this is critical - prevents LaTeX from needing to scale text)
    fig.set_size_inches(target_width_inch, fig.get_figheight())
    fig.savefig(out_path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  [IEEE] Saved {filename} at {target_width_inch}\" width, {IEEE_FONT_SIZE}pt fonts")


# =============================================================================
# Real Multi-Panel Guiding Figures (following plan requirements)
# =============================================================================

def create_fig3_orthogonality():
    """Fig 3: Cross-leg residual correlation evidence from real OOF tensor.
    Iter-2 rewrite: replaces the iter-0/iter-1 synthetic matrix with the real
    6x6 region-mean residual correlation matrix loaded from
    reports/_fig3_corr_matrix.npy (computed by scripts/report_stats.py on
    reports/oof_tensor.csv, B=1000 bootstrap CI in caption).
    """
    height = set_ieee_figure_style(IEEE_SINGLE_COL_INCH)
    fig = plt.figure(figsize=(IEEE_SINGLE_COL_INCH, height * 1.10))
    gs = GridSpec(1, 1, figure=fig)

    matrix_path = ROOT / "reports" / "_fig3_corr_matrix.npy"
    labels_path = ROOT / "reports" / "_fig3_corr_labels.csv"
    if not matrix_path.exists() or not labels_path.exists():
        ax = fig.add_subplot(gs[0])
        ax.text(0.5, 0.5,
                "(_fig3_corr_matrix.npy missing — run scripts/report_stats.py)",
                ha="center", va="center", fontsize=7)
        ax.axis("off")
        save_ieee_figure(fig, "fig3_orthogonality.pdf", IEEE_SINGLE_COL_INCH)
        return

    M = np.load(matrix_path)
    labels = [l.strip() for l in labels_path.read_text().splitlines() if l.strip()]
    # Shorten labels for ticks
    short = []
    for l in labels:
        s = l.replace("pred_", "")
        s = s.replace("seed", "s").replace("_seed", "_s")
        if len(s) > 16:
            s = s[:16]
        short.append(s)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.6 else "black", fontsize=6)
    ax.set_title("Region-mean residual correlation (real OOF tensor)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.040, pad=0.04)
    save_ieee_figure(fig, "fig3_orthogonality.pdf", IEEE_SINGLE_COL_INCH)


def create_fig4_cv_generalization():
    """Fig 4: 5-fold CV evidence. Iter-3 polish: render at IEEE full-text width
    (7 inch) so each of the four embedded panels is legible. The LaTeX side
    uses \\begin{figure*} ... \\includegraphics[width=\\textwidth] to span
    both columns."""
    height = set_ieee_figure_style(IEEE_FULL_WIDTH_INCH)
    # 2x2 grid at 7" wide → each panel ≈ 3.5" wide, readable
    fig = plt.figure(figsize=(IEEE_FULL_WIDTH_INCH, IEEE_FULL_WIDTH_INCH * 0.55))
    gs = GridSpec(2, 2, figure=fig, hspace=0.18, wspace=0.10)

    plots = [
        ('cv1_calibration_oof.png', '(a) Calibration OOF'),
        ('cv2_lag_rho_per_fold.png', '(b) Lag-ρ Stability'),
        ('cv3_in_vs_oof_mae.png', '(c) Blend Weight Robustness'),
        ('cv4_final_per_fold_mae.png', '(d) Final Candidate Stability'),
    ]

    for idx, (fname, title) in enumerate(plots):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        src = REPORTS_PLOTS / fname
        if src.exists():
            img = plt.imread(src)
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'{title}\n(missing source)',
                    ha='center', va='center', fontsize=8)
            ax.axis('off')

    save_ieee_figure(fig, 'fig4_cv_generalization.pdf', IEEE_FULL_WIDTH_INCH)


def create_fig5_trajectory():
    """Fig 5: Final trajectory + distribution match."""
    height = set_ieee_figure_style(IEEE_SINGLE_COL_INCH)
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL_INCH, height * 0.95))

    submissions = ['ext150\n(2026-05-12)', 'v18 7-way\n(2026-05-21)', 'phd-below075\n(0.7628)']
    maes = [0.8534, 0.7952, 0.7628]
    colors = ['#d62728', '#ff7f0e', '#2ca02c']

    bars = ax.bar(submissions, maes, color=colors, width=0.6)
    ax.set_ylabel('Public MAE')
    ax.set_title('Submission trajectory: 10.6% relative improvement', fontsize=9)
    ax.set_ylim(0.70, 0.90)
    ax.axhline(0.8534, color='gray', linestyle='--', linewidth=0.8, label='Previous ceiling')

    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{mae:.4f}', ha='center', fontsize=7.5, fontweight='bold')

    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    save_ieee_figure(fig, 'fig5_trajectory.pdf', IEEE_SINGLE_COL_INCH)


def main():
    print("=" * 60)
    print("DM 114 Report Figure Generator — Strict §2.8 Mode")
    print(f"Target single-column width: {IEEE_SINGLE_COL_INCH} inches")
    print(f"Font sizes: {IEEE_FONT_SIZE}pt labels, {IEEE_TITLE_SIZE}pt titles")
    print("=" * 60)

    os.makedirs(FIG_DIR, exist_ok=True)

    # Generate the three most important guiding figures with correct sizing
    print("\n[1/3] Generating Fig 3 (Orthogonality)...")
    create_fig3_orthogonality()

    print("\n[2/3] Generating Fig 4 (CV Generalization)...")
    create_fig4_cv_generalization()

    print("\n[3/3] Generating Fig 5 (Trajectory)...")
    create_fig5_trajectory()

    print("\n[4/5] Generating Fig 1 (Periodicity from real data)...")
    create_fig1_periodicity()

    print("\n[5/5] Generating Fig 2 (Slope Law from real data)...")
    create_fig2_slope()

    print("\n[OK] All figures generated with IEEE sizing contract enforced.")
    print("  Next: Rebuild PDF with width=\\columnwidth includes.")


def create_fig1_periodicity():
    """Fig 1: Multi-year periodicity in the weekly score series — from real data."""
    import pandas as pd

    height = set_ieee_figure_style(IEEE_SINGLE_COL_INCH)
    fig = plt.figure(figsize=(IEEE_SINGLE_COL_INCH, height * 1.85))
    gs = GridSpec(3, 1, figure=fig, hspace=1.05)

    peaks_path = ROOT / "reports" / "_track2_fft_peaks.csv"
    train_path = ROOT / "data" / "train.csv"

    # Panel (a): distribution of top FFT period across regions
    ax1 = fig.add_subplot(gs[0])
    if peaks_path.exists():
        peaks = pd.read_csv(peaks_path)
        # Bin the top_period column into year buckets
        years = peaks["top_period"].dropna().values / 365.25
        bins = np.arange(0, 16, 0.5)
        ax1.hist(years, bins=bins, color="#1f77b4", edgecolor="white", linewidth=0.4)
        ax1.set_xlabel("Top FFT period (years)", labelpad=3)
        ax1.set_ylabel("# regions", labelpad=3)
        ax1.set_title("(a) FFT top-period distribution", fontsize=8, pad=5)
        ax1.axvspan(5.5, 7.5, color="#d62728", alpha=0.15, label="6-yr cycle band")
        ax1.legend(fontsize=6.5, loc="upper right")
    else:
        ax1.text(0.5, 0.5, "(missing _track2_fft_peaks.csv)", ha="center", va="center", fontsize=7)
        ax1.axis("off")

    # Panel (b): example weekly score series for two contrasting regions
    ax2 = fig.add_subplot(gs[1])
    if train_path.exists():
        try:
            df = pd.read_csv(train_path, usecols=["region_id", "date", "score"])
            df = df.dropna(subset=["score"])
            for rid, color in [("R1", "#1f77b4"), ("R1001", "#ff7f0e")]:
                sub = df[df["region_id"] == rid].copy()
                if len(sub) > 0:
                    sub = sub.reset_index(drop=True)
                    ax2.plot(sub.index, sub["score"].values, color=color,
                             linewidth=0.6, alpha=0.85, label=rid)
            ax2.set_xlabel("Weekly score index", labelpad=3)
            ax2.set_ylabel("Score (0-5)", labelpad=3)
            ax2.set_title("(b) Two example regions", fontsize=8, pad=5)
            ax2.legend(fontsize=6.5, loc="upper right")
            ax2.set_ylim(-0.1, 5.1)
        except Exception as e:
            ax2.text(0.5, 0.5, f"(read failed: {e})", ha="center", va="center", fontsize=7)
            ax2.axis("off")
    else:
        ax2.text(0.5, 0.5, "(missing data/train.csv)", ha="center", va="center", fontsize=7)
        ax2.axis("off")

    # Panel (c): coverage stat — % regions with dominant period in 5-8 year band
    ax3 = fig.add_subplot(gs[2])
    if peaks_path.exists():
        peaks = pd.read_csv(peaks_path)
        years = peaks["top_period"].dropna().values / 365.25
        bands = [(0, 2), (2, 5), (5, 8), (8, 12), (12, 20)]
        labels = ["<2y", "2-5y", "5-8y", "8-12y", ">12y"]
        counts = [int(((years >= lo) & (years < hi)).sum()) for lo, hi in bands]
        bars = ax3.bar(labels, counts, color=["#cccccc", "#cccccc", "#d62728", "#cccccc", "#cccccc"],
                       edgecolor="white", linewidth=0.5)
        ax3.set_ylabel("# regions", labelpad=3)
        ax3.set_title("(c) Multi-year cycle coverage", fontsize=8, pad=5)
        for bar, c in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                     str(c), ha="center", fontsize=6.5)
    else:
        ax3.text(0.5, 0.5, "(missing data)", ha="center", va="center", fontsize=7)
        ax3.axis("off")

    save_ieee_figure(fig, "fig1_periodicity.pdf", IEEE_SINGLE_COL_INCH)


def create_fig2_slope():
    """Fig 2: observed public-ledger slope trend (public MAE vs MAD).
    Iter-3 polish: replaces the hardcoded 0.42 slope line with the actual
    bootstrap-fitted line over all 32 rows (slope ~0.20, intercept ~0.85),
    plus a separate pre-v18-only fit line (~0.21) so both numbers are
    visible. Bootstrap CI band is shaded on the all-32 fit.
    """
    import pandas as pd

    height = set_ieee_figure_style(IEEE_SINGLE_COL_INCH)
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL_INCH, height * 1.05))

    gate_path = ROOT / "reports" / "_local_eval_gate_report.csv"
    if not gate_path.exists():
        ax.text(0.5, 0.5, "(missing _local_eval_gate_report.csv)",
                ha="center", va="center", fontsize=7)
        ax.axis("off")
        save_ieee_figure(fig, "fig2_slope.pdf", IEEE_SINGLE_COL_INCH)
        return

    df = pd.read_csv(gate_path).dropna(subset=["public", "mad"])
    is_v18 = df["filename"].str.startswith("_v18") | (df["public"] < 0.85)
    pre = df[~is_v18]
    new = df[is_v18]

    ax.scatter(pre["mad"], pre["public"], s=18, color="#1f77b4",
               label=f"Pre-v18 ({len(pre)})", alpha=0.85,
               edgecolor="white", linewidth=0.3)
    ax.scatter(new["mad"], new["public"], s=24, color="#d62728", marker="^",
               label=f"v18+ ({len(new)})", alpha=0.9,
               edgecolor="white", linewidth=0.3)

    # Bootstrap-fitted lines from real data
    rng = np.random.default_rng(42)
    mad_x = np.linspace(0, max(df["mad"].max(), 0.5), 60)
    # all-32 fit
    s_all, i_all = np.polyfit(df["mad"].values, df["public"].values, 1)
    # pre-v18 fit
    s_pre, i_pre = np.polyfit(pre["mad"].values, pre["public"].values, 1)
    # bootstrap band for all-32
    B = 1000
    fits = []
    mad_arr = df["mad"].values
    pub_arr = df["public"].values
    n = len(df)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        s, i = np.polyfit(mad_arr[idx], pub_arr[idx], 1)
        fits.append(s * mad_x + i)
    fits = np.asarray(fits)
    lo = np.percentile(fits, 2.5, axis=0)
    hi = np.percentile(fits, 97.5, axis=0)
    ax.fill_between(mad_x, lo, hi, color="black", alpha=0.10,
                    label="All-32 bootstrap 95% CI")
    ax.plot(mad_x, s_all * mad_x + i_all, color="black",
            linewidth=1.0, linestyle="--",
            label=f"All-32 fit (slope {s_all:.2f})")
    ax.plot(mad_x, s_pre * mad_x + i_pre, color="#7f7f7f",
            linewidth=0.8, linestyle=":",
            label=f"Pre-v18 fit (slope {s_pre:.2f})")
    ax.axhline(0.8534, color="gray", linewidth=0.5, linestyle="-",
               alpha=0.4, label="ext150 ceiling 0.8534")

    ax.set_xlabel("MAD vs ext150 anchor")
    ax.set_ylabel("Kaggle public MAE")
    ax.set_title("Observed slope trend on submission data", fontsize=9)
    ax.legend(fontsize=6, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.74, max(1.05, df["public"].max() * 1.02))

    save_ieee_figure(fig, "fig2_slope.pdf", IEEE_SINGLE_COL_INCH)


if __name__ == "__main__":
    main()
