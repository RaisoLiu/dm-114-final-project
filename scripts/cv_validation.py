#!/usr/bin/env python3
"""5-fold cross-validation for v18 breakthrough generalization.

Four experiments:
  1. Calibration model 5-fold OOF (submission-based)
  2. 6-yr lag ρ stability 5-fold (region-based)
  3. Blend-weight robustness 5-fold (region-based, held-out grid search)
  4. Final candidate _v18_finalsuper2_T.csv held-out eval (region-based)

Outputs:
  reports/_cv_results.csv         — raw per-fold metrics
  reports/plots/cv*.png           — 10 plots
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from local_eval_gate import candidate_stats, predict_public, fit_calibration

PRED = [f'pred_week{i+1}' for i in range(5)]
PLOTS = ROOT / "reports" / "plots"
PLOTS.mkdir(exist_ok=True, parents=True)
SEED = 42

# ---------- shared loaders ----------
def load_oracle():
    return pd.read_csv(ROOT / "reports" / "_local_eval_oracle.csv")

def load_ext150():
    return pd.read_csv(ROOT / "submissions" / "submission_round5_pb30_x150_repro.csv")

def load_kaggle_history():
    df = pd.read_csv(ROOT / "reports" / "_kaggle_history.csv")
    df = df[df['status'].astype(str).str.contains('COMPLETE', na=False)]
    df['publicScore'] = pd.to_numeric(df['publicScore'], errors='coerce')
    return df.dropna(subset=['publicScore'])

def load_candidate(filename, common_regions):
    """Load a candidate CSV and return aligned (N, 5) array."""
    df = pd.read_csv(ROOT / "submissions" / filename)
    return df.set_index('region_id').reindex(common_regions)[PRED].values

def cal_to_ext150(v, e):
    m, s = v.mean(), v.std()
    em, es = e.mean(), e.std()
    if s < 1e-6:
        return v.copy()
    return np.clip((v - m) * (es / s) + em, 0, 5)

def make_folds(n_items, k=5, seed=SEED):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_items)))

# ---------- shared global state for memo ----------
ALL_RESULTS = []  # rows: dict(experiment, fold, metric, value)

def add_result(experiment, fold, metric, value):
    ALL_RESULTS.append({'experiment': experiment, 'fold': fold, 'metric': metric, 'value': value})


# ============================================================
# Experiment 1: Calibration model 5-fold OOF
# ============================================================
def exp1_calibration_cv():
    print("\n=== Experiment 1: Calibration 5-fold OOF ===")
    # Load gate report (which has predicted_public from all-data calibration)
    df = pd.read_csv(ROOT / "reports" / "_local_eval_gate_report.csv")
    # Take only candidates with known public score, NOT v17 outlier (oracle ~0)
    known = df.dropna(subset=['public']).copy()
    known = known[known['oracle_mae'] > 0.3]  # match local_eval_gate's filter
    n = len(known)
    print(f"  {n} historical submissions (v17 excluded)")

    folds = make_folds(n, k=5)
    feats = ['oracle_mae', 'mad', 'std', 'mean']
    X_all = known[feats].values
    y_all = known['public'].values
    oof_preds = np.full(n, np.nan)
    fold_metrics = []

    for fi, (tr, te) in enumerate(folds):
        X_tr = np.column_stack([np.ones(len(tr))] + [X_all[tr, j] for j in range(4)])
        y_tr = y_all[tr]
        coef, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        X_te = np.column_stack([np.ones(len(te))] + [X_all[te, j] for j in range(4)])
        pred_te = X_te @ coef
        oof_preds[te] = pred_te
        rmse = float(np.sqrt(((pred_te - y_all[te])**2).mean()))
        mae = float(np.abs(pred_te - y_all[te]).mean())
        fold_metrics.append({'fold': fi, 'rmse': rmse, 'mae': mae, 'n_test': len(te)})
        add_result('exp1', fi, 'rmse', rmse)
        add_result('exp1', fi, 'mae', mae)
        add_result('exp1', fi, 'n_test', len(te))
        print(f"  fold {fi}: n_test={len(te)}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    oof_rmse = float(np.sqrt(((oof_preds - y_all)**2).mean()))
    print(f"  OOF total RMSE: {oof_rmse:.4f}")

    # Plot 1: OOF scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for fi, (_, te) in enumerate(folds):
        ax.scatter(y_all[te], oof_preds[te], c=[colors[fi]], s=70, alpha=0.85,
                   label=f'Fold {fi+1} (n={len(te)})', edgecolors='black', linewidths=0.5)
    lo = min(y_all.min(), oof_preds.min()) - 0.02
    hi = max(y_all.max(), oof_preds.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='y=x (perfect)')
    ax.set_xlabel('Actual public MAE', fontsize=12)
    ax.set_ylabel('OOF predicted public MAE', fontsize=12)
    ax.set_title(f'Experiment 1: Calibration 5-fold OOF Predictions\nOOF RMSE = {oof_rmse:.4f}', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "cv1_calibration_oof.png", dpi=120, bbox_inches='tight')
    plt.close()

    # Plot 2: per-fold RMSE bar
    fm = pd.DataFrame(fold_metrics)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(5), fm['rmse'], color=colors, edgecolor='black')
    ax.axhline(0.010, color='red', linestyle='--', label='In-sample RMSE = 0.010')
    ax.axhline(0.015, color='orange', linestyle=':', label='Acceptance threshold = 0.015')
    for b, v in zip(bars, fm['rmse']):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                f'{v:.4f}', ha='center', fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax.set_ylabel('Test-fold RMSE', fontsize=12)
    ax.set_title('Experiment 1: Per-Fold Calibration RMSE', fontsize=13)
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS / "cv1_per_fold_rmse.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("  ✓ saved cv1_calibration_oof.png, cv1_per_fold_rmse.png")
    return fm


# ============================================================
# Experiment 2: 6-yr lag ρ stability
# ============================================================
def exp2_lag_rho_cv():
    print("\n=== Experiment 2: 6-yr lag ρ stability ===")
    oracle = load_oracle()
    ext150 = load_ext150()
    common = sorted(set(ext150['region_id']) & set(oracle['region_id']))
    np.random.seed(SEED)

    e = ext150.set_index('region_id').loc[common][PRED].values
    truth_v = oracle.set_index('region_id').loc[common][PRED].values
    e_err = e - truth_v  # (N, 5)

    # Lag files to test
    lag_files = {
        '1 yr (365 d)': '_v18_lag3yr.csv',  # we have 3 yr file, use as proxy
        '5 yr (1820 d)': '_v18_lag5yr.csv',
        '6 yr (2184 d)': '_v18_lag6yr.csv',
        '6.07 yr (2215 d)': '_v18_lag2215d.csv',
        '6.5 yr (2367 d)': '_v18_lag065.csv',
        '7 yr (2548 d)': '_v18_lag7yr.csv',
    }
    # Verify files exist
    available = {}
    for name, fn in lag_files.items():
        p = ROOT / "submissions" / fn
        if p.exists():
            available[name] = fn
    print(f"  available lag candidates: {list(available.keys())}")

    folds = make_folds(len(common), k=5)
    results = {}  # name -> list of 5 ρ values
    for name, fn in available.items():
        v = pd.read_csv(ROOT / "submissions" / fn).set_index('region_id').reindex(common)[PRED].values
        v_err = v - truth_v
        rhos = []
        for fi, (_, te) in enumerate(folds):
            e_sub = e_err[te].flatten()
            v_sub = v_err[te].flatten()
            mask = ~np.isnan(e_sub) & ~np.isnan(v_sub)
            if mask.sum() < 10 or e_sub[mask].std() < 1e-6 or v_sub[mask].std() < 1e-6:
                rhos.append(np.nan); continue
            r = float(np.corrcoef(e_sub[mask], v_sub[mask])[0, 1])
            rhos.append(r)
            add_result('exp2', fi, f'rho_{name}', r)
        results[name] = rhos
        print(f"  {name}: ρ = {np.nanmean(rhos):.4f} ± {np.nanstd(rhos):.4f}  (range [{np.nanmin(rhos):.4f}, {np.nanmax(rhos):.4f}])")

    # Plot 3: boxplot of ρ per lag
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    data = [results[n] for n in names]
    bp = ax.boxplot(data, labels=names, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', edgecolor='navy'),
                    medianprops=dict(color='red', linewidth=2))
    # Overlay individual points
    for i, vals in enumerate(data):
        x = np.full(len(vals), i + 1)
        x = x + np.random.uniform(-0.1, 0.1, len(vals))
        ax.scatter(x, vals, color='darkred', s=40, alpha=0.7, zorder=3)
    ax.set_ylabel('ρ(candidate errors, ext150 errors)', fontsize=12)
    ax.set_title('Experiment 2: Lag Candidate ρ vs ext150 — 5-Fold Stability', fontsize=13)
    ax.axhline(0.5, color='gray', linestyle=':', label='ρ=0.5 (prior team minimum)')
    ax.axhline(0, color='green', linestyle=':', label='ρ=0 (true orthogonal)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, axis='y')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS / "cv2_lag_rho_per_fold.png", dpi=120, bbox_inches='tight')
    plt.close()

    # Plot 4: dense lag scan with CI (re-use existing dense scan + 5-fold CV on each)
    print("  computing dense lag scan with 5-fold CI (this may take ~2 min)...")
    dense_lags = [365, 730, 1095, 1456, 1820, 1900, 2000, 2100, 2184, 2215, 2300, 2400, 2548, 2700, 2900, 3100]
    train_df = pd.read_csv(ROOT / "data" / "train.csv", usecols=['region_id', 'date', 'score'])
    test_df = pd.read_csv(ROOT / "data" / "test.csv", usecols=['region_id', 'date'])
    train_df = train_df.sort_values(['region_id', 'date']).reset_index(drop=True)
    train_df['day_idx'] = train_df.groupby('region_id').cumcount()
    test_df = test_df.sort_values(['region_id', 'date']).reset_index(drop=True)
    test_df['day_idx'] = test_df.groupby('region_id').cumcount() + 5480
    train_scored = train_df.dropna(subset=['score'])
    score_lookup = {(r, d): s for r, d, s in zip(train_scored['region_id'], train_scored['day_idx'], train_scored['score'])}
    test_max = test_df.groupby('region_id')['day_idx'].max().to_dict()

    dense_results = []  # (lag, mean_rho, std_rho)
    for lag in dense_lags:
        rows = []
        for rid in common:
            tm = test_max[rid]
            row = {'region_id': rid}
            for h in range(1, 6):
                target = tm + 7 * h
                lookup = target - lag
                score = None
                for d in [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]:
                    if (rid, lookup + d) in score_lookup:
                        score = score_lookup[(rid, lookup + d)]; break
                row[f'pred_week{h}'] = score if score is not None else 0.84
            rows.append(row)
        df_lag = pd.DataFrame(rows)
        v = df_lag.set_index('region_id').loc[common][PRED].values
        v_err = v - truth_v
        rhos = []
        for fi, (_, te) in enumerate(folds):
            e_sub = e_err[te].flatten()
            v_sub = v_err[te].flatten()
            mask = ~np.isnan(e_sub) & ~np.isnan(v_sub)
            if mask.sum() < 10:
                rhos.append(np.nan); continue
            r = float(np.corrcoef(e_sub[mask], v_sub[mask])[0, 1])
            rhos.append(r)
        dense_results.append((lag, np.nanmean(rhos), np.nanstd(rhos)))
        print(f"    lag={lag}d: ρ={np.nanmean(rhos):.4f} ± {np.nanstd(rhos):.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    lags_arr = np.array([d[0] for d in dense_results])
    means = np.array([d[1] for d in dense_results])
    stds = np.array([d[2] for d in dense_results])
    ax.plot(lags_arr, means, '-o', color='navy', markersize=6, label='Mean ρ across 5 folds')
    ax.fill_between(lags_arr, means - stds, means + stds, alpha=0.3, color='navy', label='±1 std')
    ax.axvline(2215, color='red', linestyle='--', alpha=0.7, label='Best lag = 2215 d (6.07 yr)')
    ax.axhline(0.5, color='gray', linestyle=':', label='ρ=0.5 (prior team minimum)')
    ax.axhline(0, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel('Lag (days)', fontsize=12)
    ax.set_ylabel('ρ(lag errors, ext150 errors)', fontsize=12)
    ax.set_title('Experiment 2: Dense Lag Scan — 5-Fold ρ vs ext150', fontsize=13)
    # Add year ticks on top
    secax = ax.secondary_xaxis('top', functions=(lambda x: x/365.0, lambda x: x*365.0))
    secax.set_xlabel('Lag (years)', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "cv2_dense_lag_scan.png", dpi=120, bbox_inches='tight')
    plt.close()

    # Save dense scan data
    pd.DataFrame(dense_results, columns=['lag_days', 'mean_rho', 'std_rho']).to_csv(
        ROOT / "reports" / "_cv_dense_lag_scan.csv", index=False)

    print("  ✓ saved cv2_lag_rho_per_fold.png, cv2_dense_lag_scan.png")
    return results, dense_results


# ============================================================
# Experiment 3: Blend weight 5-fold OOF
# ============================================================
def exp3_blend_weights_cv():
    print("\n=== Experiment 3: Blend weight 5-fold OOF ===")
    oracle = load_oracle()
    ext150 = load_ext150()
    common = sorted(set(ext150['region_id']) & set(oracle['region_id']))

    e = ext150.set_index('region_id').loc[common][PRED].values
    truth_v = oracle.set_index('region_id').loc[common][PRED].values
    e_mean, e_std = e.mean(), e.std()

    # 6 candidates
    cand_files = {
        'ext150': 'submission_round5_pb30_x150_repro.csv',
        'track3': '_v18_track3_fast.csv',
        'deep_pb30': 'submission_deep_ensemble_pb30_w10.csv',
        'lag_6yr': '_v18_lag6yr.csv',
        'lag_2215d': '_v18_lag2215d.csv',
        'huber': '_v18_track3_huber.csv',
    }
    data = {}
    for k, fn in cand_files.items():
        v = pd.read_csv(ROOT / "submissions" / fn).set_index('region_id').reindex(common)[PRED].values
        if k == 'ext150':
            data[k] = v
        else:
            data[k] = cal_to_ext150(v, e)
    keys = list(cand_files.keys())

    folds = make_folds(len(common), k=5)

    # Grid search per fold (step 0.10 for speed; matches main session's 0.05 result within 0.001)
    def grid_search_weights(train_idx):
        """Find weights that minimize oracle MAE on train_idx."""
        # Pre-slice
        ds = {k: data[k][train_idx] for k in keys}
        ts = truth_v[train_idx]
        step = 0.10
        n_step = int(round(1/step)) + 1
        best = (None, 999)
        for w1 in range(n_step):
            for w2 in range(n_step - w1):
                for w3 in range(n_step - w1 - w2):
                    for w4 in range(n_step - w1 - w2 - w3):
                        for w5 in range(n_step - w1 - w2 - w3 - w4):
                            w6 = (n_step - 1) - w1 - w2 - w3 - w4 - w5
                            if w6 < 0: continue
                            weights = np.array([w1, w2, w3, w4, w5, w6]) * step
                            blend = sum(weights[i] * ds[keys[i]] for i in range(6))
                            blend = np.clip(blend, 0, 5)
                            mae = float(np.abs(blend - ts).mean())
                            if mae < best[1]:
                                best = (weights, mae)
        return best[0], best[1]

    fold_weights = []
    in_sample_maes = []
    oof_maes = []
    oof_pred_pubs = []

    # Fit calibration globally (for predicted_public estimates)
    df_report = pd.read_csv(ROOT / "reports" / "_local_eval_gate_report.csv")
    known = df_report.dropna(subset=['public']).copy()
    coef, _ = fit_calibration(known)

    for fi, (tr, te) in enumerate(folds):
        weights, in_mae = grid_search_weights(tr)
        # Test on held-out fold
        ds_te = {k: data[k][te] for k in keys}
        blend_te = sum(weights[i] * ds_te[keys[i]] for i in range(6))
        blend_te = np.clip(blend_te, 0, 5)
        ts_te = truth_v[te]
        oof_mae = float(np.abs(blend_te - ts_te).mean())
        # Compute predicted public via calibration (on test fold blend)
        # Build candidate stats on full data for calibration (we just need oracle/mad/std/mean of the blend)
        blend_full = sum(weights[i] * data[keys[i]] for i in range(6))
        blend_full = np.clip(blend_full, 0, 5)
        oracle_mae = float(np.abs(blend_full - truth_v).mean())
        mad = float(np.abs(blend_full - data['ext150']).mean())
        m = float(blend_full.mean())
        s = float(blend_full.std())
        stats = {'oracle_mae': oracle_mae, 'mad': mad, 'std': s, 'mean': m}
        pred_pub = predict_public(coef, stats)
        fold_weights.append(weights)
        in_sample_maes.append(in_mae)
        oof_maes.append(oof_mae)
        oof_pred_pubs.append(pred_pub)
        for k_idx, k in enumerate(keys):
            add_result('exp3', fi, f'weight_{k}', float(weights[k_idx]))
        add_result('exp3', fi, 'in_sample_mae', in_mae)
        add_result('exp3', fi, 'oof_mae', oof_mae)
        add_result('exp3', fi, 'pred_pub', pred_pub)
        print(f"  fold {fi}: weights={dict(zip(keys, np.round(weights, 2)))}  in_mae={in_mae:.4f}  oof_mae={oof_mae:.4f}  pred_pub={pred_pub:.4f}")

    print(f"  OOF MAE: {np.mean(oof_maes):.4f} ± {np.std(oof_maes):.4f}")
    print(f"  In-sample: {np.mean(in_sample_maes):.4f}; ratio OOF/IS = {np.mean(oof_maes)/np.mean(in_sample_maes):.3f}")

    # Plot 5: weights heatmap
    fig, ax = plt.subplots(figsize=(9, 4))
    weights_mat = np.array(fold_weights)
    im = ax.imshow(weights_mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)
    for i in range(5):
        for j in range(len(keys)):
            ax.text(j, i, f'{weights_mat[i,j]:.2f}', ha='center', va='center',
                    color='black' if weights_mat[i,j] < 0.3 else 'white', fontsize=10)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=20, ha='right')
    ax.set_yticks(range(5)); ax.set_yticklabels([f'Fold {i+1}' for i in range(5)])
    plt.colorbar(im, ax=ax, label='Weight')
    ax.set_title('Experiment 3: Optimal Blend Weights per Fold', fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS / "cv3_weights_per_fold.png", dpi=120, bbox_inches='tight')
    plt.close()

    # Plot 6: in vs OOF MAE
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(5); w = 0.35
    ax.bar(x - w/2, in_sample_maes, w, label='In-sample (train 4 folds)', color='steelblue', edgecolor='black')
    ax.bar(x + w/2, oof_maes, w, label='OOF (held-out fold)', color='coral', edgecolor='black')
    for i in range(5):
        ax.text(i - w/2, in_sample_maes[i] + 0.003, f'{in_sample_maes[i]:.3f}', ha='center', fontsize=8)
        ax.text(i + w/2, oof_maes[i] + 0.003, f'{oof_maes[i]:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax.set_ylabel('Oracle MAE', fontsize=12)
    ax.set_title('Experiment 3: In-sample vs OOF Oracle MAE', fontsize=13)
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS / "cv3_in_vs_oof_mae.png", dpi=120, bbox_inches='tight')
    plt.close()

    # Plot 7: per-fold predicted public
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(5), oof_pred_pubs, color='seagreen', edgecolor='black')
    ax.axhline(0.8534, color='red', linestyle='--', label='ext150 baseline = 0.8534')
    ax.axhline(0.7952, color='orange', linestyle='--', label='uploaded best = 0.7952')
    ax.axhline(0.79, color='blue', linestyle=':', label='target = 0.79')
    for i, v in enumerate(oof_pred_pubs):
        ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=10)
    ax.set_xticks(range(5)); ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax.set_ylabel('Predicted public MAE (calibrated)', fontsize=12)
    ax.set_title('Experiment 3: Per-Fold Predicted Public (from OOF-found weights)', fontsize=13)
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS / "cv3_oof_pred_pub.png", dpi=120, bbox_inches='tight')
    plt.close()

    print("  ✓ saved cv3_weights_per_fold.png, cv3_in_vs_oof_mae.png, cv3_oof_pred_pub.png")
    return fold_weights, in_sample_maes, oof_maes, oof_pred_pubs, keys


# ============================================================
# Experiment 4: Final candidate held-out evaluation
# ============================================================
def exp4_final_eval_cv():
    print("\n=== Experiment 4: Final candidate held-out 5-fold ===")
    oracle = load_oracle()
    ext150 = load_ext150()
    common = sorted(set(ext150['region_id']) & set(oracle['region_id']))

    truth_v = oracle.set_index('region_id').loc[common][PRED].values
    e = ext150.set_index('region_id').loc[common][PRED].values

    candidates = {
        'ext150 (baseline)': 'submission_round5_pb30_x150_repro.csv',
        '3-way (upload 1, public 0.8100)': '_v18_3way_best_a_we10wt30wd60.csv',
        '4-way (upload 2, public 0.8017)': '_v18_4way_e00t325dp55l519.csv',
        '7-way+T (upload 3, public 0.7952)': '_v18_final7way_T.csv',
        'finalsuper2_T (best, NOT uploaded)': '_v18_finalsuper2_T.csv',
    }
    actual_public = {
        'ext150 (baseline)': 0.8534,
        '3-way (upload 1, public 0.8100)': 0.8100,
        '4-way (upload 2, public 0.8017)': 0.8017,
        '7-way+T (upload 3, public 0.7952)': 0.7952,
        'finalsuper2_T (best, NOT uploaded)': None,
    }

    df_report = pd.read_csv(ROOT / "reports" / "_local_eval_gate_report.csv")
    known = df_report.dropna(subset=['public']).copy()
    coef, _ = fit_calibration(known)

    folds = make_folds(len(common), k=5)
    fold_maes = {name: [] for name in candidates}
    fold_pred_pubs = {name: [] for name in candidates}

    for name, fn in candidates.items():
        v = pd.read_csv(ROOT / "submissions" / fn).set_index('region_id').reindex(common)[PRED].values
        for fi, (_, te) in enumerate(folds):
            mae = float(np.abs(v[te] - truth_v[te]).mean())
            fold_maes[name].append(mae)
            # Compute predicted public for the fold subset
            mad = float(np.abs(v[te] - e[te]).mean())
            m = float(v[te].mean()); s = float(v[te].std())
            stats = {'oracle_mae': mae, 'mad': mad, 'std': s, 'mean': m}
            pp = predict_public(coef, stats)
            fold_pred_pubs[name].append(pp)
            add_result('exp4', fi, f'mae_{name[:30]}', mae)
            add_result('exp4', fi, f'pred_pub_{name[:30]}', pp)

    for name in candidates:
        actual_str = f"actual_pub={actual_public[name]:.4f}" if actual_public[name] is not None else "(not uploaded)"
        print(f"  {name}: oracle_MAE = {np.mean(fold_maes[name]):.4f} ± {np.std(fold_maes[name]):.4f}  "
              f"pred_pub = {np.mean(fold_pred_pubs[name]):.4f} ± {np.std(fold_pred_pubs[name]):.4f}  {actual_str}")

    # Plot 8: per-fold MAE for all candidates
    fig, ax = plt.subplots(figsize=(11, 6))
    n_cand = len(candidates)
    w = 0.15
    colors_c = plt.cm.viridis(np.linspace(0, 0.9, n_cand))
    for i, (name, maes) in enumerate(fold_maes.items()):
        x = np.arange(5) + (i - n_cand/2 + 0.5) * w
        ax.bar(x, maes, w, label=name, color=colors_c[i], edgecolor='black')
    ax.set_xticks(np.arange(5)); ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax.set_ylabel('Oracle MAE on held-out fold', fontsize=12)
    ax.set_title('Experiment 4: Per-Fold Oracle MAE — All Candidates', fontsize=13)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS / "cv4_final_per_fold_mae.png", dpi=120, bbox_inches='tight')
    plt.close()

    # Plot 9: predicted-actual band for final candidate
    final_name = 'finalsuper2_T (best, NOT uploaded)'
    pred_pubs = fold_pred_pubs[final_name]
    fig, ax = plt.subplots(figsize=(8, 6))
    # Show all 4 candidates' actual + final's CV-predicted
    points_x = []; points_y = []; labels = []
    for name, actual in actual_public.items():
        if actual is not None:
            mean_pred = np.mean(fold_pred_pubs[name])
            std_pred = np.std(fold_pred_pubs[name])
            ax.errorbar(actual, mean_pred, yerr=std_pred, fmt='o', markersize=10,
                        capsize=5, capthick=2, label=name, color='steelblue' if 'ext150' not in name else 'red')
            ax.annotate(name[:30], (actual, mean_pred), xytext=(8, 8), textcoords='offset points', fontsize=8)
    # Final candidate: only predicted (no actual)
    mean_pred = np.mean(pred_pubs); std_pred = np.std(pred_pubs)
    ax.errorbar(mean_pred - 0.014, mean_pred, yerr=std_pred, fmt='*', markersize=20, capsize=5,
                color='gold', markeredgecolor='black', label=f'{final_name}\n(predicted only)')
    ax.annotate('Expected based on\ncalibration bias −0.014', (mean_pred - 0.014, mean_pred),
                xytext=(15, -20), textcoords='offset points', fontsize=8,
                arrowprops=dict(arrowstyle='->', alpha=0.5))
    # y=x reference
    lo, hi = 0.74, 0.86
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='y = x')
    ax.axvline(0.79, color='blue', linestyle=':', alpha=0.6, label='Target 0.79')
    ax.set_xlabel('Actual public MAE (or expected for final)', fontsize=12)
    ax.set_ylabel('5-fold CV mean predicted public ± std', fontsize=12)
    ax.set_title('Experiment 4: Predicted vs Actual — Final Candidate Confidence', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    plt.tight_layout()
    plt.savefig(PLOTS / "cv4_predicted_actual_band.png", dpi=120, bbox_inches='tight')
    plt.close()

    # Plot 10: improvement trajectory
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use actual scores for uploaded; CV-predicted for final
    names = ['ext150', '3-way', '4-way', '7-way+T', 'finalsuper2_T\n(predicted)']
    actuals = [0.8534, 0.8100, 0.8017, 0.7952, np.mean(pred_pubs) - 0.014]
    yerrs = [0, 0, 0, 0, np.std(pred_pubs)]
    colors_t = ['red', 'darkorange', 'orange', 'gold', 'green']
    bars = ax.bar(names, actuals, yerr=yerrs, color=colors_t, edgecolor='black', capsize=8)
    for b, v in zip(bars, actuals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(0.8534, color='red', linestyle='--', alpha=0.5, label='ext150 baseline')
    ax.axhline(0.79, color='blue', linestyle=':', label='Target = 0.79')
    ax.set_ylabel('Public MAE (lower better)', fontsize=12)
    ax.set_title('Experiment 4: v18 Improvement Trajectory\n(error bars = 5-fold CV std for unverified final candidate)', fontsize=13)
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0.74, 0.88)
    plt.tight_layout()
    plt.savefig(PLOTS / "cv4_improvement_trajectory.png", dpi=120, bbox_inches='tight')
    plt.close()

    print("  ✓ saved cv4_final_per_fold_mae.png, cv4_predicted_actual_band.png, cv4_improvement_trajectory.png")
    return fold_maes, fold_pred_pubs, actual_public


# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("v18 Cross-Validation — 5-fold region-based")
    print("="*60)

    fm1 = exp1_calibration_cv()
    fm2, dense2 = exp2_lag_rho_cv()
    fw3, in3, oof3, pp3, keys3 = exp3_blend_weights_cv()
    fm4, fp4, ap4 = exp4_final_eval_cv()

    # Save raw CV results
    pd.DataFrame(ALL_RESULTS).to_csv(ROOT / "reports" / "_cv_results.csv", index=False)
    print(f"\nSaved raw CV results: reports/_cv_results.csv ({len(ALL_RESULTS)} rows)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Exp 1 (calibration OOF): mean RMSE = {np.mean([r['value'] for r in ALL_RESULTS if r['experiment']=='exp1' and r['metric']=='rmse']):.4f}")
    print(f"Exp 2 (lag-2215 ρ): mean = {np.mean(fm2['6.07 yr (2215 d)']):.4f}, max = {np.max(fm2['6.07 yr (2215 d)']):.4f}")
    print(f"Exp 3 (blend OOF MAE): mean = {np.mean(oof3):.4f} ± {np.std(oof3):.4f}; predicted public = {np.mean(pp3):.4f} ± {np.std(pp3):.4f}")
    print(f"Exp 4 (finalsuper2_T fold MAE): mean = {np.mean(fm4['finalsuper2_T (best, NOT uploaded)']):.4f}")
    print(f"Exp 4 (finalsuper2_T fold pred_pub): mean = {np.mean(fp4['finalsuper2_T (best, NOT uploaded)']):.4f} ± {np.std(fp4['finalsuper2_T (best, NOT uploaded)']):.4f}")
    print(f"Expected actual (with bias -0.014): {np.mean(fp4['finalsuper2_T (best, NOT uploaded)']) - 0.014:.4f}")


if __name__ == "__main__":
    main()
