# DM 114 Final Project — Strategy Journey

This document is the **entry point for team members writing the report**. It walks the methods we tried, what each taught us, and where the ceiling currently sits.

The chronological strategy reports are in `reports/`; the method-level failure lessons are in `docs/memory/`; the iteration plans are in `docs/plans/`. This file ties them together.

## Final scores (as of 2026-05-18)

| Submission | Method family | Public MAE | Notes |
|---|---|---|---|
| **ext150 (team best)** | `extrapolate_150` transform on pb30 | **0.8534** | Anchor; sharpens pb30 toward truth distribution |
| pb30 (my best) | `redo_blend_pb30` — 70% multi-slice LGBM/HGB ensemble + 30% public-best | 0.8593 | Plan v2 final |
| latent_nowcast round 2 | 4-leg ensemble with latent nowcast leg | 0.8609 | Plan v3 |
| 9-leg MLP stacker | Stacker over 9 GBDT legs | 0.8688 | Val 0.310, public 0.8688 → val→public divergence confirmed |
| 6-leg deep ensemble 10% blend | CNN + LSTM ensemble × 10% + ext150 × 90% | 0.8767 | Deep family closed |
| Plan v7 U1 DOY blend w08 | per-region DOY signal × 8% + ext150 × 92% | 0.8848 | Slope +0.48/MAD; DOY family closed |
| Plan v7 U2 H2 Recent-Zero α=0.20 | Shrink ext150 on 576 "recent-zero" regions | 0.8919 | Slope +0.42/MAD; selective-shrinkage failed |
| Plan v7 U3 v6 3-leg shifted w15 | Adversarial-val LGBM ensemble × 15% + ext150 × 85% | 0.8853 | Slope +0.56/MAD; v6 family closed |
| ext170 transform | Stronger extrapolation (factor 1.7 vs 1.5) | 0.9208 | Direction is wrong; weakens ext150 |

**Headline:** ext150's **0.8534** is our team's standing best and has been confirmed as a ceiling across 3 structurally distinct candidate families.

## The strategy timeline

### Plan v1–v2 (early May 2026): Baselines + sharpening
- Built LightGBM and HistGradientBoostingRegressor over 91-day weather windows (1071-feature pipeline at `src/drought/features.py`).
- Discovered `extrapolate_150` post-processing transform on pb30 → ext150 0.8534. This became the team anchor.
- Strategy report: `reports/strategy_2026-05-10_0810_three_uploads.md`.

### Plan v3 (mid May): Multi-slice ensembling + latent nowcast
- Built `redo_blend_pb30` (70% multi-slice ensemble + 30% pb30) → 0.8593.
- 4-leg round 2 ensemble with latent_nowcast leg → 0.8609 (worse).
- Strategy report: `reports/strategy_2026-05-11_redo.md`.

### Plan v4: Stackers + per-DOY shifts
- 9-leg MLP stacker (val 0.310, public 0.8688) — first hard val→public divergence evidence.
- Memory: `docs/memory/feedback_stacker_overfit.md` (GroupKFold(region) leaks structure across folds).

### Plan v5: Deep models (CNN, LSTM, Transformer)
- 3 architectures with bounded-sigmoid output + L1 loss.
- CNN val 0.343 → public 0.97 (saturated H1 predictions, OOD on `cutoff_age` feature).
- LSTM val 0.40 → public 0.9032.
- Transformer collapsed to all-zeros in refit-all phase (`docs/memory/feedback_trans_bounded_sigmoid_collapse.md`).
- 6-leg deep 10% blend → 0.8767 (worse than ext150). Slope +0.37 → deep family permanently closed.
- Memory: `docs/memory/feedback_deep_*.md`.

### Plan v6: Adversarial validation + residual-on-ext150
- Built `scripts/adversarial_validation.py` — LightGBM classifier to identify test-like anchors (AUC 0.97).
- Replaced the `valid_deltas {728,735,742}` slice (target mean 1.16) with adversarial top-5% per region (target mean 1.35, closer to public 1.21).
- E1 LightGBM with adversarial val → public 0.8886.
- 3-leg shifted blend → public 0.8853.
- Stronger ext170 transform → 0.9208 (transform direction confirmed wrong beyond ext150).
- Memory: `docs/memory/project_mad_slope.md`, `feedback_blend_vs_transform.md`.

### Plan v7 (2026-05-18): 2-layer agent system + 3 strategic uploads
- Designed an OKR-aligned agent system (supervisor + Agent A/B/C specialists). See `docs/plans/plan_v8_external_training.md` for the latest version (overwrote Plan v7 after completion).
- 3 uploads, each with structurally distinct method:
  - U1: DOY blend w08 → 0.8848
  - U2: H2 Recent-Zero Mask α=0.20 → 0.8919
  - U3: v6 3-leg shifted w15 → 0.8853
- All three positively correlated (ρ > 0) with ext150 errors. Anti-correlation never achieved.
- Memory: `docs/memory/feedback_plan_v7_3upload_retro.md` — the conclusive retro.

### Plan v8: Research external training methods (NEXT)
- Conclusion from v7: we have exhausted the existing weather+score+DOY feature space.
- Plan v8 (in `docs/plans/plan_v8_external_training.md`) is a research-only plan to catalog external approaches: external data, pretrained foundation models, transfer-learning / domain-adaptation paradigms, Kaggle pattern mining.
- Not yet executed at time of repo creation.

## The empirical MAD-vs-public slope law

A central empirical finding from this competition: **for any candidate with MAD(candidate, ext150) > 0, the public MAE goes UP by approximately 0.42 × MAD or more.**

Evidence (uploads with known MAD + public):

| Candidate | MAD vs ext150 | Public | Slope |
|---|---|---|---|
| ext170 transform | 0.092 | 0.9208 | +0.733 |
| DOY w08 blend | 0.065 | 0.8848 | +0.483 |
| H2 α=0.20 (best slope) | 0.091 | 0.8919 | +0.422 |
| v6 3-leg w15 | 0.057 | 0.8853 | +0.561 |
| Deep ensemble w10 | 0.063 | 0.8767 | +0.370 |

**No candidate has demonstrated negative slope.** Achieving public MAE < 0.8534 requires a candidate whose per-region errors are *anti-correlated* (ρ < 0) with ext150's errors. Plan v7 exhausted DOY signal, selective shrinkage, and adversarial-val LGBM as paths to this — none worked.

The only remaining angles (Plan v8) are signals genuinely outside the existing feature space: external satellite data, pretrained time-series foundation models, or transfer-learning paradigms.

## What team members should read for the report

For the methodology section of your report:

1. **`README.md`** — project setup, pipeline overview.
2. **This file (`docs/JOURNEY.md`)** — strategy timeline + final results.
3. **`reports/strategy_2026-05-11_redo.md`** — most comprehensive strategy doc, written after Plan v3.
4. **`docs/memory/`** — short failure-mode notes; each is a single distilled lesson.
5. **`docs/plans/plan_v8_external_training.md`** — what's planned next.

For the methods description:
- `src/drought/features.py` — the 1071-feature pipeline (91-day weather windows, DOY encodings, per-region rolling stats).
- `src/drought/modeling.py` — LightGBM + HGB training, ensembling, calibration.
- `scripts/train_gap_model.py` — the main training driver.
- `scripts/adversarial_validation.py` — the test-like anchor classifier.
- `scripts/extrapolate_150.py` (or referenced wherever the transform lives) — the **ext150** sharpening transform that gave us our team best.
- `scripts/h2_recent_zero_mask.py` — Plan v7 U2 implementation.
- `scripts/cycle_doy_baseline.py` — Plan v7 U1 building block.
- `scripts/v6_blend.py`, `scripts/final_blend.py` — blending utilities.
- `scripts/quantile_map_calibrate.py` — per-DOY calibration.

For results / honest assessment:
- The 0.8534 ceiling story — `docs/memory/feedback_plan_v7_3upload_retro.md`.
- Val→public divergence — `docs/memory/feedback_stacker_overfit.md`, `feedback_deep_test_distribution_shift.md`.
- MAD-slope law — `docs/memory/project_mad_slope.md`, `feedback_blend_vs_transform.md`.

## Data note for the report

The competition data (`data/train.csv`, `data/test.csv`) is NOT in this repo (Kaggle TOS + size). Team members need to download from the Kaggle competition page (URL in `ref/koggle_link`). The submission CSVs are also not tracked — they are regenerable from the scripts. The submission filename + Kaggle public score is in this JOURNEY for the team to reference.

## Open question for the report

Why did the field reportedly break < 0.80 while we plateau at 0.8534? Three hypotheses (no evidence yet):
1. **External data**: someone used satellite (NDVI/SPI/SPEI) or reanalysis features.
2. **Internal-data trick we missed**: a feature in `data/train.csv` we have not extracted (e.g. cross-region neighbor effects, exact synthetic-year cycle phase).
3. **Different model class**: foundation-model zero-shot (TimeGPT/Chronos) or LLM-as-forecaster.

Plan v8's literature scan will narrow this. For the report, the honest framing is: "we identified a real ceiling at 0.8534 for the candidate space we explored; we identify 3 hypotheses for the gap to the leaderboard's < 0.80 territory and propose a follow-up research direction."
