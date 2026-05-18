---
name: deep-cnn-converges-on-val-but-saturates-on-test
description: First proper-GPU deep CNN (Plan v5) converged to val 0.357 but test predictions saturate at 5.0 for H1/H2 — severe OOD shift between train+val anchors and test anchors
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e0c2f638-bb8b-4524-a4af-72207d0a6fd2
---

In Plan v5 (2026-05-14), `train_deep_model.py --arch cnn --seed 114 --epochs 100 --device cuda` with 128 samples/region, 5-fold GroupKFold:
- **OOF MAE 0.3566** on slice {728,735,742} — competitive with v2 GBDT best (0.362).
- Val pred mean 1.07 vs y_true 1.16 (val MAE 0.39, mean OK).
- **Test pred mean 2.90** (vs target ~1.21):
  - H1: mean 4.60, **71% saturated at 5.0**
  - H2: mean 4.28, 57% saturated
  - H3: mean 2.97, 5.9% saturated
  - H4: mean 1.70, 0.4% saturated
  - H5: mean 0.95, 0.2% saturated

**Why:** validation slice deltas {728,735,742} are inside training-anchor distribution (gap_mode=blackout91 trains with varied cutoff_age), but test anchors are at cutoff_age≈0 (the very end of history). The CNN learned that recent-cutoff patterns predict high severity — a pattern that holds in train (where high-severity examples cluster near training-end) but explodes on test. Per-horizon shift calibration can't fully recover: ~71% of H1 predictions are clamped at 5.0, losing info.

**How to apply:** Treat deep-model val OOF MAE with extreme skepticism in this competition — it can hide a fundamental train→test distribution failure that GBDT models avoid because their feature engineering bounds extrapolation. Before trusting a deep leg, check: (a) test pred mean vs val pred mean, (b) per-horizon saturation rate. If H1 saturation > 20%, the leg is functionally broken; calibration can't reverse information loss. Mitigations to try in the future: drop `cutoff_age` from input scalars; bounded-output head (e.g. `5 * sigmoid(raw)`); upsample anchors with cutoff_age near 0 during training. Also: see [[mlp-stacker-groupkfold-validation-does-not-transfer-to-public]] — the same "val-fits-but-doesn't-transfer" pattern.
