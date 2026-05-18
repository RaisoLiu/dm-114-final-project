---
name: transformer-bounded-sigmoid-l1-collapses-to-zero
description: "TransformerEncoder + 5*sigmoid output wrapper + L1Loss collapses to predicting 0 during refit-all, even when GroupKFold folds gave reasonable (val 0.86) results"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e0c2f638-bb8b-4524-a4af-72207d0a6fd2
---

In Plan v5 (2026-05-14) on DM 114, a `DeepTrans` model (4-layer TransformerEncoder, d_model=64 originally then bumped to d_model=128 with CLS token + pre-LN + LR warmup) trained with the `BoundedOutputWrapper` (`pred = 5 * sigmoid(raw)`) and `L1Loss` exhibits a pathological pattern:
- **GroupKFold folds 1, 2, 3, 5**: val MAE 0.85-0.87 (constant-predictor level — equivalent to predicting global mean) due to early stopping after model fails to escape the "predict global ~1.5" local minimum.
- **Fold 4** (one specific GroupKFold split): val MAE 0.41 — model DID escape and learned, ran for ~30 min (~6× longer than other folds).
- **Refit-all step (after CV)**: trains 100 epochs on ALL data with no held-out validation. Model parameters drift toward raw → -∞, making `5*sigmoid(raw) → 0`. **Final test predictions are all 0.00**, MAD vs ext150 = 1.23 (completely useless).

The same hyperparameters in `DeepCNN` (val 0.34) and `DeepLSTM` (val 0.27-0.29) work perfectly. The capacity fix (d_model 64 → 128, mean-pool → CLS, dim_feedforward 128 → 512, add 10-epoch LR warmup) did NOT improve trans val MAE — both architectures show identical 0.85+ pattern.

**Why:** L1 loss has constant-magnitude gradients (±1). Combined with `5*sigmoid` saturation at the extremes, the optimizer has no informative gradient signal once the transformer's attention-pooled representation collapses (which it does easily because Q·K^T attention scores over smoothly-varying weather time-series tend to be near-uniform → encoder output becomes near-identical across positions → CLS/mean pool is similar across regions). The optimizer then "discovers" that pushing all raw outputs negative is a trivial way to reduce L1 loss when the target distribution is heavily skewed toward 0 (target mean 0.84). CNN/LSTM avoid this because their inductive biases (local conv kernels, recurrent gating) produce more region-discriminating pooled features even at random init, giving the optimizer a usable starting gradient.

**How to apply:** Do not use `BoundedOutputWrapper(5*sigmoid)` with transformer encoders for this competition's data unless paired with: (a) MSE loss instead of L1 (gradient scales with error), (b) much more aggressive gradient clipping, (c) a different output head (e.g., 2.5 + 2.5·tanh(raw·0.5) — softer bounded), or (d) a per-region init that gives the model a "warm start" away from the collapse basin. For DM 114, the cleaner path is to abandon transformer in the ensemble and lean on CNN + LSTM with the bounded wrapper. See also [[deep-cnn-converges-on-val-but-saturates-on-test]] for the broader val→public mismatch on this dataset.
