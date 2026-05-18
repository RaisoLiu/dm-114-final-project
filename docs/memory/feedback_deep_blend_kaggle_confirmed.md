---
name: deep-ensemble-10pct-blend-with-ext150-confirmed-worse-on-kaggle
description: "6-leg deep ensemble (2 CNN + 4 LSTM, all bounded sigmoid) at 10% blend with ext150 anchor scored 0.8767 — 0.0233 WORSE than ext150 alone (0.8534), confirming deep predictions have correlated errors with ext150 in DM 114"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e0c2f638-bb8b-4524-a4af-72207d0a6fd2
---

In Plan v5 (2026-05-14) on DM 114, after fixing the deep CNN/LSTM OOD blowup (bounded `5*sigmoid` output + drop `cutoff_age` scalar), a final 6-leg deep ensemble (2 CNN + 4 LSTM seeds, trans excluded as confirmed broken) was blended at 10% with the team's public best `submission_round5_pb30_x150_repro.csv` (ext150, score 0.8534). The blend `submission_deep_ensemble_ext150_w10.csv` had MAD vs ext150 = **0.063** (well under the 0.10 eligibility gate) but scored:

**Public MAE: `0.8767`** — `+0.0233` WORSE than ext150 alone, on a 10% deep blend. Empirical MAD-to-public slope = `+0.37 per unit MAD` (steeper than the prior `+0.23/unit` upper bound for transforms documented in [[mad-vs-public-slope]]).

**Individual deep legs' public scores:**
- `cnn_fixed_s114` (val OOF 0.343): public `0.9706`
- `lstm_fixed_s114` (val OOF 0.40 with fold-4 outlier 0.89): public `0.9032`
- (other seeds not uploaded but consistent val 0.27-0.34)

**Why:** Deep models' validation MAE (0.27-0.34) was BETTER than GBDT's (0.36-0.38) on the same slice {728, 735, 742}, but their public score is `0.10-0.12` WORSE. The val→public mismatch is not just a magnitude issue — deep predictions' errors are *positively correlated* with ext150's residual errors on public regions. Adding deep to ext150 amplifies wrong directions instead of cancelling them. The deep architecture learned validation-slice patterns (which include the high-severity plateau at deltas {721-756}) that don't generalize to the public-test anchor positions (last days of training history, cutoff_age ≈ 91, different DOY distribution). See [[deep-cnn-converges-on-val-but-saturates-on-test]] for OOD shift and [[transformer-bounded-sigmoid-l1-collapses-to-zero]] for trans-specific failure.

**How to apply:** For DM 114 (and likely any forecasting competition where validation must be held-out from end-of-train), DO NOT trust deep model val MAE as a proxy for public MAE — the train-anchor-sampling distribution (`np.arange(365, last_usable - 420, 7)` in train_deep_model.py) systematically excludes test-like positions, creating an OOD gap. Required experiments before trusting deep for public: (a) adversarial validation between train anchors and test anchors to confirm distribution shift, (b) retrain with broader anchor sampling that includes positions within 90 days of `last_usable`, (c) try deep OOF as features for an LGBM stacker rather than direct blend — LGBM might learn to use deep features only where they help. As of 2026-05-14 with 2 quotas left, this deep-ensemble track is closed; the path to <0.85 will not come from this family without a fundamentally different validation/training setup.
