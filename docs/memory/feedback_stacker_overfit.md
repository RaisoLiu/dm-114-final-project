---
name: mlp-stacker-groupkfold-validation-does-not-transfer-to-public
description: Cross-validation MAE improvements from per-region MLP stackers (val 0.376 → 0.310) did NOT transfer to public test in DM 114
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e0c2f638-bb8b-4524-a4af-72207d0a6fd2
---

In Plan v4 (2026-05-13), a per-horizon MLPRegressor stacker (hidden 64,32) trained on a 9-leg OOF tensor (3 b91 + 3 v2 + 3 v2_pl) with GroupKFold(region, n_splits=5) achieved validation MAE **`0.310`** — substantially better than:
- 3-leg simple mean (val `0.376` → public `0.8593` via Round 1 pb30)
- Ridge stacker (val `0.357`)

**Public MAE for the 9-leg MLP stacker shift+0.27**: **`0.8688`** — `+0.0095` WORSE than Round 1 pb30 (`0.8593`), and `+0.0154` worse than the team's actual best ext150 (`0.8534`).

A 50/50 blend with ext150 came out `0.8674` — monotonically worse than ext150 alone. So the stacker output is structurally pulling predictions AWAY from public truth.

**Why:** GroupKFold(region) holds out whole regions in each fold; the stacker can learn region-specific patterns on training regions, but at test time those patterns don't transfer to the SAME regions on FUTURE anchor dates. The stacker's improvement over leg-mean was apparently coming from memorizing per-region structure on the validation slice, which is at deltas {728, 735, 742} from training end — not from generalizable signal.

**How to apply:** For DM 114 (or similar small-OOF + GroupKFold setups), do NOT trust a stacker's cross-validation MAE improvement as a proxy for public MAE. Use simple-average blending OR genuine post-processing (like the extrapolate_150 transform `mean + 1.5 × (pred - mean)`) which has been empirically validated on public. If you want to test a stacker concept anyway, upload at most ONE candidate and bound MAD vs the known public best to ≤ 0.10 — pure stackers with MAD 0.24+ have historically failed on this competition.
