---
name: feedback-doy-blend-failed
description: DOY blend family Kaggle-confirmed worse than ext150 in DM 114; slope +0.48 vs ext150
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e0c2f638-bb8b-4524-a4af-72207d0a6fd2
---

DOY blend w08 (per-region DOY-cycle signal × 0.08 + ext150 × 0.92, MAD 0.065 vs ext150) uploaded 2026-05-18 → public **0.8848**.

That's +0.0314 over ext150's 0.8534. Empirical slope = +0.0314 / 0.065 = **+0.48 per unit MAD** — much steeper than the prior +0.23 average from the v5/v6 ext150-family probes.

**Why:** The DOY signal had low (0.31) error-correlation vs ext150 on internal stats, but in practice its per-region rankings, when blended in, push the wrong regions in the wrong direction. Low *predictor* correlation ≠ low *error* correlation against public truth.

**How to apply:**
- The entire DOY-blend family (w05/w08/w10/w12 from `submissions/submission_doy_blend_w*.csv`) is now closed for DM 114. Don't burn the remaining 2/3 uploads on smaller weights — w05 (MAD ~0.04) is still expected at ~0.873, worse than 0.8534.
- KR1 (beat ext150) requires something with *anti-correlated* (ρ<0) per-region errors vs ext150, not just low correlation.
- Future candidates must satisfy this test on the adversarial-val slice before being shipped.

Related: [[feedback_deep_blend_kaggle_confirmed]] documented same slope phenomenon (+0.37) for deep ensemble blends. Both confirm: low-correlation isn't enough; need anti-correlation to break the ext150 ceiling.
