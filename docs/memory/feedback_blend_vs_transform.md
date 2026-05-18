---
name: Convex blends behave differently from transforms in MAD-public-MAE space
description: When estimating a candidate's public MAE from MAD vs public-best, treat the MAD slope as an upper bound for convex blends, not a point estimate
type: feedback
originSessionId: e0c2f638-bb8b-4524-a4af-72207d0a6fd2
---
The empirical slope `+0.23 public-MAE per unit MAD` was estimated from three failed post-processing transforms in May 2026 (`hsharp`, `hybrid_qsharp`, `qmap100`). It correctly predicted those FAILURES but **overestimated** the worsening for genuine convex blends.

**Why:** On 2026-05-11, `submission_redo_blend_pb30.csv` (MAD `0.091` vs prior public-best `0.8773`) was expected by the slope to score around `0.898`. Actual public score was `0.8593` — better than the prior public-best by `0.0180`, the opposite direction from the proxy. A convex blend cannot move *further* from the truth than the worse of its two endpoints; transforms can.

**How to apply:** When recommending a single Kaggle upload that risks deviating from a proven public-best CSV:
- Use `MAD ≤ 0.10` as a hard *safety* gate (still defensible — the slope-derived worst case is < `+0.023`).
- Do NOT use `predicted_public = public_best + 0.23 × MAD` as an expected-value forecast for convex blends. Use it only as the worst-case-plus-some-headroom check.
- Tiebreak among eligible blend candidates by maximizing the ensemble fraction (more new signal), not by minimizing MAD — this captures the upper-tail probability that the new model is genuinely better.
