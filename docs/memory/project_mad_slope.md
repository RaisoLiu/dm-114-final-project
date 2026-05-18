---
name: MAD-vs-public-MAE slope for the DM 114 Kaggle leaderboard
description: Empirical slope for how much public MAE worsens per unit MAD distance from the public-best CSV
type: project
originSessionId: e0c2f638-bb8b-4524-a4af-72207d0a6fd2
---
From the 3 uploads on 2026-05-10 (each compared to public-best `submission_blackout91_blend_cal02_fast64_gpu_shift35.csv`, public 0.8773):

| Candidate | MAD vs public-best | Public delta vs 0.8773 |
|---|---:|---:|
| hsharp | 0.173 | +0.024 |
| hybrid_qsharp | 0.334 | +0.072 |
| qmap100 | 0.685 | +0.158 |

Slope ≈ **+0.23 public-MAE per unit MAD**. With MAD ≤ 0.10, expected worst-case worsening ≈ +0.023, capping public MAE near 0.90.

**Why:** This is the empirical safety bound for any single Kaggle upload that diverges from the proven public-best CSV. It explains why post-processing transforms with high MAD (qmap100 at MAD 0.685) catastrophically worsened public MAE.

**How to apply:** Treat MAD ≤ 0.10 vs the current public-best as a hard eligibility gate for any future upload. Reject candidates whose MAD exceeds the bound, regardless of how strong the local validation evidence is.
