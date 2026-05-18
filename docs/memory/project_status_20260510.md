---
name: DM 114 final project status as of 2026-05-12 evening (9-leg MLP stacker)
description: Concrete state after Plan v4 multi-track research; stacker val MAE 0.310 represents new best evidence
type: project
originSessionId: e0c2f638-bb8b-4524-a4af-72207d0a6fd2
---

State after Plan v4 (2026-05-12 18:30 Asia/Taipei):

**Public leaderboard (Kaggle history):**
- Public best: `0.8534` from `submission_redo_extrapolate_150_mean12334.csv` (concurrent parallel work; transform = `mean + 1.5*(pb30 - mean)`)
- Previous best: `0.8593` from `submission_redo_blend_pb30.csv` (Round 1 v1+v2 blend)
- Our recent attempts (all worse than 0.8534): 0.8599 (round2_blend_pb30), 0.8609 (round2_ensemble), 0.8687 (concurrent broad)

**Trained base learners (9 legs available):**
- v1 (blackout91, original features): 3 seeds, val MAE 0.375-0.381
- v2 (blackout91 + extended 20 features: pressure anomaly, precip intensity, ET, neighbor, DOY): 3 seeds, val MAE 0.362-0.367
- v2_pl (public_like gap-mode + extended features): 3 seeds, val MAE 0.394-0.399
- v1_pl (Round 4 public_like, original features, only val preds — no submissions): 3 seeds, val MAE 0.425-0.431

**MLP stacker results (val MAE on multi-slice {728, 735, 742}):**
- Leg-mean baseline (9 legs): 0.368
- 3-leg v1 only MLP: 0.342
- 6-leg v1+v2 MLP: 0.328
- **9-leg v1+v2+v2pl MLP (64,32) max500: 0.3100** ← best
- Ridge stacker on 9 legs: 0.41 (worse — MLP much better)

**Ready-to-upload candidates (in submissions/, all from 9-leg stacker):**
- `submission_stacker_9leg_v2mlp_shift27.csv` — primary, mean 1.219, MAD 0.236 vs pb30
- `submission_stacker_9leg_v2mlp_shift30.csv` — mean 1.249, MAD 0.235
- `submission_stacker_9leg_v2mlp_shift27_x150.csv` — extrapolation aggressive, mean 1.243, MAD 0.438
- `submission_redo_extrapolate_150_mean12334.csv` reproduced as `submission_round5_pb30_x150_repro.csv` (matches 0.8534)

**Next quota window:** ~2026-05-13 09:08 Asia/Taipei (24h after last submission at 01:08 UTC 2026-05-12).

**Key empirical lessons:**
- val MAE → public MAE ratio: pb30 had 0.376 → 0.8593 (ratio 2.286). Whether 0.310 stacker val translates to public 0.310 × 2.286 = 0.71 is uncertain.
- Extrapolation transform (`mean + factor × (pred - mean)`) DOES help on public for genuine model improvements (slope NEGATIVE) — opposite of post-processing transforms (slope +0.23 per MAD).
- MLP stacker > Ridge stacker because non-linear per-region adaptation helps even with GroupKFold leakage protection.
- Adding architectural diversity (gap-modes blackout91 + public_like) helps stacker by 0.01-0.02 val MAE.

**How to apply:** When recommending the next upload, use `submission_stacker_9leg_v2mlp_shift27.csv` as the primary attempt. It's the strongest val-MAE candidate. If it improves (e.g., < 0.85), try shift30 or extrapolation variants. If it worsens (> 0.86), consider blending with extrapolate_150 (known 0.8534) for safer ground.
