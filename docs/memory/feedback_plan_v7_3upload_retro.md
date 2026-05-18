---
name: feedback-plan-v7-3upload-retro
description: Plan v7 3-upload portfolio results 2026-05-18; ext150 0.8534 confirmed as ceiling across 3 distinct candidate families
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e0c2f638-bb8b-4524-a4af-72207d0a6fd2
---

Plan v7 was a 2-layer agent system (supervisor + Agent A/B/C specialists) targeting #1 leaderboard within 3 uploads. All three uploads scored *above* team-best ext150 (0.8534), confirming the ceiling hypothesis.

| Cycle | Candidate family | MAD vs ext150 | Public | Implied slope |
|---|---|---|---|---|
| U1 | DOY-cycle blend (8% × per-region DOY signal + 92% ext150) | 0.065 | 0.8848 | +0.483 |
| U2 | Selective shrinkage (H2: shrink ext150 by α=0.20 on 576 regions where last-12 scores=0 AND ext150_avg>1.0) | 0.091 | 0.8919 | +0.422 |
| U3 | Adversarial-val-trained LGBM 3-leg ensemble (per-horizon mean-shifted to ext150, 15% blend) | 0.057 | 0.8853 | +0.561 |

**Why this matters:**
- Three structurally distinct candidate families (DOY, selective shrinkage, adversarial-val LGBM) all produced slopes in [+0.42, +0.56]. No anti-correlation achieved.
- The lowest slope (+0.42 H2) was the most directionally informed candidate — but adversarial-val MAE dropped 0.031 yet public went UP 0.039. **Val→public transfer is broken in DM 114**.
- The v6 adversarial-val pipeline (which we previously thought might be the key) actually had the WORST slope (+0.56). This closes the v6 family permanently.
- Across all 9 historical uploads at MAD ∈ [0.05, 0.10] vs ext150, the lowest slope ever observed was Plan v7 U2's +0.42 H2. **For ANY future candidate with MAD>0 vs ext150, expected public > 0.8534.** The way to beat 0.8534 is to find a candidate with negative MAD attribution — which only happens if the candidate's per-region errors are negatively correlated with ext150's — which we have NEVER demonstrated empirically.

**How to apply (future cycles):**
- If new attempts at DM 114 are pursued, do NOT spend cycles on (a) DOY-blend variants, (b) selective-shrinkage variants of ext150, (c) adversarial-val-trained LGBM. All three families closed.
- The only remaining direction with non-trivial P(beat) is finding a feature/signal **not derivable from the existing weather+score+DOY feature space** — e.g. external data (NOAA satellite indices, soil moisture proxies), or a novel cross-region transfer-learning hypothesis. Without new signal sources, the ceiling is real.
- For Plan v8 (if pursued), abandon the "tweak ext150" framing entirely. Build a model from a feature space ext150 cannot see.

**Process note (the 2-layer agent system):**
- The Agent A (Plan/hypothesis), Agent B (general-purpose/implement), Agent C (Plan/risk-blend) decomposition worked cleanly. No agent gave a clearly-bad recommendation given the information available.
- The failure was upstream of the agent system — the candidate space was too narrow. Even the best-process couldn't manufacture signal that wasn't in the data.

Related: [[feedback_doy_blend_failed]], [[feedback_deep_blend_kaggle_confirmed]], [[feedback_stacker_overfit]], [[project_mad_slope]].
