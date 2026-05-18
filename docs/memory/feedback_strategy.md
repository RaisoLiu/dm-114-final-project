---
name: Strategy preference — trust senior-researcher framing with bounded experimentation
description: User wants honest probabilistic reasoning, not pure-defense or pure-attack — accepts hybrid plans with strict eligibility gates
type: feedback
originSessionId: e0c2f638-bb8b-4524-a4af-72207d0a6fd2
---
When the user says "complete redo" or "新模型架構", do NOT take it literally if a senior-researcher analysis shows higher EV in a bounded-experimentation hybrid (new model + safe fallback gated by selection rule).

**Why:** On 2026-05-10 the user explicitly chose "complete redo" but, when shown the MAD-vs-public-MAE slope evidence (~+0.23 per unit MAD) and the 5-10% reachability assessment, picked "你判斷，我信你" — confirming they value honest framing over the literal answer they first gave. The prior 3 uploads failed precisely because the team did not bound MAD vs public-best.

**How to apply:** Before recommending an aggressive plan, surface (a) the honest probability of hitting the stated target, (b) the worst-case downside, (c) whether a bounded hybrid has higher EV. If the user disagrees after seeing this, follow their judgment; if they delegate ("信你"), pick the bounded hybrid.
