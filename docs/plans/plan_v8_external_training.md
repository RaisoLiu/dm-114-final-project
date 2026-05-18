# Plan v8 — Research: External Training Knowledge (探索外部訓練方式)

## Context

Plan v7 closed with three uploads all worse than ext150 (0.8848, 0.8919, 0.8853 vs ext150's 0.8534). Across three structurally distinct candidate families — DOY blends, selective shrinkage, adversarial-val LGBM ensembles — MAD-to-public slope stayed in [+0.42, +0.56]. No candidate produced anti-correlated (ρ<0) errors against ext150. The memory note `feedback_plan_v7_3upload_retro.md` records the conclusion: **the path to actually moving below 0.8534 requires a feature/signal source outside the existing weather+score+DOY feature space.** Plan v8 is a research plan — *not* an implementation plan — to systematically scan what "outside the existing space" can mean, before any further uploads are spent.

The deliverable of Plan v8 is *understanding*: a catalog of external training approaches with feasibility + expected-lift estimates against the DM 114 constraints. Promotion to implementation requires a separate Plan v9.

## Scope — 4 categories of "external training"

### A. External **data sources** (new features into any model)
Information that ext150 demonstrably cannot see, because it isn't in `data/train.csv` or `data/test.csv`.
- Satellite vegetation: MODIS NDVI/EVI, VIIRS surface temperature, SMAP soil moisture
- Climate reanalysis: ERA5-Land evapotranspiration, root-zone soil moisture, snow water equivalent
- Pre-computed drought indices from external providers: NOAA's USDM, NASA's GRACE-derived groundwater, PDSI, SPEI at multi-timescales
- Static layers: land cover (cropland/forest/urban), elevation, soil texture

### B. External **pretrained models** (transfer learning / feature extractors)
Models trained on far-larger corpora than DM 114, used either as feature extractors or fine-tuned heads.
- Time-series foundation models: TimeGPT, Chronos (Amazon), Lag-Llama, Moirai (Salesforce), Tiny Time Mixers
- Earth Observation FMs: Prithvi-100M (NASA/IBM), ClimaX (Microsoft), GraphCast/Pangu-Weather (DeepMind/Huawei)
- Generic LLMs as zero-shot forecasters (GPT-class models with numeric prompting)

### C. External **training paradigms** (different optimization, same data)
- Self-supervised pretraining on the 2248 regions × N-years time-series, then supervised fine-tune
- Domain adaptation (RAINCOAT frequency-domain DA, CORAL, DANN) to align train→test distribution
- Meta-learning across regions (MAML, ProtoNet) to share structure across the 2248 region-tasks
- Adversarial-feature training: explicitly train features orthogonal to ext150's predictions
- Knowledge distillation: distill a large pretrained model down to a small one usable on our compute budget

### D. External **competition solutions** (steal-from-Kaggle pattern mining)
- Winning solutions from other drought / weather / time-series Kaggle competitions (M5, Web Traffic, Rossmann, ASHRAE energy, GEFS, USGS streamflow, etc.)
- Their feature-engineering patterns, especially anything we did not try (target-relative-to-cohort, cross-region rolling rank features, etc.)
- Common ensemble structures that consistently beat single models

## Critical preliminary check — must come first

**Read the DM 114 / `data-mining-2026-final-project` Kaggle competition rules** before *any* category-A or category-B work. Many competitions forbid external data (and sometimes forbid pretrained models, depending on rules). If external data is disallowed, category A is dead and we focus on B/C/D. If only externally-published-before-deadline data is allowed, we filter category A accordingly.

This single check could collapse the plan from 4 categories to 2–3 — do it first.

## Phase 1 — Literature scan (parallel exploration)

Dispatch **4 Explore/research agents in parallel**, one per category. Each agent has a tight scope and a fixed-format deliverable.

### Agent E1 (category A — external data)
- Search for: drought-forecasting papers + Kaggle/research datasets that bring in satellite or reanalysis data
- Output: 1-page markdown table at `reports/plan_v8_cat_a_external_data.md`. Columns: dataset name | resolution (spatial/temporal) | typical drought-forecast lift in the literature | API/access cost | DM 114 compatibility (region IDs mappable? time coverage matches?) | citation.

### Agent E2 (category B — pretrained models)
- Search for: time-series foundation models + EO foundation models published 2023–2026
- Output: `reports/plan_v8_cat_b_pretrained.md`. Columns: model name | params | training corpus | zero-shot or fine-tune mode | published MAE on drought-like benchmarks | compute requirements | DM 114 compatibility | citation.

### Agent E3 (category C — training paradigms)
- Search for: domain adaptation + meta-learning + self-supervised pretraining for time series, especially RAINCOAT and successors
- Output: `reports/plan_v8_cat_c_paradigms.md`. Columns: paradigm | typical lift on shift-affected benchmarks (NeurIPS/ICML 2022–2026) | implementation difficulty | DM 114-specific notes (e.g., does it need >5 source domains? does it need a real OOD test set?) | citation.

### Agent E4 (category D — Kaggle solutions)
- Search Kaggle for winning solutions to: M5, Web Traffic, Rossmann, time-series forecast competitions; specifically look for feature-engineering tricks we have not used.
- Output: `reports/plan_v8_cat_d_kaggle.md`. List: 5–8 specific *techniques* (not solutions) with 1-sentence description, the competition where it was used, and a 1-sentence applicability note to DM 114.

**Time budget:** Phase 1 wall time = 1 hour (4 agents in parallel, ~15-min each + ~15 min synthesis).

## Phase 2 — Feasibility filter

Cross-reference Phase 1 outputs against DM 114 constraints:
- Competition rules on external data and pretrained models
- Available compute (GB10 + 4090 + 3070 cluster from memory `user_role.md`)
- Available wall time before competition deadline (verify on Kaggle competition page)
- Whether the technique can produce a *per-region 5-horizon* prediction at all (some foundation models are univariate; some don't extrapolate 5 weeks)

Filter every candidate to one of:
- **GREEN**: feasible, ≥1% expected lift, ≤8h implementation. Promote to Phase 3.
- **YELLOW**: feasible but uncertain lift, or feasible but >8h cost. Keep in reserve.
- **RED**: blocked by rules, by compute, or by output-shape mismatch.

## Phase 3 — Ranked candidate shortlist (5 ± 2 items)

For each Phase-2 GREEN candidate, write a 1-paragraph entry:
- **Mechanism**: what the technique does, in one sentence.
- **Why it could break the ceiling**: explicit anti-correlation argument vs ext150 (the lesson from `feedback_plan_v7_3upload_retro.md`).
- **Expected ΔMAE**: numeric estimate with reasoning (cite the literature lift number, discount appropriately).
- **Implementation cost**: hours, GPU-hours, data download GB.
- **Falsifiable mini-experiment**: a < 4-hour pilot that would confirm or kill the candidate before any Kaggle upload is spent.

## Phase 4 — Decision and next-plan trigger

Write a final 1-page decision document at `reports/plan_v8_decision.md`:
- Top 1–3 candidates with the rationale chain.
- Whether the team should commit to a Plan v9 (implementation) or pause DM 114 entirely (e.g. if rules forbid external data AND no category B/C/D candidate is plausibly worth the time).
- If a Plan v9 is recommended, list its first 3 experiments with falsifiable hypotheses (mirror of v6/v7 structure).

## Critical files

**Read-only (no edits in Plan v8):**
- `data/train.csv`, `data/test.csv`, `data/sample_submission.csv` — to verify field schemas and region/date coverage when assessing external-data compatibility.
- `/home/raiso/.claude/projects/-home-raiso-DM-114-FinalProject-claude/memory/*.md` — the failure-mode memory must inform Phase 2's feasibility filter.
- `reports/strategy_2026-05-11_redo.md` — the historical strategy notes for context.

**To be created during Plan v8:**
- `reports/plan_v8_cat_a_external_data.md`
- `reports/plan_v8_cat_b_pretrained.md`
- `reports/plan_v8_cat_c_paradigms.md`
- `reports/plan_v8_cat_d_kaggle.md`
- `reports/plan_v8_decision.md`

## What NOT to do in Plan v8

1. **No Kaggle uploads.** Plan v8 is research-only; no quota burned.
2. **No model training.** Phase 3's "falsifiable mini-experiment" is for Plan v9, not for v8 itself.
3. **No re-litigating Plan v7 candidate families.** DOY/H2/v6-3leg are closed (`feedback_plan_v7_3upload_retro.md`).
4. **No new feature-engineering on the *existing* weather+score+DOY space.** That has been exhausted across v1–v7.
5. **No external data acquisition** before the rules check confirms it is allowed.

## Honest expectations for what Plan v8 unlocks

- **P(Plan v8 surfaces ≥1 GREEN candidate with expected ΔMAE > -0.02)**: ~70%. Literature on drought forecasting with satellite SPI/SPEI/NDVI commonly reports 5–15% MAE improvement over weather-only baselines; that's a reasonable prior even after filtering for DM 114 compatibility.
- **P(Plan v8 surfaces a candidate that ACTUALLY beats ext150 once implemented in v9)**: ~25%. The v7 retro showed the public-MAE landscape is unforgiving; even a literature-strong technique can have its lift evaporate via the same val→public divergence DM 114 keeps producing.
- **P(Plan v8 reveals competition rules forbid external data, collapsing category A)**: ~30% based on the user's prior comment that "all groups have broken < 0.80", which implies someone *did* find a signal we don't have — which could be either an internal-data trick we missed, or external data others did use.

## Verification

End-of-plan success criteria for Plan v8:
1. All four `reports/plan_v8_cat_*.md` files exist and each contains a non-empty filtered table.
2. `reports/plan_v8_decision.md` exists with a clear go/no-go recommendation.
3. The decision document explicitly references the competition rules verdict, the memory failure modes, and at least one citation per ranked candidate.
4. Plan v9 either exists (if go) or is explicitly skipped (if no-go).
