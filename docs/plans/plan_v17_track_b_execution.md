# Plan v17 — Track B Execution: External Real-World Drought Data

## Goal

Push public MAE < 0.79 by integrating external real-world drought observations into the prediction. This is the only path that can break the slope law's positive bound (since all internal-feature-space candidates have empirically positive slope ≥ +0.195).

## Time budget

This is a **multi-day project**, not a single session. Estimated 3-7 days of focused effort depending on data availability.

## Stage 1 — Climate fingerprint refinement (½ day)

State: BASIC fingerprints already computed (`reports/climate_fingerprints.csv`, 47 features per region).

To add:
1. **Köppen climate classification** per region — derived from monthly tmp/precip patterns
2. **Continental vs maritime** indicator (annual tmp amplitude + precip seasonality)
3. **Lat estimation** (rough) — from tmp_amplitude + warmest_month + annual mean tmp
   - High amplitude + Aug warmest = Northern Continental (~40-60°N)
   - Low amplitude + low max = Coastal North or Tropical
4. **Standardize fingerprint scale** — center/normalize each feature for clean kNN distance

Output: `reports/climate_fingerprints_v2.csv` with Köppen + lat estimate.

## Stage 2 — Reference database matching (1-2 days)

Options for reference cities database (need INTERNET ACCESS — currently this env is offline):
1. **WorldClim v2** (https://www.worldclim.org/data/worldclim21.html) — 10' resolution monthly climate
2. **Köppen-Geiger 1-km map** (Beck et al. 2018, public) — for climate zone validation
3. **Cities1000** (geonames.org) — 150k cities globally with lat/lon
4. **NASA POWER** (https://power.larc.nasa.gov/) — climate data per coordinate

Process:
1. Download WorldClim 10' tiles (~500MB) — climate per 10' × 10' cell globally
2. For each synthetic region, find top-K (e.g. 5) closest WorldClim cells by fingerprint Euclidean distance in standardized space
3. Verify match quality: Pearson correlation between region's monthly tmp/precip and matched cell's monthly tmp/precip should be > 0.95 if match is good
4. Reject regions where best match has correlation < 0.8 (use ext150 fallback for them)

Output: `reports/region_to_worldclim.csv` — region_id × (matched_lat, matched_lon, match_score, fallback_flag)

## Stage 3 — Fetch real-world drought data (1-2 days)

For each matched (lat, lon), need historical SPEI/PDSI/NDVI.

Sources:
1. **CRU TS 4.07** (Climatic Research Unit) — monthly precipitation/temperature 1901-2022, 0.5° resolution
   - https://crudata.uea.ac.uk/cru/data/hrg/
   - 50MB monthly globally
2. **SPEI database** (https://spei.csic.es/) — Standardized Precipitation Evapotranspiration Index
3. **NDVI MOD13Q1** (NASA MODIS) — 16-day vegetation index 2000-present
4. **PDSI from CRU** — easier to compute from CRU TS

Recommendation: start with **SPEI from CRU** — most aligned with USDM-style drought score.

Process:
1. Download CRU TS 4.07 NetCDF (~2GB) — pre/tmp monthly grid
2. For each matched coordinate, extract SPEI time series 1950-2022
3. Convert SPEI ∈ [-3, +3] to drought severity [0, 5] using USDM mapping:
   - SPEI < -2: D4 (exceptional drought, score 5)
   - -2 ≤ SPEI < -1.5: D3 (extreme, score 4)
   - -1.5 ≤ SPEI < -1: D2 (severe, score 3)
   - -1 ≤ SPEI < -0.7: D1 (moderate, score 2)
   - -0.7 ≤ SPEI < -0.5: D0 (abnormal dry, score 1)
   - SPEI ≥ -0.5: no drought (score 0)

Output: `reports/region_external_spei.csv` — region_id × (date, spei, drought_score)

## Stage 4 — Temporal alignment (1-2 days, RISKIEST step)

Synthetic data uses unusual year encoding (e.g. R1 train ends 3019-12-31, R1001 train ends 23103-12-31). These are NOT real calendar dates.

Need to find the REAL TIME WINDOW that each region corresponds to.

Approach: for each region, find the time window in CRU TS (1901-2022) where weather most closely matches the region's training data.

1. Compute monthly aggregates of synthetic region weather (tmp/prec) over its 15-year train period
2. Slide a 15-year window through CRU TS at matched coordinate
3. Find window with maximum Pearson correlation (or minimum RMSE) on monthly weather
4. Verify match: correlation > 0.95 expected for true alignment

Output: `reports/region_temporal_alignment.csv` — region_id × (real_start_date, real_end_date, alignment_score)

## Stage 5 — Build prediction candidate (½ day)

For each test region's prediction date (delta_h from synthetic train_end):
1. Look up matched_real_date = matched_start + delta_h
2. Extract real_drought_score at matched_lat/lon, matched_real_date from SPEI database
3. Use this as raw_external_pred[region, horizon]
4. Blend: candidate = α × external_pred + (1 - α) × ext150
5. Sweep α and find optimal on OOF (using OOF subset where matching is reliable)

Output: `submissions/_v17_external_drought_blend.csv`

## Stage 6 — Upload + verify (½ day)

Upload top-3 candidates from various α blends. Verify slope on public.

## Critical risks / fail modes

1. **No reference database access** (offline environment) — Stage 2 dead in water
2. **Climate fingerprint matching is ambiguous** — many real cities match each synthetic region. Pick wrong one → wrong drought data.
3. **Temporal alignment fails** — synthetic data might be sampled from MULTIPLE real cities/dates, not a single one
4. **External data doesn't match synthetic patterns** — synthetic noise or processing differs from real
5. **Slope law still applies** — external data is just another candidate; its delta from ext150 might still have positive slope

If risks 1-4 hit: pivot to "synthetic data structure exploitation" (cycle-phase based on global 6-yr cycle we identified).

## Decision points

- **DP1 (after Stage 1)**: Are fingerprint refinements yielding clear climate zones? If no clear structure, abort to backup plan.
- **DP2 (after Stage 2)**: Are matches > 0.9 correlation for > 50% of regions? If no, abort.
- **DP3 (after Stage 4)**: Is temporal alignment finding > 0.95 correlation for > 50% of regions? If no, the synthetic data isn't a direct sample of real city data, and we need to give up on this approach.

## Today's actionable items (Stage 1 only)

In the remaining session today, I can do:
1. Verify climate fingerprint structure — confirm features look like real cities
2. Köppen classification per region
3. Cluster regions into climate zones (K-means)
4. Compute per-cluster OOF bias correction (more robust than per-region)
5. Test as candidate on OOF

This is essentially **Track B Stage 1 + per-cluster-bias correction** as a quick first try. Expected public: not <0.79 (still internal data), but informative.

## Next session priorities

- Stage 2 requires INTERNET ACCESS for WorldClim download
- User needs to confirm access and possibly download manually
- Or: explore alternative approach via existing OSM/free climate database
