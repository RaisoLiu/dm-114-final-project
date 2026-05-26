# DM 114 Final Report — Three-Iteration PI Review Log

Reviewer persona: a senior CS faculty member at a top US university (CMU/MIT/Stanford/Berkeley caliber), reading the rendered PDF end-to-end on a 30-minute train ride. The reviewer has not seen any of the team's internal documents.

Scope of each iteration is set by the approved plan at `~/.claude/plans/system-instruction-you-are-working-ticklish-cocke.md`.

---

## Iteration 0 — Starting state (before any iteration)

- File: `report/DM_project_Group_3.pdf` (rebuilt with `a4paper` during Phase 0).
- Length: **2 pages**.
- Figures: 5 placeholders/synthetic plots; `fig1_periodicity.pdf` and `fig2_slope.pdf` regenerated from real data in Phase 0.
- Tables: **none**.
- References: 3 entries (Iglovikov satellite imagery, Chronos, LightGBM).
- Reproducibility check: `make phd-below075` produces a CSV that matches the uploaded `submission_phd_below075_20260522.csv` to ≤ 1 ULP per cell (IEEE-754 last-digit drift across re-runs); the report's earlier "bit-for-bit identical" claim is too strong.
- Kaggle team display name `Team 3` must be verified by the user — cannot check from SSH.

---

## Iteration 1 — Observed PI critique (read the Phase-0 PDF)

### What the PI saw
1. **Length: 2 pages.** Spec demands 5–8 pages excluding refs. Disqualifying gap.
2. **Figure 1 caption does not match its panels.** The caption advertises panels "(a) ACF peaks at 2215 days; (b) FFT confirms dominance; (c) severity is much higher in the public-matched phase; (d) example alignments." The actual panels are (a) FFT top-period histogram, (b) two example weekly score series for R1/R1001, (c) coverage bar chart of regions per period band. **No ACF, no severity-phase comparison, no alignments.** This is the single worst issue — a reader sees a caption claiming evidence the figure does not show.
3. **Figure 3 (Orthogonality) is built from hardcoded numpy arrays.** Inspection of `generate_figures.py` shows the correlation matrix is `np.array([[1.00, 0.55, 0.107, ...]])` and the scatter panel is `np.random.normal(...)`. **A PI will instantly call this fabricated** because the bottom-right curve is `0.8534 − 0.85·w + 2.1·w²` — an analytic formula, not data. Iteration 2 (statistics) must replace this with a real bootstrap from `oof_tensor.csv`.
4. **No tables.** Spec literally asks for "self-defined baselines" and "comparison with your baselines"; the report contains zero tables.
5. **No data statistics anywhere.** The reader has no sense of N (regions, anchors, samples), score distribution, train/test split sizes, or how the validation set was constructed.
6. **No per-horizon results.** The task is 5-horizon prediction; absence of per-horizon MAE is a structural omission.
7. **No ablation study.** "Three-track architecture" is asserted but the individual track contributions are not measured.
8. **No Related Work section.** Spec mandates one.
9. **No Limitations section.** A graduate-grade CS paper without limitations is suspicious.
10. **References issues**: Iglovikov is a *satellite imagery segmentation* paper — irrelevant to time-series drought forecasting. Kechyn (WaveNet Kaggle 2018) is listed in the spec but missing here. Only 3 entries total; expect 5–8 for a 5–8-page paper.
11. **Method section is ~200 words** with a single 4-item bulleted list. Spec asks for 1–3 pages including formulas/pseudocode/diagrams; this is far below threshold.
12. **Abstract contains a point estimate "correlation 0.107"** with no confidence interval; iter 2 must replace with `0.107 ± 0.027` (the 5-fold CV interval already in `data_characteristics_v1.md`).
13. **Slope law claimed in abstract but not formally defined.** What does "slope" mean numerically? The reader must wait for fig 2 to even guess.
14. **Reproducibility section ends with "matches the uploaded artifact"** — fine — but does not specify Python version, library versions, the data hash, or that float-precision drift across runs is < 1 ULP. A PI will want every reproducibility fact pinned.

### Actions taken in iter 1
- **A**: Created `reports/review_log.md` (this file).
- **B**: Rewrote `report/DM_project_Group_3.tex` to the target 5–6 pages with this final structure:
  Abstract → I. Project Summary → II. Related Work → III. Proposed Method → IV. Experiments (with T1/T2/T3 tables and Ablation) → V. Reproducibility → VI. Limitations → References.
- **C**: Embedded three real tables directly in the tex:
  - **T1** Dataset statistics (2,248 regions, 1,485,928 train rows, 233,792 validation rows, public test mean 1.2088, 6-yr cycle, OOD cutoff_age 0–730 d train / ~531 d test, score 0–5 integer).
  - **T2** Baseline ladder (5 medians from `baselines.json`; 6 LightGBM/HGB OOF MAEs from `oof_tensor.json`; ext150 anchor public 0.8534; final blend public 0.7628). **ExtraTrees row dropped** (no artefact on disk).
  - **T3** Per-horizon MAE for the 6 OOF legs, computed from `oof_tensor.csv` via `groupby('horizon')` (script-line embedded in the caption for verifiability). No ext150 or final-blend per-horizon row (artefacts not on disk).
- **D**: Replaced Fig 1 caption to match the actual panels (FFT histogram / example series / coverage bars), removed the false ACF/severity claims.
- **E**: Citation verification done in iter 1 (Plan Reviewer correction). Final reference list, every entry verified to exist:
  - Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," NeurIPS 2017.
  - Kechyn et al., "Sales forecasting using WaveNet within the framework of the Kaggle competition," arXiv:1803.04037, 2018. (Listed in the assignment spec.)
  - van den Oord et al., "WaveNet: A Generative Model for Raw Audio," arXiv:1609.03499, 2016.
  - Ansari et al., "Chronos: Learning the Language of Time Series," arXiv:2403.07815, 2024.
  - Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting," IJF 2021. (arXiv:1912.09363)
  - Svoboda et al., "The Drought Monitor," Bull. Amer. Meteor. Soc., vol. 83, no. 8, 2002, pp. 1181–1190.
  - Efron, "Bootstrap Methods: Another Look at the Jackknife," Ann. Statist. 1979 (cited in support of iter-2 CIs).
  **Dropped**: Iglovikov 2017 (satellite imagery segmentation — wrong domain).
- **F**: Softened the reproducibility claim to "matches the uploaded artifact at ≤ 1 ULP per cell"; added Python version and library list.
- **G**: Added VI. Limitations naming: synthetic-data caveat, public/private LB regime risk (qualitative only), the OOD `cutoff_age` shift.
- **H**: Built `report/DM_project_Group_3.pdf` and snapshotted to `reports/DM_project_Group_3_iter1.pdf`.

### Deferred to iter 2 (intentional)
- Fig 3 / Fig 4 still use hardcoded synthetic data; iter 2 will replace the correlation matrix and weight-curve with real bootstrap output from `oof_tensor.csv` and the 32-row gate report.
- Calibration `R² = 0.985` claim is not yet replaced with LOOCV R² (iter-2 task).
- Slope-law slope still presented as a single number; iter-2 will add the bootstrap 95 % CI.
- Ablation table is the initial 2-row form (anchor → +best lag leg). Iter 2 will extend to the 3-row form (+Track 3 CNN) once the deep-CNN validation CSV is loaded and per-horizon-merged with the OOF tensor.

### Out of scope (per user/plan)
- v17 deanonymization story remains excluded from the report and the log.
- No new Kaggle uploads.
- No model retraining.

---

## Iteration 2 — Observed PI critique (read the iter-1 PDF)

### What the PI saw
1. **Tables I and IV right-truncated in the rendered PDF.** Cell values like `2,2`, `5,4`, `1,485,9`, `{0, 1, 2, 3, 4,` and the `Δ` column header are clipped at the right margin because `\small` alone doesn't enforce column-fitting in IEEEtran's narrow conference columns.
2. **Verbatim reproducibility block runs past the right margin.** `submission_phd_below07` shown without the trailing `5_$(date...).csv`.
3. **Fig 3 is still synthetic** (correlation matrix is `np.array([[1.00, 0.55, 0.107, ...]])`, scatter is `np.random.normal`, weight curve is the analytic `0.8534 − 0.85·w + 2.1·w²`). I (iter 1) already flagged this as deferred to iter 2; PI now sees it and wants real OOF data.
4. **Calibration "R² = 0.985" claim is in-sample only.** PI wants LOOCV.
5. **"Slope law ≈ 0.42·MAD" claim is asserted without a CI** and without specifying which subset of submissions it was fit on.
6. **No Threats to Validity section** even though Limitations exists; PI wants a separate, structured threats list that says, point-by-point, "this claim, this evidence, this bound."
7. **5 pages including refs → body ≈ 4.5 pages → just under spec minimum of 5 body pages.** Iter 2 must add real content (LOOCV paragraph, bootstrap paragraph, Threats subsection) to push body to ≥ 5 pages.
8. **Lag-2215's `ρ = 0.107` is cited but not bootstrapped here.** OOF predictions for the lag-2215 leg are not on disk, so the bootstrap cannot be recomputed in this report; PI wants explicit acknowledgment of that limit rather than a fabricated bootstrap.

### Actions taken in iter 2
- **A**: Real on-disk statistics computed by an inline script on the remote:
  - Calibration gate (32-row): in-sample R² = 0.9855, RMSE = 0.0103; **LOOCV R² = 0.9472, LOOCV RMSE = 0.0196**. The 0.038 drop is modest (real signal); the doubling of RMSE is the honest uncertainty band.
  - Slope law (32-row, B=1000 bootstrap): **slope = 0.20 (95% CI [0.05, 0.26]), intercept = 0.85 (95% CI [0.82, 0.88])**. The earlier "0.42" figure does NOT reproduce on the full data, only on legacy subsets.
  - Cross-leg residual correlation for the 6 OOF GBDT legs (2{,}248 regions): tight `b91` block at ρ ≈ 0.98, tight `pl` block at ρ ≈ 0.98, cross-family ρ ≈ 0.79–0.82. Persisted to `reports/_fig3_corr_matrix.npy` for the figure generator to load.
- **B**: Replaced the synthetic Fig 3 with the real 6×6 matrix loaded from disk; new caption explicitly states the data source and the limit (lag-2215 OOF not on disk → 0.107 cited from `data_characteristics_v1.md`, not recomputed).
- **C**: Tables I, III, IV wrapped in `\resizebox{\columnwidth}{!}{...}` — fixes the right-truncation in one place.
- **D**: Verbatim reproducibility block moved into a proper float (Figure 6) with hand-broken lines inside `\begin{footnotesize}`, fixing the page-overrun.
- **E**: Replaced the calibration-gate paragraph in Section III-E with the LOOCV numbers + uncertainty band sentence; removed the older "report it in Iteration 2" hand-wave.
- **F**: Rewrote Section IV-F (slope law) with the bootstrap CI and an explicit honesty paragraph about why the all-32-row slope ($0.20$) differs from older internal $0.42$.
- **G**: Added new Section IV-G "Cross-leg residual correlation (real OOF)" with a citation discipline statement.
- **H**: Added Section IV-I "Threats to validity" with four numbered threats and the bounding evidence for each.
- **I**: Built iter2 PDF (now 6 pages, A4), snapshotted to `reports/DM_project_Group_3_iter2.pdf`.

### Deferred to iter 3 (intentional)
- Fig 2 still hardcodes the `0.42·MAD` reference line and the caption still references it; iter 3 will replace this with the bootstrap-fitted line from real data.
- The Efron `[7]` citation appears immediately after the orthogonality claim in IV-G, which a PI could mis-read as supporting the $0.107$ point estimate; iter 3 will rephrase so the citation clearly attaches to the bootstrap methodology, not the orthogonality value.
- "Listing 6" mid-text reference vs the rendered "Fig. 6" label is a cosmetic mismatch; iter 3 unifies to "Figure 6."
- Author Contributions block is still missing; iter 3 adds it.

### Out of scope (per plan)
- v17 remains entirely absent.
- No model retraining, no Kaggle uploads.
- Fig 4 panels remain illegible thumbnails of the existing `reports/plots/cv*.png` files; replacing them would require re-running the 5-fold CV plotting pipeline, which the plan explicitly excludes.

---

## Iteration 3 — Observed PI critique (read the iter-2 PDF)

### What the PI saw
1. **Fig 2's chart legend still hardcodes "Empirical slope law (0.8534 + 0.42·MAD)"** (the old line from iter 1) and the caption text reinforces it, even though the prose in Section IV-F now states the bootstrap-fitted slope is $0.20$. Two contradictory numbers in the same figure-text pair is the kind of inconsistency a PI flags immediately.
2. **The Efron `[7]` citation in Section IV-G** sits at the end of the sentence "...as quantified by the 5-fold region cross-validation reported in `reports/data_characteristics_v1.md` [7]." — which can be read as Efron being the source of the $0.107$ number rather than the source of the bootstrap procedure used elsewhere. Needs a one-clause clarification.
3. **"Listing 6"** prose vs **"Fig. 6"** rendered label. The verbatim block is a `figure` environment, so LaTeX numbered it as Fig 6; the prose still calls it Listing 6. Cosmetic but PI-visible.
4. **No Author Contributions section.** Conventional in group projects.
5. **Page count** is 6 — page 6 is half Limitations + References, so body excluding refs is ≈ 5 pages → just at the spec minimum. No further padding needed.
6. **Notation** is now consistently `ρ` throughout; no further unification required.
7. **Fig 2's pre-v18 fit shows slope 0.21**, not the older 0.42 — so the prose claim "the pre-v18 slope of $\approx 0.42$ is recovered when restricted to pre-v18 only" is also wrong on the real data. The actual restriction gives $\approx 0.21$. PI flags this as another internal inconsistency.
8. **Reproducibility check confirmed in Phase 0**: `make phd-below075` produces a CSV byte-identical to the uploaded artefact at $\leq 1$ ULP per cell; this is now correctly described in Section V-B as "≤ 1 ULP per cell" rather than "bit-for-bit identical."

### Actions taken in iter 3
- **A**: Regenerated `fig2_slope.pdf` with bootstrap-fitted lines from real data — `All-32 fit (slope 0.20)`, `Pre-v18 fit (slope 0.21)`, plus the bootstrap 95% CI shaded band. Removed the hardcoded `0.42` line entirely.
- **B**: Updated the Fig 2 caption to describe the new bootstrap lines and the shaded CI band, removing the dashed-0.42 language.
- **C**: Updated the Section IV-F prose to match the actual pre-v18 bootstrap result ($0.21$), with an honest sentence that the historical $0.42$ was an upper-tail subset and not the population fit.
- **D**: Rewrote the Efron citation sentence in Section IV-G so the `[7]` attaches to "the resampling procedure of Efron" rather than to the orthogonality value.
- **E**: Changed "Listing~\ref{lst:repro}" to "Figure~\ref{fig:repro}" so the prose matches the rendered "Fig. 6" label.
- **F**: Added an "Author Contributions" subsection at the end of Section VI, pointing to the GitHub contribution log and naming the artefact ledger from which the report's claims were drawn.
- **G**: Rebuilt the PDF, re-verified A4 page size (595.276 × 841.89 pts), 6-page count, zero PLACEHOLDER strings, `make check` confirms `Team 3`, `0.7628`, the exact submission filename, and the GitHub URL all appear.
- **H**: Snapshotted to `reports/DM_project_Group_3_iter3.pdf`; copied iter3 over the canonical `report/DM_project_Group_3.pdf`; synced all three iteration PDFs + this log into local `.context/`.

### Final QA results
- A4 page size: 595.276 × 841.89 pts ✓
- Pages: 6 total (body ≈ 5 pages, references ≈ 1 page) → within spec 5--8 body pages ✓
- All 5 figures (and Fig 6 verbatim block) render with real content ✓
- `pdftotext | grep -i placeholder` returns 0 hits ✓
- `grep -iE "v17|deanonym|cdminix|real.match|1866"` returns 0 hits ✓
- 7 references, every entry verified against arXiv / DOI / journal record ✓
- `make phd-below075` reproduces the uploaded CSV at ≤ 1 ULP per cell ✓

### Items the user must verify (cannot check from SSH)
- **Kaggle team display name is exactly `Team 3`** — confirm on the competition's Team page before the June 10 deadline.
- **GitHub repository visibility is public** and the latest `report/DM_project_Group_3.tex` + the runnable `README.md` are committed before the deadline.
- **Author names** in the Author Contributions block use the placeholder "All authors are members of Team 3" — fill in real names if the team prefers explicit attribution.

### Out of scope (per plan)
- v17 deanonymization remains entirely excluded.
- Fig 4 sub-panels remain small thumbnails of `reports/plots/cv*.png` — re-rendering them at full IEEE quality would require re-running the CV plotting pipeline, which is not in scope.
- No new Kaggle submissions made; no models retrained; no edits outside the `report/`, `reports/`, and `report/figures/` directories.

---

## Iteration 4 — 2026-05-26  (response to external PI/reviewer critique of `iter3.pdf`)

### Reviewer critique observed (paraphrased, full text in `.context/agent_reviews/iter3_review.md`)
- Title `Public-Leaderboard Multi-Track Ensembling` over-foregrounds public-LB tuning.
- ρ = 0.107 ± 0.027 used as abstract headline despite OOF predictions not preserved → reviewer demanded either re-preservation or abstract demotion.
- Table IV is public-submission trajectory, not a controlled ablation; reviewer wanted a 9-row controlled OOF ablation.
- Fig 2 caption ends at 0.7952 without bridging to the final 0.7628; Fig 3 caption says "on-diagonal blocks" (incorrect phrasing); Fig 4 panels reportedly small.
- `$(date +%Y%m%d)` in Makefile produces date-dependent submission filename → breaks reproducibility on any day != 2026-05-22.
- README lacked inline cache SHA256s, only referenced ARTIFACTS.md.
- GitHub repo at canonical URL returns 404 → publishing/reproducibility blocker.

### Actions taken
- **Title**: renamed to `Seasonal-Lag and Multi-Track Ensembling for Drought Severity Forecasting (DM~114 Kaggle)`.
- **9-row ablation regenerated from scratch**: `scripts/regen_lag_2215_oof.py` (deterministic; 9 s CPU) and `scripts/regen_ssl_oof.py` (single forward pass of `checkpoints/track1_finetuned.pt`; 1 s on GPU) produce row-aligned OOFs that join into `oof_tensor.csv` on `(row_index, region_id, horizon)`. `scripts/build_ablation_9row.py` assembles the 9 rows into `reports/ablation_9row.{csv,md}` and the new Table V in the PDF.
- **Cross-leg ρ recomputed**: `scripts/compute_cross_leg_rho.py` (region-resampled bootstrap B=1000) gives lag-2215 ρ=0.51 [0.485, 0.533], CNN ρ=0.54 [0.524, 0.562], SSL ρ=0.33 [0.311, 0.349]. **Result supersedes the prior ρ=0.107 figure** — abstract and §Cross-leg residual correlation rewritten to lead with the recomputed numbers; the Threats-to-Validity (iii) entry now documents the supersession explicitly.
- **Affine-clip framing**: Table V exposes OOF MAE 0.5325 for the no-affine blend vs 0.5766 with affine — opposite ordering to public MAE (0.7628 wins on public). This is incorporated into Threats (iv) as direct evidence that affine-and-clip is public-slice calibration, not a generalization step.
- **Fig 2 caption**: appended explicit bridge sentence to Fig 5 / Table IV explaining why the 0.7628 affine submission is excluded from the MAD plot.
- **Fig 3 caption**: "on-diagonal blocks" → "within-family blocks".
- **Fig 4 caption**: panel (b) wording updated to mark the original 0.107 ± 0.027 stability number as superseded by §Cross-leg residual correlation, panel retained as historical context.
- **Makefile**: `PHD_SUBMISSION` filename hardcoded to `submission_phd_below075_20260522.csv`; `make ablation` target added.
- **README**: inline SHA256 checksum block for 8 critical cache files; "Clean-clone reproduction" walkthrough section; explicit `--force-synthesis` clarification.
- **Verbatim code (Fig 6)**: `$(date +%Y%m%d)` → hardcoded `20260522` so the PDF matches the Makefile.
- Rebuild + checks all pass: A4, 8 pages (7+ content, ~0.3 references on page 8), `make check` ✓, `make verify-submission` ✓ (max abs diff 4.4e-16).
- iter4 PDF snapshot: `reports/DM_project_Group_3_iter4.pdf` (SHA `f1159ece...`).

### Pushbacks (reviewer suggestions declined or modified)
- **Reviewer**: rename `--force-synthesis` flag. **Pushback**: cosmetic concern; renaming would touch script + Makefile + manifest + README. Instead added one-line clarification in README that the flag does not synthesize labels.
- **Reviewer**: lag-2215 may risk "weather-only" rule violation. **Pushback**: spec p.10 weather-only constraint is about `test.csv` inference inputs only; `train.csv` explicitly provides historical scores as supervised labels, so autoregressive lag features are standard. Did however reword "pure lag-2215 d score lookup" → "autoregressive seasonal baseline at a 2215-day lag (memorised historical-label lookup over train.csv)" to pre-empt TA confusion.
- **Reviewer**: Fig 4 panels too small. **Pushback**: `figures/generate_figures.py` already lays out Fig 4 as a 2×2 grid at 7" full-text-width; rendered iter4 pages confirm panels are readable. Source-PNG resolutions (825-1305 × 588-708 px) are sufficient at the rendered scale. No change.

### Outstanding human-only blockers
- 🔴 GitHub repo `https://github.com/RaisoLiu/dm-114-finalproject` still returns 404 (unauthenticated). Remote has commits ready to push but the GitHub-side repo must be created/made public before June 10.
- 🔴 Kaggle Display Name must be exactly `Team 3`. Must be confirmed in Kaggle UI.
- ⚠️  ~80 experiment scripts referenced by the report (`scripts/analyze_data_distribution.py`, `scripts/multi_blend_grid.py`, `scripts/local_eval_gate.py`, `scripts/track*.py`, etc.) remain untracked on the remote git tree, consistent with the prior iter3 commit's narrow scope. Before the GitHub repo is "runnable per spec", the user should decide whether to commit these in a follow-up.
