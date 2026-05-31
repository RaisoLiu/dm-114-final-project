# DM 114 Final Project Artifact Manifest

This manifest records the artefacts needed to audit the reported final submission and PDF.
The final submission path is a cached-prediction re-blend, not a full retraining of every historical model.

Generated: 2026-05-31 13:01:01 UTC

| Path | Role | Size bytes | SHA256 | Produced by | Consumed by |
|---|---|---:|---|---|---|
| `submissions/submission_phd_below075_20260522.csv` | Uploaded final Kaggle submission; public MAE 0.7628 | 227167 | `1b7f6ab6484673901eb433c1fa69a4656648c68156253298a712117988a97540` | make phd-below075 cached re-blend | Kaggle upload and make verify-submission |
| `reports/training_menu_v1.json` | Training menu controlling the final cached blend | 3622 | `4885b0c9f003329aa1f561601521d78e0e4bbb42e53c069e0d2e18ba7a8bbd21` | scripts/analyze_data_distribution.py --force-synthesis --emit-menu | scripts/multi_blend_grid.py --fixed |
| `reports/oof_tensor.csv` | Aligned OOF predictions for retained GBDT legs | 4166125 | `2f5ed136f8106dc79222d636b37bd697bdedec5e4df5744a036419645158ed96` | scripts/build_oof_tensor.py / retained experiment cache | Table III and Fig. 3 |
| `reports/_local_eval_gate_report.csv` | 32-row public upload ledger and calibration-gate source | 72317 | `2fe0767025259be39ab6078a59ecb4215f1c413349604a8799ea0890abc0f8ab` | scripts/local_eval_gate.py / Kaggle history audit | Table IV and Fig. 2 |
| `reports/_track2_fft_peaks.csv` | Per-region FFT peak summary for periodicity analysis | 326664 | `51839f77a03c41055d96a18f5e154a844bee947632d8cdf4d14111c08d26e049` | scripts/track2_phase_eda.py | Fig. 1 |
| `reports/data_characteristics_v1.md` | Dataset statistics and retained 5-fold rho summary | 5454 | `e5579d1deb0705e64f943985f3b4428b340dd11212e74b4fb9b4054a8a0d3032` | scripts/analyze_data_distribution.py --emit-menu | Table I and orthogonality wording |
| `report/figures/generate_figures.py` | Report figure generator | 18440 | `9e2f28d824a1ab730c049972a14827b7371ead41ec05a44fb0fa4671f1ed2da5` | manual report code | cd report && make |
| `report/figures/fig1_periodicity.pdf` | Periodicity figure | 18263 | `726601d2e12f2081f7839b313e92ee8cfbede4d2a74844e84bb48ca39753393a` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig2_slope.pdf` | Public-ledger slope trend figure | 18521 | `617ef064f6e5f1fc3762b2be446ba2dccc4a9bc0e6eef6450a22b48d08125f1b` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig3_orthogonality.pdf` | OOF residual correlation figure | 15519 | `3d9757900459621e293fb687711848e14d28bdbc4580346310df51cb642afa9c` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig4_cv_generalization.pdf` | 5-fold CV diagnostic figure | 107674 | `776df5bc814449ec267d80e4831b19fc424160bec1a6217fbfbfe74cdd5a121c` | reports/plots/cv*.png via report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig5_trajectory.pdf` | Public submission trajectory figure | 18790 | `402d660e45590d37e9d04ba15ada8ca3fdcc5123e4ac66a8f5b87185b40c177c` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/DM_project_Group_3.pdf` | Canonical final report PDF | 301250 | `0c7ea1cb1d5a0a3863db32f904f7cd05eea19bbd4eae672d3f6a08fcdd4f168e` | cd report && make | course submission |

## Verification

- `make verify-submission` rebuilds the cached final CSV to `/tmp/dm114_verify_submission.csv`, validates its schema, and compares it with `submissions/submission_phd_below075_20260522.csv`.
- `cd report && make && make check` regenerates figures and the canonical A4 report PDF.
- Kaggle API audit on 2026-05-25 showed `submission_phd_below075_20260522.csv` as COMPLETE with public MAE `0.7628`; `submission_v17_real_match.csv` was `SubmissionStatus.ERROR`.
