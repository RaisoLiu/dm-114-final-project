# DM 114 Final Project Artifact Manifest

This manifest records the artefacts needed to audit the reported final submission and PDF.
The final submission path is a cached-prediction re-blend, not a full retraining of every historical model.

Generated: 2026-05-26 07:19:23 UTC

| Path | Role | Size bytes | SHA256 | Produced by | Consumed by |
|---|---|---:|---|---|---|
| `submissions/submission_phd_below075_20260522.csv` | Uploaded final Kaggle submission; public MAE 0.7628 | 227167 | `1b7f6ab6484673901eb433c1fa69a4656648c68156253298a712117988a97540` | make phd-below075 cached re-blend | Kaggle upload and make verify-submission |
| `reports/training_menu_v1.json` | Training menu controlling the final cached blend | 3622 | `4885b0c9f003329aa1f561601521d78e0e4bbb42e53c069e0d2e18ba7a8bbd21` | scripts/analyze_data_distribution.py --force-synthesis --emit-menu | scripts/multi_blend_grid.py --fixed |
| `reports/oof_tensor.csv` | Aligned OOF predictions for retained GBDT legs | 4166125 | `2f5ed136f8106dc79222d636b37bd697bdedec5e4df5744a036419645158ed96` | scripts/build_oof_tensor.py / retained experiment cache | Table III and Fig. 3 |
| `reports/_local_eval_gate_report.csv` | 32-row public upload ledger and calibration-gate source | 72317 | `2fe0767025259be39ab6078a59ecb4215f1c413349604a8799ea0890abc0f8ab` | scripts/local_eval_gate.py / Kaggle history audit | Table IV and Fig. 2 |
| `reports/_track2_fft_peaks.csv` | Per-region FFT peak summary for periodicity analysis | 326664 | `51839f77a03c41055d96a18f5e154a844bee947632d8cdf4d14111c08d26e049` | scripts/track2_phase_eda.py | Fig. 1 |
| `reports/data_characteristics_v1.md` | Dataset statistics and retained 5-fold rho summary | 5454 | `7cb4a3bff38736dbeb69174e4e9e7526292d48290456781f8ada35c465a36708` | scripts/analyze_data_distribution.py --emit-menu | Table I and orthogonality wording |
| `report/figures/generate_figures.py` | Report figure generator | 14984 | `1d3c21b5d9bd8b3d705361b1613fcda3e2db44d7bc0c0f21bdfb60a155c86aa2` | manual report code | cd report && make |
| `report/figures/fig1_periodicity.pdf` | Periodicity figure | 18263 | `1f94bcbd102c48572bb4893439bcf468b28461cf28706061b793080996fc8202` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig2_slope.pdf` | Public-ledger slope trend figure | 18521 | `29c5b30430600352ac5d7a7f90283c6be536e948c286431a652092f8efb16dd7` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig3_orthogonality.pdf` | OOF residual correlation figure | 15519 | `a3398b9caee2d53367cd52eb640679d5b54248370ec1fd0502f048fd00e3d1e9` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig4_cv_generalization.pdf` | 5-fold CV diagnostic figure | 173918 | `5b277e84559a7c2391d028d7b160dd0469e55c487ffbac255cfbac0f86bed778` | reports/plots/cv*.png via report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig5_trajectory.pdf` | Public submission trajectory figure | 18790 | `1f9359913a718977980c43a0c08045403ba906b812f2a4b0d0d286f57b3b0cde` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/DM_project_Group_3.pdf` | Canonical final report PDF | 345935 | `b72da14cc352edb37739c34df8a710703f9de84183b980122679bddcabb109cd` | cd report && make | course submission |

## Verification

- `make verify-submission` rebuilds the cached final CSV to `/tmp/dm114_verify_submission.csv`, validates its schema, and compares it with `submissions/submission_phd_below075_20260522.csv`.
- `cd report && make && make check` regenerates figures and the canonical A4 report PDF.
- Kaggle API audit on 2026-05-25 showed `submission_phd_below075_20260522.csv` as COMPLETE with public MAE `0.7628`; `submission_v17_real_match.csv` was `SubmissionStatus.ERROR`.
