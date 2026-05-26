# DM 114 Final Project Artifact Manifest

This manifest records the artefacts needed to audit the reported final submission and PDF.
The final submission path is a cached-prediction re-blend, not a full retraining of every historical model.

Generated: 2026-05-25 11:09:29 UTC

| Path | Role | Size bytes | SHA256 | Produced by | Consumed by |
|---|---|---:|---|---|---|
| `submissions/submission_phd_below075_20260522.csv` | Uploaded final Kaggle submission; public MAE 0.7628 | 227167 | `1b7f6ab6484673901eb433c1fa69a4656648c68156253298a712117988a97540` | make phd-below075 cached re-blend | Kaggle upload and make verify-submission |
| `reports/training_menu_v1.json` | Training menu controlling the final cached blend | 3622 | `4885b0c9f003329aa1f561601521d78e0e4bbb42e53c069e0d2e18ba7a8bbd21` | scripts/analyze_data_distribution.py --force-synthesis --emit-menu | scripts/multi_blend_grid.py --fixed |
| `reports/oof_tensor.csv` | Aligned OOF predictions for retained GBDT legs | 4166125 | `2f5ed136f8106dc79222d636b37bd697bdedec5e4df5744a036419645158ed96` | scripts/build_oof_tensor.py / retained experiment cache | Table III and Fig. 3 |
| `reports/_local_eval_gate_report.csv` | 32-row public upload ledger and calibration-gate source | 72317 | `2fe0767025259be39ab6078a59ecb4215f1c413349604a8799ea0890abc0f8ab` | scripts/local_eval_gate.py / Kaggle history audit | Table IV and Fig. 2 |
| `reports/_track2_fft_peaks.csv` | Per-region FFT peak summary for periodicity analysis | 326664 | `51839f77a03c41055d96a18f5e154a844bee947632d8cdf4d14111c08d26e049` | scripts/track2_phase_eda.py | Fig. 1 |
| `reports/data_characteristics_v1.md` | Dataset statistics and retained 5-fold rho summary | 5454 | `2047ccd8be562da5ad32351c4f12546cd344a9972ed70f784be9b4f8410d6017` | scripts/analyze_data_distribution.py --emit-menu | Table I and orthogonality wording |
| `report/figures/generate_figures.py` | Report figure generator | 14984 | `1d3c21b5d9bd8b3d705361b1613fcda3e2db44d7bc0c0f21bdfb60a155c86aa2` | manual report code | cd report && make |
| `report/figures/fig1_periodicity.pdf` | Periodicity figure | 18263 | `942a99d38b0e874c2c92fac44aaa34448f02baf62a1c0d6c2fc04aa237ea9bbf` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig2_slope.pdf` | Public-ledger slope trend figure | 18521 | `d2b4510dc880bd994d38562a0d6d859b455279679fd64210e75a7282c94bdaf8` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig3_orthogonality.pdf` | OOF residual correlation figure | 15519 | `7c9c66f0a19125210e5849fdfbf4d93114a196f926920e0192d08e01c8ac5d8b` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig4_cv_generalization.pdf` | 5-fold CV diagnostic figure | 173918 | `678081f6bc4bc6c39fd4e9a54502c42927d829869cd29a5c2b44709b5b915169` | reports/plots/cv*.png via report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig5_trajectory.pdf` | Public submission trajectory figure | 18790 | `a3b3cc716ab987d2fd501ff3aa2db10213ed53357b313114d87c9e2133f1909f` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/DM_project_Group_3.pdf` | Canonical final report PDF | 337409 | `0e3c5fce71af45beaf26b3d360cb7d051ca92a381b1485bd9afacdcf27879f6c` | cd report && make | course submission |

## Verification

- `make verify-submission` rebuilds the cached final CSV to `/tmp/dm114_verify_submission.csv`, validates its schema, and compares it with `submissions/submission_phd_below075_20260522.csv`.
- `cd report && make && make check` regenerates figures and the canonical A4 report PDF.
- Kaggle API audit on 2026-05-25 showed `submission_phd_below075_20260522.csv` as COMPLETE with public MAE `0.7628`; `submission_v17_real_match.csv` was `SubmissionStatus.ERROR`.
