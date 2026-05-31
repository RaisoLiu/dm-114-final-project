# DM 114 Final Project Artifact Manifest

This manifest records the artefacts needed to audit the reported final submission and PDF.
The final submission path is a cached-prediction re-blend, not a full retraining of every historical model.

Generated: 2026-05-31 12:30:16 UTC

| Path | Role | Size bytes | SHA256 | Produced by | Consumed by |
|---|---|---:|---|---|---|
| `submissions/submission_phd_below075_20260522.csv` | Uploaded final Kaggle submission; public MAE 0.7628 | 227167 | `1b7f6ab6484673901eb433c1fa69a4656648c68156253298a712117988a97540` | make phd-below075 cached re-blend | Kaggle upload and make verify-submission |
| `reports/training_menu_v1.json` | Training menu controlling the final cached blend | 3622 | `4885b0c9f003329aa1f561601521d78e0e4bbb42e53c069e0d2e18ba7a8bbd21` | scripts/analyze_data_distribution.py --force-synthesis --emit-menu | scripts/multi_blend_grid.py --fixed |
| `reports/oof_tensor.csv` | Aligned OOF predictions for retained GBDT legs | 4166125 | `2f5ed136f8106dc79222d636b37bd697bdedec5e4df5744a036419645158ed96` | scripts/build_oof_tensor.py / retained experiment cache | Table III and Fig. 3 |
| `reports/_local_eval_gate_report.csv` | 32-row public upload ledger and calibration-gate source | 72317 | `2fe0767025259be39ab6078a59ecb4215f1c413349604a8799ea0890abc0f8ab` | scripts/local_eval_gate.py / Kaggle history audit | Table IV and Fig. 2 |
| `reports/_track2_fft_peaks.csv` | Per-region FFT peak summary for periodicity analysis | 326664 | `51839f77a03c41055d96a18f5e154a844bee947632d8cdf4d14111c08d26e049` | scripts/track2_phase_eda.py | Fig. 1 |
| `reports/data_characteristics_v1.md` | Dataset statistics and retained 5-fold rho summary | 5454 | `41b07a979d47eef8220afa1b49316b47a99cd36481cbb45b95613e00f6b5b9fc` | scripts/analyze_data_distribution.py --emit-menu | Table I and orthogonality wording |
| `report/figures/generate_figures.py` | Report figure generator | 14906 | `f5b7d22f7903b4bfcde50c7c56fe59fa48f8ba964f21b5a87a89e00b5323e27d` | manual report code | cd report && make |
| `report/figures/fig1_periodicity.pdf` | Periodicity figure | 18263 | `3e29e9903fb2996c035965471f7f6c108c98f5a98e2556d39801e46463a69753` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig2_slope.pdf` | Public-ledger slope trend figure | 18521 | `2d7c6914e2406beb7364b67c6d4b1faa9848d35d4711d944416fc978abcf2473` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig3_orthogonality.pdf` | OOF residual correlation figure | 15519 | `e217d911c97f5c6c0b26f40938767d5e28932f90b040a20ac0e67d63030b252b` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig4_cv_generalization.pdf` | 5-fold CV diagnostic figure | 107674 | `0c8be61689bdcd36eef890300355400477cb3e3f5f2bf36cd93323f4137d0daf` | reports/plots/cv*.png via report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/figures/fig5_trajectory.pdf` | Public submission trajectory figure | 18790 | `24fb613e9bc93818382ce3dd69ef54b4f175fd49350c0564809b458a351286cd` | report/figures/generate_figures.py | report/DM_project_Group_3.tex |
| `report/DM_project_Group_3.pdf` | Canonical final report PDF | 279541 | `58408865d63b17419ac5cb39c103c4d9045d2fb26cd272596539a9da6b6675b2` | cd report && make | course submission |

## Verification

- `make verify-submission` rebuilds the cached final CSV to `/tmp/dm114_verify_submission.csv`, validates its schema, and compares it with `submissions/submission_phd_below075_20260522.csv`.
- `cd report && make && make check` regenerates figures and the canonical A4 report PDF.
- Kaggle API audit on 2026-05-25 showed `submission_phd_below075_20260522.csv` as COMPLETE with public MAE `0.7628`; `submission_v17_real_match.csv` was `SubmissionStatus.ERROR`.
