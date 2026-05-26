# Controlled OOF ablation (iter4)

All rows computed on the same OOF tensor (33720 rows; 6,744 anchors × 5 horizons; 5-fold region-CV).

| # | Configuration | OOF MAE | Public MAE | Source |
|---|---|---:|---:|---|
| 1 | Weather-only single GBDT (best of 6) | 0.3753 | — | `pred_b91_lgbm_s271828` |
| 2 | Lag-only baseline (2215 d lookup) | 0.9673 | — | `pred_lag2215` |
| 3 | Lag + per-horizon bias correction | 1.0694 | — | `pred_lag2215 + bias` |
| 4 | GBDT anchor (6-leg b91+pl blend) | 0.3910 | 0.8534 | `mean(gbdt_cols)` |
| 5 | GBDT + lag (A+C renormalized) | 0.6840 | — | `0.444*A + 0.556*C` |
| 6 | GBDT + lag + CNN | 0.5389 | 0.8017 | `weighted A+C+B_cnn` |
| 7 | GBDT + lag + CNN + SSL (full deep) | 0.5523 | 0.7952 | `weighted A+C+B` |
| 8 | Full blend, NO affine/clip | 0.5325 | — | `0.2A + 0.45B + 0.25C + 0.1D~A` |
| 9 | Full blend + affine + clip (final) | 0.5766 | 0.7628 | `affine(shift=-0.16,scale=0.98,clip<=3.0)` |
