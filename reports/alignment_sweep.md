# Plan v9 D1 — Alignment sweep result

**Test sample**: 2248 regions with ≥6 historical weekly scores in train.csv.

## T1 — ext150 per-horizon means

| Horizon | Mean | Std |
|---|---|---|
| pred_week1 | 1.3504 | 1.1686 |
| pred_week2 | 1.3070 | 1.1335 |
| pred_week3 | 1.2670 | 1.0929 |
| pred_week4 | 1.1548 | 1.0198 |
| pred_week5 | 1.0538 | 0.9672 |
| **Overall** | **1.2266** | — |

## T2 — Forward trend correlation

`corr( (pred_w5 − pred_w1), (y_last − y_5wk_back) ) = +0.0087`

- rho > 0.20 → forward alignment confirmed (persistence holds in test direction)
- rho ~ 0 → weak signal but no bug
- rho < −0.2 → reversed alignment bug suspected

## T3 — Alternative alignments

- MAE(ext150, past_5_weeks_observed) = 0.9789
- MAE(ext150, last_score_persisted) = 0.9643

If reverse << persistence by > 10%, suggests ext150 may be predicting past instead of future.

## Verdict

Inconclusive (rho_fwd near zero). Likely no off-by-one bug, but signal is weak.

No backwards-alignment bug detected.
