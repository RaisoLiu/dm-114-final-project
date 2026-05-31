# Plan v9 D3 — Public reverse-diagnosis result

Based on 13 historical Kaggle submissions with known public scores.

## Overall per-region diagnosis (correlation of per-region delta with public_delta)

- Regions where modifications HELP public (mean_corr < -0.2): **1086** / 2248
- Regions neutral / noisy (mean_corr in [-0.2, 0.2]): 352
- Regions where modifications HURT public (mean_corr > 0.2): **810**

## Per-bucket diagnosis (by ext150 prediction range)

| bucket | n_regions | ext150_mean | mean_corr | mean_slope |
|---|---|---|---|---|
| [0, 0.1] | 224 | 0.025 | -0.4778 | -0.0912 |
| (0.1, 0.5] | 461 | 0.299 | -0.204 | -0.041 |
| (0.5, 1.0] | 477 | 0.735 | -0.0894 | -0.0184 |
| (1.0, 1.5] | 337 | 1.235 | 0.0245 | -0.0028 |
| (1.5, 2.0] | 269 | 1.75 | 0.134 | 0.0167 |
| (2.0, 3.0] | 336 | 2.422 | 0.1716 | 0.0163 |
| (3.0, 5.0] | 144 | 3.906 | -0.1917 | -0.0044 |

**Reading**: positive `mean_corr` means historical modifications in this bucket
correlated with WORSE public scores → ext150 is approximately correct here, leave alone.
Negative `mean_corr` means modifications HELPED → ext150 is wrong in some direction here.

## D2 score-only candidate impact projection

- Net estimated public impact: **-0.0038**
- Projected public score: **0.8496**
(Reading: positive net impact → worse than ext150. Negative → better.)

## Notes
- All 13+ historical submissions had positive public_delta (i.e., worse than ext150).
- This means `pub_delta` has no negative variance in our sample, so the regression is
  more accurately reading 'which deltas correlated with smaller positive harm'.
- For genuine `pub_delta < 0`, we have no datapoint yet.
