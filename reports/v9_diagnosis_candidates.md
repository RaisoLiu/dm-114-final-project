# Plan v9 D4 — Diagnosis-aware candidates

All variants apply the D2 score-only prediction selectively to ext150 based on the D3 per-region slope.

| Variant | Mean | MAD vs ext150 | Expected impact | Expected public |
|---|---|---|---|---|
| A_helpful_only | 0.8859 | 0.4267 | -0.02324 | 0.8302 |
| D_top1000_regions | 0.8774 | 0.4345 | -0.02216 | 0.8312 |
| C_top20pct | 0.9821 | 0.2916 | -0.01941 | 0.8340 |
| D_top500_regions | 0.9645 | 0.3145 | -0.01912 | 0.8343 |
| C_top10pct | 1.0757 | 0.1734 | -0.01369 | 0.8397 |
| D_top200_regions | 1.0904 | 0.1553 | -0.01171 | 0.8417 |
| C_top5pct | 1.1394 | 0.0975 | -0.00870 | 0.8447 |
| B_weighted | 1.1652 | 0.0720 | -0.00537 | 0.8480 |

**Caveats**:
- Estimates extrapolate from 13 historical datapoints all having `pub_delta > 0`.
- True public score depends on whether the score-only model's structural information
  matches public truth, not just historical submission deltas.
- MAD gate per Plan v9: ≤ 0.30 (Variant A is likely above; Variants C/D should be in range).
