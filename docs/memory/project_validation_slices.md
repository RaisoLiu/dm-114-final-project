---
name: Validation slice severity map for DM 114 drought project
description: Which `--valid-deltas` values give which target severity, and why 1460/1825 are NOT high-severity
type: project
originSessionId: e0c2f638-bb8b-4524-a4af-72207d0a6fd2
---
Synthetic-year structure in `data/train.csv` is multi-year periodic (years span 5–6 digit synthetic IDs). Empirically, target means at validation deltas:

| delta | target mean | severity |
|---:|---:|---|
| 0 | ≈ 0.34 | very low (latest-anchor; misleading) |
| 365 | ≈ 0.74 | low (the trap that hurt 2026-05-10 uploads) |
| 721, 728 | ≈ 1.13 | high (good) |
| 735 | ≈ 1.17 | high (canonical reference) |
| 742 | ≈ 1.19 | high (closest to public 1.2088) |
| 749, 756 | ≈ 1.16 | high |
| 1460 | ≈ 0.76 | low-medium (NOT public-like; misleads) |
| 1825 | ≈ 0.85 | low-medium (NOT public-like; misleads) |
| 2200 | ≈ 1.97 | very high (severity-stress only) |

**Why:** The high-severity plateau is a single ~one-month band centered around year-2-ago (deltas 721–756). The naive guess "go bigger for higher severity" is wrong because the synthetic year cycle re-enters lower-severity zones at multi-year offsets. This was discovered on 2026-05-10 during planning for the 2026-05-11 redo run.

**How to apply:** When choosing `--valid-deltas` for the gap_model trainer, use {721, 728, 735, 742, 749} as the high-severity protocol; add 365 with low weight only as a sanity slice. Do NOT use 1460 or 1825 expecting high severity. Always re-verify target mean per slice with a probe run before committing a long training to a slice set.
