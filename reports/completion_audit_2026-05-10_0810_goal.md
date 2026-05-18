# Completion Audit: 2026-05-10 08:10 Upload Goal

## Objective

By **2026-05-10 08:10 Asia/Taipei**, upload three Kaggle prediction files and obtain a Kaggle public MAE below `0.8`.

## Success Criteria

| Requirement | Evidence | Status |
| --- | --- | --- |
| Three concrete upload files selected | `reports/strategy_2026-05-10_0810_three_uploads.md` lists the three files and upload order | Prepared |
| Files exist locally | `bash scripts/submit_three_20260510_0810.sh --dry-run` validated all three current queue files | Prepared |
| Files pass local submission format validation | Dry-run output validated row count, prediction columns, range, and mean for all three current queue files | Prepared |
| Checksums recorded | Dry-run output printed SHA-256 for all three current queue files | Prepared |
| Machine-readable manifest exists | `reports/submit_manifest_20260510_0810.json` records target time, commands, final queue, messages, checksums, and prediction statistics | Verified |
| Manifest matches executable queue | `PYTHONPATH=src .venv/bin/python scripts/verify_0810_manifest.py` verified manifest, submit script arrays, file checksums, row counts, and prediction stats | Verified |
| Safe upload procedure exists | `scripts/submit_three_20260510_0810.sh` defaults to `--dry-run`; requires `--execute` to submit | Prepared |
| Early accidental upload is guarded | `bash scripts/submit_three_20260510_0810.sh --execute` before target exits with code `3` and prints "Refusing to execute before target time" without submitting | Verified |
| Optional timed execution wrapper exists | `scripts/wait_until_submit_20260510_0810.sh --dry-run` validates immediately without waiting or upload; `--execute` waits until `2026-05-10 08:10:00 Asia/Taipei` before calling the submit script | Verified |
| Resilient quota retry runner exists | `scripts/resilient_submit_20260510_0810.sh --dry-run` verifies the manifest, validates the timed submit path without upload, and documents execute-mode retry after `2026-05-10 09:36 Asia/Taipei` from the first missing manifest item | Verified |
| Execution log will be preserved | Script writes `reports/submit_three_20260510_0810_<mode>_<timestamp>.log` | Prepared |
| Failed submission handling is safe | `--execute` stops immediately if any `submit_kaggle.py` call fails, preserving remaining queued submissions | Prepared |
| Resume after partial success is supported | `scripts/submit_three_20260510_0810.sh` accepts `--start-at 2` or `--start-at 3` to skip already submitted queue items | Prepared |
| Public score parser works | Local parser extracted scores `[0.8921, 0.8773, 0.8804, 1.0839, 1.0884, 1.2088]` and best `0.8773` from Kaggle CLI output | Verified |
| Submission quota risk surfaced | Dry-run preflight warns that three 2026-05-09 submissions are still visible in the last 24h if Kaggle timestamps are UTC | Verified |
| Current public baseline known | Kaggle list shows current best public score `0.8773` from `submission_blackout91_blend_cal02_fast64_gpu_shift35.csv` | Verified |
| New hybrid candidate evidence recorded | `reports/postprocess_hybrid_qsharp_diagnostics.json` shows diagnostic validation MAE `0.277918`, better than hsharp `0.287728` and hqmap `0.285052` | Verified |
| Anchor split risk checked | `reports/postprocess_anchor_split_diagnostics.json` shows fixed `hybrid_qsharp`, `hsharp`, and `hqmap` all improve both latest and older anchor splits; `hsharp` is most robust under split-to-split tuning | Verified |
| Full qmap public-mean shot checked | `reports/full_qmap_anchor_split_diagnostics.json` shows full qmap to known validation mean improves both anchor splits; public all-zero score gives the analogous public mean signal | Verified |
| Script syntax and tests pass | `bash -n` passed for shell scripts; `.venv/bin/python -m py_compile scripts/verify_0810_manifest.py` passed; `.venv/bin/python -m pytest -q` passed `4` tests | Verified |
| Three uploads actually made by deadline | Three uploads were completed after the target time, because execution was started at `2026-05-10 09:49 Asia/Taipei` | Not achieved |
| Three uploads actually made after user approval | Kaggle refs `52499764`, `52499781`, and `52499787` completed successfully | Achieved |
| At least one public score below `0.8` | New public scores were `0.9013`, `0.9490`, and `1.0352`; best visible public score remains `0.8773` | Not achieved |

## Selected Upload Queue

1. `submissions/submission_post_publicbest_hsharp_opt_mkeep.csv`
2. `submissions/submission_post_publicbest_hybrid_qsharp_opt_mkeep.csv`
3. `submissions/submission_post_publicbest_qmap100.csv`

## Dry-Run Evidence

Command:

```bash
bash scripts/submit_three_20260510_0810.sh --dry-run
```

Validated:

```text
submissions/submission_post_publicbest_hsharp_opt_mkeep.csv
Rows: 2,248
Prediction range: 0.043417 to 5.000000
Prediction mean: 1.233909

submissions/submission_post_publicbest_hybrid_qsharp_opt_mkeep.csv
Rows: 2,248
Prediction range: 0.134191 to 5.000000
Prediction mean: 1.233909

submissions/submission_post_publicbest_qmap100.csv
Rows: 2,248
Prediction range: 0.000000 to 5.000000
Prediction mean: 1.208808
```

Checksums:

```text
d54d71c3d8f8371897cb98475fa73eba370c338fe42b6ef9ac5ecf332a9ecc85  submissions/submission_post_publicbest_hsharp_opt_mkeep.csv
999e4d6c2599cb86510c60aef64ec9cfae6192a9f3a91a1cad5267a9de770c39  submissions/submission_post_publicbest_hybrid_qsharp_opt_mkeep.csv
0ab358610d1cdb30adbda5eb4a852b676cb16ec86fb27b0820a7a6cd6eee2a15  submissions/submission_post_publicbest_qmap100.csv
```

Latest dry-run log:

```text
reports/submit_three_20260510_0810_dry-run_20260509_223822.log
```

## Actual Upload Result

Execution started after the target deadline, following user approval on `2026-05-10 09:49 Asia/Taipei`.

New submissions:

```text
ref       fileName                                                 submitted at UTC              public
52499764  submission_post_publicbest_hsharp_opt_mkeep.csv          2026-05-10 01:50:01.857000   0.9013
52499781  submission_post_publicbest_hybrid_qsharp_opt_mkeep.csv   2026-05-10 01:51:07.337000   0.9490
52499787  submission_post_publicbest_qmap100.csv                   2026-05-10 01:51:39.193000   1.0352
```

Execution logs:

```text
reports/resilient_submit_20260510_0810_20260510_094955.log
reports/submit_three_20260510_0810_execute_20260510_094955.log
reports/submit_three_20260510_0810_execute_20260510_095102.log
```

## Current Kaggle State

Recent successful submissions:

```text
submission_post_publicbest_qmap100.csv                     public 1.0352
submission_post_publicbest_hybrid_qsharp_opt_mkeep.csv     public 0.9490
submission_post_publicbest_hsharp_opt_mkeep.csv            public 0.9013
submission_blackout91_blend_valid1_fast64_gpu_shift35.csv  public 0.8921
submission_blackout91_blend_cal02_fast64_gpu_shift35.csv   public 0.8773
submission_blackout91_blend_cal02_fast64_gpu_shift25.csv   public 0.8804
submission_gap_zero_fast64.csv                             public 1.0839
submission_last_train_score.csv                            public 1.0884
submission_global_median.csv                               public 1.2088
```

## Audit Result

The preparation deliverables were complete:

- Strategy document exists.
- Upload script exists.
- Three files are selected, present, validated, and checksummed.
- Manifest and verifier exist and match the executable queue.

The upload execution deliverable is complete:

- All three selected files were submitted and reached `SubmissionStatus.COMPLETE`.

The full competition objective is **not complete**:

- The uploads were completed after the `2026-05-10 08:10 Asia/Taipei` target time.
- No public score below `0.8` was obtained.
- Best visible public score remains `0.8773` from `submission_blackout91_blend_cal02_fast64_gpu_shift35.csv`.

Post-upload forensic follow-up:

- `reports/post_upload_forensic_2026-05-10.md` records the failed public-feedback pattern and the validation/public distribution mismatch.
- `scripts/train_gap_model.py` now supports `--valid-deltas` for explicit public-like validation offset experiments.
- `scripts/train_gap_model.py` now parses synthetic-date months correctly for region-month weather baselines; the previous fixed-position month slice was wrong for wide synthetic years.
- `scripts/submit_three_20260510_0810.sh` was hardened to avoid `head`/`pipefail` premature exits while polling Kaggle submissions.
- The best not-yet-uploaded local candidate after the month fix is `submissions/submission_blackout91_blend_cal02_fast64_gpu_monthfix_shift30.csv`; it is validated locally but not uploaded.
