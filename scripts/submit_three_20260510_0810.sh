#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
START_AT="1"
FORCE_BEFORE_TARGET="0"
TARGET_LOCAL="2026-05-10 08:10:00"
TARGET_TZ="Asia/Taipei"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--execute" ]]; then
  echo "Usage: bash scripts/submit_three_20260510_0810.sh [--dry-run|--execute] [--start-at 1|2|3] [--force-before-target]" >&2
  exit 2
fi
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-at)
      START_AT="${2:-}"
      shift 2
      ;;
    --force-before-target)
      FORCE_BEFORE_TARGET="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash scripts/submit_three_20260510_0810.sh [--dry-run|--execute] [--start-at 1|2|3] [--force-before-target]" >&2
      exit 2
      ;;
  esac
done
if [[ "$START_AT" != "1" && "$START_AT" != "2" && "$START_AT" != "3" ]]; then
  echo "--start-at must be 1, 2, or 3." >&2
  exit 2
fi

if [[ "$MODE" == "--execute" && "$FORCE_BEFORE_TARGET" != "1" ]]; then
  target_epoch="$(TZ="$TARGET_TZ" date -d "$TARGET_LOCAL" +%s)"
  now_epoch="$(date +%s)"
  if (( now_epoch < target_epoch )); then
    echo "Refusing to execute before target time." >&2
    echo "Target: $(TZ="$TARGET_TZ" date -d "@$target_epoch" --iso-8601=seconds) ($TARGET_TZ)" >&2
    echo "Now:    $(date --iso-8601=seconds)" >&2
    echo "Use scripts/wait_until_submit_20260510_0810.sh --execute to wait safely." >&2
    echo "Use --force-before-target only if you intentionally want to submit early." >&2
    exit 3
  fi
fi

export PYTHONPATH=src

mkdir -p reports
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="reports/submit_three_20260510_0810_${MODE#--}_${RUN_STAMP}.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "Mode: $MODE"
echo "Start at queue item: $START_AT"
echo "Started at: $(date --iso-8601=seconds)"
echo "Log: $LOG_PATH"

fetch_kaggle_submissions() {
  KAGGLE_API_TOKEN="$(cat .kaggle/access_token)" .venv/bin/kaggle competitions submissions \
    -c data-mining-2026-final-project
}

print_kaggle_submissions_head() {
  local limit="${1:-12}"
  local submissions
  submissions="$(fetch_kaggle_submissions)"
  printf '%s\n' "$submissions" | sed -n "1,${limit}p"
}

FILES=(
  "submissions/submission_post_publicbest_hsharp_opt_mkeep.csv"
  "submissions/submission_post_publicbest_hybrid_qsharp_opt_mkeep.csv"
  "submissions/submission_post_publicbest_qmap100.csv"
)

MESSAGES=(
  "0810 upload 1 public-best horizon-sharpen keepmean"
  "0810 upload 2 public-best hybrid qmap-sharpen keepmean"
  "0810 upload 3 public-best full-qmap public-mean"
)

echo "Validating queued submissions..."
for file in "${FILES[@]}"; do
  .venv/bin/python scripts/validate_submission.py "$file"
done

echo
echo "Submission checksums:"
sha256sum "${FILES[@]}"

echo
echo "Recent Kaggle submissions:"
if [[ -f .kaggle/access_token ]]; then
  print_kaggle_submissions_head 12
else
  echo ".kaggle/access_token not found; Kaggle CLI may require separate auth."
fi

echo
echo "Submission quota preflight:"
.venv/bin/python - <<'PY'
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone

env = os.environ.copy()
token_path = ".kaggle/access_token"
if os.path.exists(token_path):
    with open(token_path, encoding="utf-8") as f:
        env["KAGGLE_API_TOKEN"] = f.read().strip()
out = subprocess.check_output(
    [".venv/bin/kaggle", "competitions", "submissions", "-c", "data-mining-2026-final-project"],
    text=True,
    env=env,
)
now = datetime.now(timezone.utc).replace(tzinfo=None)
recent = []
for line in out.splitlines():
    m = re.search(r"\s(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)\s+", line)
    if not m:
        continue
    try:
        submitted = datetime.fromisoformat(m.group(1))
    except ValueError:
        continue
    if now - submitted < timedelta(hours=24):
        recent.append((submitted, line.strip()))
print(f"Completed/submitted rows visible in last 24h if Kaggle timestamps are UTC: {len(recent)}")
for submitted, row in recent[:5]:
    print(f"  {submitted.isoformat(sep=' ')}  {row[:120]}")
if len(recent) >= 3:
    print("WARNING: If the competition uses a rolling 24h limit of 3 submissions, quota may not be available yet.")
PY

if [[ "$MODE" == "--dry-run" ]]; then
  echo
  echo "Dry run complete. To submit these three files at/after the target time, run:"
  echo "  bash scripts/submit_three_20260510_0810.sh --execute"
  echo "Or start the timed wrapper before the target time:"
  echo "  bash scripts/wait_until_submit_20260510_0810.sh --execute"
  exit 0
fi

poll_scores() {
  echo
  echo "Polling Kaggle submission status..."
  for attempt in $(seq 1 18); do
    local submissions
    echo "Poll attempt $attempt at $(date --iso-8601=seconds)"
    submissions="$(fetch_kaggle_submissions)"
    printf '%s\n' "$submissions" | sed -n '1,12p'
    if grep -q "SubmissionStatus.PENDING" <<<"$submissions"; then
      sleep 20
    else
      break
    fi
  done
}

check_target_score() {
  .venv/bin/python - <<'PY'
import os
import re
import subprocess
import sys

env = os.environ.copy()
token_path = ".kaggle/access_token"
if os.path.exists(token_path):
    with open(token_path, encoding="utf-8") as f:
        env["KAGGLE_API_TOKEN"] = f.read().strip()
out = subprocess.check_output(
    [".venv/bin/kaggle", "competitions", "submissions", "-c", "data-mining-2026-final-project"],
    text=True,
    env=env,
)
scores = []
for line in out.splitlines():
    match = re.search(r"SubmissionStatus\.COMPLETE\s+([0-9]+(?:\.[0-9]+)?)", line)
    if match:
        scores.append(float(match.group(1)))
if not scores:
    print("No completed public scores found yet.")
    sys.exit(0)
best = min(scores)
print(f"Best public score visible now: {best:.4f}")
if best < 0.8:
    print("TARGET_REACHED_BELOW_0.8")
PY
}

echo
echo "Submitting queued files..."
for i in "${!FILES[@]}"; do
  item_number=$((i + 1))
  if (( item_number < START_AT )); then
    echo "Skipping queued file $item_number because --start-at $START_AT was requested: ${FILES[$i]}"
    continue
  fi
  if ! .venv/bin/python scripts/submit_kaggle.py "${FILES[$i]}" --message "${MESSAGES[$i]}"; then
    echo
    echo "Submission failed for ${FILES[$i]}."
    echo "Stopping immediately so remaining queued submissions are not attempted."
    echo "If this was a quota error, wait for quota reset before retrying."
    exit 1
  fi
  poll_scores
  check_target_score
done

echo
echo "Final Kaggle submission list:"
print_kaggle_submissions_head 12
check_target_score

echo
echo "Finished at: $(date --iso-8601=seconds)"
echo "Log saved to: $LOG_PATH"
