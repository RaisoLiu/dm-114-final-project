#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
RETRY_LOCAL="2026-05-10 09:36:00"
TARGET_TZ="Asia/Taipei"
MANIFEST="reports/submit_manifest_20260510_0810.json"

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/resilient_submit_20260510_0810.sh [--dry-run|--execute]

Dry-run mode verifies the manifest and validates the timed submit path without upload.
Execute mode waits until 2026-05-10 08:10 Asia/Taipei, attempts the queue, and if that
attempt fails, waits until 2026-05-10 09:36 Asia/Taipei before resuming from the first
manifest file not visible in Kaggle submissions.
EOF
}

case "$MODE" in
  --dry-run|--execute)
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 2
    ;;
esac

export PYTHONPATH=src

next_start_at() {
  .venv/bin/python - "$MANIFEST" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
queue = manifest["queue"]
env = os.environ.copy()
token_path = ".kaggle/access_token"
if os.path.exists(token_path):
    with open(token_path, encoding="utf-8") as f:
        env["KAGGLE_API_TOKEN"] = f.read().strip()
out = subprocess.check_output(
    [".venv/bin/kaggle", "competitions", "submissions", "-c", manifest["competition"]],
    text=True,
    env=env,
)
visible = out.splitlines()
for item in queue:
    basename = Path(item["path"]).name
    complete_or_pending = any(
        basename in line and ("SubmissionStatus.COMPLETE" in line or "SubmissionStatus.PENDING" in line)
        for line in visible
    )
    if not complete_or_pending:
        print(item["order"])
        break
else:
    print(4)
PY
}

wait_until_retry_time() {
  local retry_epoch now_epoch remaining sleep_for
  retry_epoch="$(TZ="$TARGET_TZ" date -d "$RETRY_LOCAL" +%s)"
  now_epoch="$(date +%s)"
  if (( now_epoch >= retry_epoch )); then
    echo "Retry time has already passed; resuming now."
    return 0
  fi
  remaining=$((retry_epoch - now_epoch))
  echo "Waiting ${remaining}s until retry time $(TZ="$TARGET_TZ" date -d "@$retry_epoch" --iso-8601=seconds) ($TARGET_TZ)."
  while (( remaining > 0 )); do
    sleep_for=$(( remaining < 60 ? remaining : 60 ))
    sleep "$sleep_for"
    now_epoch="$(date +%s)"
    remaining=$((retry_epoch - now_epoch))
    if (( remaining > 0 )); then
      echo "Still waiting for retry: ${remaining}s remaining at $(date --iso-8601=seconds)"
    fi
  done
}

echo "Mode: $MODE"
echo "Manifest: $MANIFEST"
PYTHONPATH=src .venv/bin/python scripts/verify_0810_manifest.py --manifest "$MANIFEST"

if [[ "$MODE" == "--dry-run" ]]; then
  echo
  echo "Dry-run: validating timed submit path; no waiting and no upload."
  bash scripts/wait_until_submit_20260510_0810.sh --dry-run
  echo
  echo "Dry-run complete. Execute mode would wait for 08:10 and retry after 09:36 only if needed."
  exit 0
fi

set +e
bash scripts/wait_until_submit_20260510_0810.sh --execute
first_status=$?
set -e
if [[ "$first_status" -eq 0 ]]; then
  echo "Initial 08:10 submit attempt completed successfully."
  exit 0
fi

echo "Initial submit attempt failed with exit code $first_status."
start_at="$(next_start_at)"
if [[ "$start_at" == "4" ]]; then
  echo "All manifest files are visible in Kaggle submissions; not retrying."
  exit 0
fi
if [[ "$start_at" != "1" && "$start_at" != "2" && "$start_at" != "3" ]]; then
  echo "Unexpected next start index: $start_at" >&2
  exit 4
fi

wait_until_retry_time
echo "Retrying from queue item $start_at."
bash scripts/submit_three_20260510_0810.sh --execute --start-at "$start_at"
