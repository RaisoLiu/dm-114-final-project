#!/usr/bin/env bash
set -euo pipefail

MODE="--dry-run"
TARGET_LOCAL="2026-05-10 08:10:00"
TARGET_TZ="Asia/Taipei"
PASSTHRU=()

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/wait_until_submit_20260510_0810.sh [--dry-run|--execute] [--target "YYYY-MM-DD HH:MM:SS"] [--timezone TZ] [--start-at 1|2|3]

Dry-run mode does not wait and does not upload; it validates the current queue immediately.
Execute mode waits until the target time, then calls scripts/submit_three_20260510_0810.sh --execute.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--execute)
      MODE="$1"
      shift
      ;;
    --target)
      TARGET_LOCAL="${2:-}"
      shift 2
      ;;
    --timezone)
      TARGET_TZ="${2:-}"
      shift 2
      ;;
    --start-at)
      PASSTHRU+=("--start-at" "${2:-}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$TARGET_LOCAL" || -z "$TARGET_TZ" ]]; then
  echo "--target and --timezone must not be empty." >&2
  exit 2
fi

target_epoch="$(TZ="$TARGET_TZ" date -d "$TARGET_LOCAL" +%s)"
target_iso="$(TZ="$TARGET_TZ" date -d "@$target_epoch" --iso-8601=seconds)"
now_epoch="$(date +%s)"

echo "Mode: $MODE"
echo "Target time: $target_iso ($TARGET_TZ)"

if [[ "$MODE" == "--dry-run" ]]; then
  echo "Dry-run mode: validating the current queue immediately; no waiting and no upload."
  exec bash scripts/submit_three_20260510_0810.sh --dry-run "${PASSTHRU[@]}"
fi

if (( now_epoch < target_epoch )); then
  remaining=$((target_epoch - now_epoch))
  echo "Waiting ${remaining}s before executing upload script..."
  while (( remaining > 0 )); do
    sleep_for=$(( remaining < 60 ? remaining : 60 ))
    sleep "$sleep_for"
    now_epoch="$(date +%s)"
    remaining=$((target_epoch - now_epoch))
    if (( remaining > 0 )); then
      echo "Still waiting: ${remaining}s remaining at $(date --iso-8601=seconds)"
    fi
  done
else
  echo "Target time has already passed; executing upload script now."
fi

exec bash scripts/submit_three_20260510_0810.sh --execute "${PASSTHRU[@]}"
