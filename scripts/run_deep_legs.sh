#!/bin/bash
cd /home/raiso/DM_114_FinalProject_claude
PYTHON=/home/raiso/DM_114_FinalProject/.venv/bin/python
LOG_DIR=logs/deep_training
mkdir -p "$LOG_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] runner started" > "$LOG_DIR/runner.log"

for arch in cnn lstm trans; do
  for seed in 114 271828; do
    TAG="${arch}_s${seed}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $TAG" >> "$LOG_DIR/runner.log"
    if "$PYTHON" scripts/train_deep_model.py \
        --arch "$arch" --seed "$seed" \
        --epochs 100 --batch 512 --device cuda \
        --train-samples-per-region 128 --n-folds 5 \
        --valid-deltas 728,735,742 --gap-mode blackout91 \
        --output "submissions/submission_deep_${TAG}.csv" \
        --validation-pred-output "reports/deep_${TAG}_validation_predictions.csv" \
        --report-output "reports/deep_${TAG}.json" \
        > "$LOG_DIR/${TAG}.log" 2>&1; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  $TAG" >> "$LOG_DIR/runner.log"
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL  $TAG (rc=$?)" >> "$LOG_DIR/runner.log"
    fi
  done
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] runner complete" >> "$LOG_DIR/runner.log"
