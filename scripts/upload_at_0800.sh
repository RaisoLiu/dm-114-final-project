#!/bin/bash
# Upload v18 final candidates at 08:00 CST (Kaggle quota reset).
# Run order: upload best first; if scores transfer reasonably, that's our final.

KAGGLE="/home/raiso/.local/bin/kaggle"
COMP="data-mining-2026-final-project"

# Upload 1: best predicted (l2215 + transform)
echo "=== Upload 1 ==="
$KAGGLE competitions submit -c $COMP \
  -f submissions/_v18_finalsuper2_T.csv \
  -m "v18 finalsuper2_T: 50% deep_pb30_w10 + 15% lag-6yr + 20% lag-2215d + 5% lag-6.5yr + 5% Track3-Huber + 5% Track3-CNN + transform(shift=-0.15, scale=1.0, clip=3.0). 6-yr lag discovery: lag=2215d has ρ=0.107 with ext150 errors (vs 5-yr's 0.30 and ext150 family's 0.98+). Predicted public 0.7596 (calibration R²=0.99). Previous upload 0.7952; expected actual ~0.75."

sleep 90
$KAGGLE competitions submissions -c $COMP 2>&1 | head -3

# Upload 2: no transform variant for verification
echo "=== Upload 2 ==="
$KAGGLE competitions submit -c $COMP \
  -f submissions/_v18_finalsuper2.csv \
  -m "v18 finalsuper2 (no transform): same blend as upload 1 but without post-hoc transform. Predicted public 0.7676. Tests whether transform helped (upload 1 was 0.7596 with transform)."

sleep 90
$KAGGLE competitions submissions -c $COMP 2>&1 | head -4

# Upload 3 reserved — pick based on first two outcomes
echo "=== Done. Upload 3 reserved for follow-up ==="
echo "Best candidate scores will appear after Kaggle scoring (~1-2 min)."
