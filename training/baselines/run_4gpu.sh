#!/usr/bin/env bash
set -euo pipefail

# Paper-ready architecture launcher. Each Python process owns one visible GPU and
# trains independent seeds serially inside that model process. Scientific budget
# and data identity are injected by `training.paper`; do not add epoch/batch/config
# overrides here.
SEEDS="2027,2028,2029,2030,2031"
OUTPUT_ROOT="training/baselines/results"

# Non-neural references. The paper dispatcher still requires a CUDA-capable
# formal runtime, but these jobs themselves do not consume meaningful GPU time.
CUDA_VISIBLE_DEVICES=0 python -m training.paper baseline \
  --track architecture --model dummy --seeds "${SEEDS}" --output-root "${OUTPUT_ROOT}"
CUDA_VISIBLE_DEVICES=0 python -m training.paper baseline \
  --track architecture --model xgboost --seeds "${SEEDS}" --output-root "${OUTPUT_ROOT}"

models=(cgcnn_controlled schnet_controlled angle_moment state_threebody)
pids=()
for gpu in 0 1 2 3; do
  echo "Launching paper architecture model=${models[$gpu]} on physical GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m training.paper baseline \
    --track architecture --model "${models[$gpu]}" --seeds "${SEEDS}" \
    --output-root "${OUTPUT_ROOT}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

CUDA_VISIBLE_DEVICES=0 python -m training.paper baseline \
  --track architecture --model painn --seeds "${SEEDS}" --output-root "${OUTPUT_ROOT}"

# Optional evaluation of the five independently trained full-ablation checkpoints.
# It is evaluation-only; set RUN_FULL_SYSTEM=1 after the full ablation seeds exist.
if [[ "${RUN_FULL_SYSTEM:-0}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES=0 python -m training.paper baseline \
    --track full-system --model ours_full --seeds "${SEEDS}" \
    --output-root "${OUTPUT_ROOT}"
fi

cat <<'EOF'
Architecture/full-system jobs finished. Do NOT create the final benchmark table yet unless all four official-upstream backbones have also completed in their pinned isolated environments.
After the official track is complete, run:
  python -m training.paper baseline-summary
EOF
