#!/usr/bin/env bash
set -euo pipefail

# Four-GPU convenience launcher for the controlled NFE baselines.
# Override any variable before invocation, for example:
#   SEEDS=2027,2028,2029 EPOCHS=120 OURS_CHECKPOINT=/path/best.pt bash training/baselines/run_4gpu.sh

SEEDS="${SEEDS:-2027,2028,2029,2030,2031}"
EPOCHS="${EPOCHS:-160}"
BATCH_SIZE="${BATCH_SIZE:-96}"
CONFIG="${CONFIG:-training/configs/nfe_predictor.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-training/baselines/results}"

python training/baselines/export_manifest.py --config "${CONFIG}"
python training/baselines/run.py \
  --model dummy --config "${CONFIG}" --seeds "${SEEDS}" --output-root "${OUTPUT_ROOT}"
python training/baselines/run.py \
  --model xgboost --config "${CONFIG}" --seeds "${SEEDS}" --output-root "${OUTPUT_ROOT}"

models=(cgcnn schnet alignn m3gnet)
for gpu in 0 1 2 3; do
  model="${models[$gpu]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/baselines/run.py \
    --model "${model}" \
    --config "${CONFIG}" \
    --seeds "${SEEDS}" \
    --device cuda:0 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --output-root "${OUTPUT_ROOT}" &
done
wait

if [[ -n "${OURS_CHECKPOINT:-}" ]]; then
  CUDA_VISIBLE_DEVICES=0 python training/baselines/run.py \
    --model ours \
    --config "${CONFIG}" \
    --seeds "${SEEDS}" \
    --device cuda:0 \
    --batch-size "${BATCH_SIZE}" \
    --ours-checkpoint "${OURS_CHECKPOINT}" \
    --output-root "${OUTPUT_ROOT}"
fi

python training/baselines/summarize.py \
  --results-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}"
