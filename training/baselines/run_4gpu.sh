#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-2027,2028,2029,2030,2031}"
EPOCHS="${EPOCHS:-220}"
BATCH_SIZE="${BATCH_SIZE:-96}"
CONFIG="${CONFIG:-training/configs/nfe_predictor.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-training/baselines/results}"

python training/baselines/export_manifest.py --config "${CONFIG}"
python training/baselines/run.py --track architecture --model dummy --config "${CONFIG}" --seeds "${SEEDS}" --output-root "${OUTPUT_ROOT}"
python training/baselines/run.py --track architecture --model xgboost --config "${CONFIG}" --seeds "${SEEDS}" --output-root "${OUTPUT_ROOT}"

models=(cgcnn_controlled schnet_controlled angle_moment state_threebody)
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" python training/baselines/run.py \
    --track architecture --model "${models[$gpu]}" --config "${CONFIG}" \
    --seeds "${SEEDS}" --device cuda:0 --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" --output-root "${OUTPUT_ROOT}" &
done
wait

# Run the matched PaiNN backbone after one GPU becomes free.
CUDA_VISIBLE_DEVICES=0 python training/baselines/run.py \
  --track architecture --model painn --config "${CONFIG}" --seeds "${SEEDS}" \
  --device cuda:0 --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" \
  --output-root "${OUTPUT_ROOT}"

# Full-system evaluation requires five independently trained audited full checkpoints.
if [[ "${RUN_FULL_SYSTEM:-0}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES=0 python training/baselines/run.py \
    --track full-system --model ours_full --config "${CONFIG}" --seeds "${SEEDS}" \
    --device cuda:0 --batch-size "${BATCH_SIZE}" --output-root "${OUTPUT_ROOT}"
fi

python training/baselines/summarize.py --results-root "${OUTPUT_ROOT}" --output-dir "${OUTPUT_ROOT}"
