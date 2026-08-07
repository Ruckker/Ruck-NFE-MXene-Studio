#!/usr/bin/env bash
set -euo pipefail

# Four-GPU launcher for the audited benchmark tracks.
# The architecture track trains independent class+score-only models.
# The full-system track evaluates independently trained full checkpoints from
# runs/ablations/full/seed_<seed>/best.pt.

SEEDS="${SEEDS:-2027,2028,2029,2030,2031}"
EPOCHS="${EPOCHS:-220}"
BATCH_SIZE="${BATCH_SIZE:-96}"
CONFIG="${CONFIG:-training/configs/nfe_predictor.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-training/baselines/results}"
FULL_SYSTEM_ROOT="${FULL_SYSTEM_ROOT:-runs/ablations/full}"
RUN_FULL_SYSTEM="${RUN_FULL_SYSTEM:-1}"

python training/baselines/export_manifest.py --config "${CONFIG}"
python training/baselines/run.py \
  --track architecture --model dummy \
  --config "${CONFIG}" --seeds "${SEEDS}" --output-root "${OUTPUT_ROOT}"
python training/baselines/run.py \
  --track architecture --model xgboost \
  --config "${CONFIG}" --seeds "${SEEDS}" --output-root "${OUTPUT_ROOT}"

models=(cgcnn schnet alignn m3gnet)
for gpu in 0 1 2 3; do
  model="${models[$gpu]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/baselines/run.py \
    --track architecture \
    --model "${model}" \
    --config "${CONFIG}" \
    --seeds "${SEEDS}" \
    --device cuda:0 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --output-root "${OUTPUT_ROOT}" &
done
wait

CUDA_VISIBLE_DEVICES=0 python training/baselines/run.py \
  --track architecture \
  --model painn \
  --config "${CONFIG}" \
  --seeds "${SEEDS}" \
  --device cuda:0 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --output-root "${OUTPUT_ROOT}"

if [[ "${RUN_FULL_SYSTEM}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES=0 python training/baselines/run.py \
    --track full-system \
    --model ours_full \
    --config "${CONFIG}" \
    --seeds "${SEEDS}" \
    --device cuda:0 \
    --batch-size "${BATCH_SIZE}" \
    --ours-root "${FULL_SYSTEM_ROOT}" \
    --output-root "${OUTPUT_ROOT}"
fi

python training/baselines/summarize.py \
  --results-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}"
