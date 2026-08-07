#!/usr/bin/env bash
set -euo pipefail

# Run the seven controlled predictor ablations by distributing independent
# single-GPU jobs across four visible GPUs. Override variables before launch:
#   SEEDS=2027,2028,2029 EPOCHS=220 BATCH_SIZE=96 bash training/ablations/run_4gpu.sh

SEEDS="${SEEDS:-2027,2028,2029,2030,2031}"
EPOCHS="${EPOCHS:-220}"
BATCH_SIZE="${BATCH_SIZE:-96}"
PATIENCE="${PATIENCE:-35}"
CONFIG="${CONFIG:-training/configs/nfe_predictor.yaml}"
ABLATIONS="${ABLATIONS:-full no_vector no_global no_masked_pretrain no_denoise no_auxiliary_regression classification_only}"

IFS=',' read -r -a seed_array <<< "${SEEDS}"
read -r -a ablation_array <<< "${ABLATIONS}"

pids=()
gpu=0

wait_batch() {
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  pids=()
  gpu=0
}

for seed in "${seed_array[@]}"; do
  for ablation in "${ablation_array[@]}"; do
    echo "Launching ablation=${ablation} seed=${seed} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -m nfe_model.train_ablation \
      --config "${CONFIG}" \
      --ablation "${ablation}" \
      --seed "${seed}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --patience "${PATIENCE}" &
    pids+=("$!")
    gpu=$((gpu + 1))
    if [[ "${gpu}" -eq 4 ]]; then
      wait_batch
    fi
  done
done

if [[ "${#pids[@]}" -gt 0 ]]; then
  wait_batch
fi

python training/ablations/summarize.py \
  --runs-root runs/ablations \
  --output-dir training/ablations/results
