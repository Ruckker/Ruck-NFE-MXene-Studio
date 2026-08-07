#!/usr/bin/env bash
set -euo pipefail

# Paper-ready launcher: four GPUs execute four independent single-process runs.
# Scientific budget/configuration is intentionally NOT overridable here; it is
# enforced by `python -m training.paper`.
SEEDS=(2027 2028 2029 2030 2031)
ABLATIONS=(
  full
  no_vector
  no_global
  no_masked_pretrain
  no_denoise
  no_self_supervision
  no_auxiliary_regression
  matched_supervision
  classification_only
)

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

for seed in "${SEEDS[@]}"; do
  for ablation in "${ABLATIONS[@]}"; do
    echo "Launching paper ablation=${ablation} seed=${seed} on physical GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -m training.paper ablation \
      --ablation "${ablation}" --seed "${seed}" &
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

# Closed-set paper summary: this fails if any preregistered ablation/seed is missing.
python -m training.paper ablation-summary
