# 训练教程 / Training Guide

## 1. Environment

Predictor training targets Linux, Python 3.10 and 1–4 CUDA GPUs. Install the project editable and
verify the CUDA-enabled PyTorch build before training. Official upstream baseline packages have
separate environments under `training/baselines/official/`.

## 2. Data/cache contract

The predictor uses `data/full/nfe_dataset.csv`, structures under `data/full/` and the graph cache
configured in `training/configs/nfe_predictor.yaml`. The formal cache is now
`nfe-mxene-cache-2.0` with:

- `intensive-slab-v2`: global slab descriptors are invariant to exact in-plane supercell replication;
- `radius-shell-complete-v2`: `max_neighbors` is a soft cap and the k-th degenerate distance shell is
  retained completely.

Changing table SHA256, cutoff, neighbor cap or either graph-semantic schema invalidates the cache and
forces rebuild. Formal checkpoint comparisons also require matching provenance.

## 3. Predictor training

Single GPU:

```bash
python training/entrypoints/train.py --gpus 1 --task predictor \
  --config training/configs/nfe_predictor.yaml --rebuild-cache
```

Four GPU:

```bash
python training/entrypoints/train.py --gpus 4 --devices 0,1,2,3 --task predictor \
  --config training/configs/nfe_predictor.yaml
```

Reference hyperparameters are 220 epochs, batch 96/GPU, AdamW 3e-4, 8-epoch warmup + cosine, AMP and
patience 35.

### Training phase terminology

The first 35 epochs are **SSL-dominant joint training**, not pure self-supervised pretraining.
Classification/regression remain active with `supervised_factor=0.25`, while masked-atom and
coordinate-denoising objectives have `ssl_factor=1.0`. After epoch 35 training becomes
supervised-dominant (`supervised_factor=1.0`) and the SSL objectives remain present at reduced
`ssl_factor=0.20`. Manuscripts and figures should use this terminology.

Checkpoint selection uses validation only. Final temperature calibration is fitted on validation
after the best checkpoint is selected. DDP evaluation de-duplicates sampler padding by stable cache
record index before metrics are computed.

## 4. Ablations and baselines

Internal ablations are documented in `training/ablations/README.md`. Baselines are separated into:
architecture-matched controls, official-upstream backbones, and full-system evaluation; see
`training/baselines/README.md`.

Do not compare an auxiliary/SSL-rich full model against a class+score-only baseline and attribute the
entire difference to equivariance. Use `matched_supervision` / architecture track for that question.

## 5. Minimum predictor report

Report at least:

- per-class precision/recall/F1/support, ROC-AUC and Average Precision;
- accuracy, balanced accuracy, macro F1, macro ROC-AUC and macro AP;
- calibrated ECE;
- NFE score MAE/RMSE/Spearman/R²;
- high-class Precision/Recall/Enrichment at top screening fractions;
- chemistry and cell-size OOD slices;
- DFT/manual verified NFE subset when making claims about real NFE states;
- paired `Split_Group` bootstrap confidence intervals for key model comparisons.

The NFE labels remain pseudo labels until independently confirmed by real-space localization evidence
(e.g. band-decomposed partial charge in the surface/vacuum region) together with dispersion/effective
mass evidence.

## 6. Production inference compatibility

`training/entrypoints/predict.py` uses a schema guard. A legacy checkpoint without
`intensive-slab-v2` and `radius-shell-complete-v2` provenance is rejected instead of silently being
fed features with new meanings. Keep old checkpoints for archival reproduction only; retrain formal
v2 predictors before new screening.

## 7. Generator and recovery

Surface/manifold generator behavior is unchanged by the benchmark audit. Keep generator checkpoints,
surface geometry summary, predictor and CHGNet assets together for reproducibility. Use separate
checkpoint directories for every ablation/seed and do not overwrite production weights.

Stop and diagnose NaN/Inf, persistent CUDA OOM, NCCL failures, cache skip fraction above the configured
limit, split-group overlap, mixed provenance, or a failed supercell-consistency audit before starting
large production runs.
