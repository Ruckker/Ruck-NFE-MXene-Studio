# 训练教程 / Training Guide

## 1. Two execution modes

### Paper-ready

Final benchmark/ablation/statistical artifacts must use:

```bash
python -m training.paper <alias> [arguments...]
```

The paper dispatcher fixes the registered v2.4 data/graph semantics and scientific budget, requires a clean Git revision and one CUDA Python process per independent training run, and blocks budget/config overrides.

### Development / smoke

Use `python -m training.formal_v2_4 ...` or the lower-level entrypoints for short smoke tests, debugging or archival reproduction. Altered-budget outputs are not paper-table artifacts.

## 2. Final predictor data/cache contract

Paper-ready configuration:

```text
training/configs/nfe_predictor_v2_4_paper_ready.yaml
cache/nfe_graphs_v2_4.pt
```

Formal graph/data identity:

- `nfe-mxene-cache-2.4`;
- `intrinsic-slab-v3` global descriptors;
- `radius-shell-complete-pair-symmetric-v3` neighbor policy;
- radius/cutoff 6 Å;
- shell-complete soft neighbor cap 36 followed by exact reverse-edge closure;
- zero skipped cache rows;
- exact structure-byte, target, cache-tensor, train-normalizer and split identities;
- slab normal vacuum strictly greater than graph cutoff.

The v3 global descriptors remove raw cell-size/vacuum shortcuts and are designed to remain unchanged under exact in-plane supercell replication, equivalent in-plane basis changes, z-origin shifts and added vacuum for the same slab.

## 3. Paper optimization protocol

Each independent neural run uses one Python process on one GPU. Four GPUs should run four independent seeds/models concurrently rather than one 4-GPU DDP run.

Registered predictor budget:

- hidden/vector channels 192/64;
- 6 interaction layers;
- 220 epochs;
- batch 96 per independent run;
- AdamW 3e-4, minimum LR 5e-6;
- 8-epoch LR warmup + cosine;
- AMP enabled;
- patience 35;
- gradient accumulation 1.

The first 35 epochs of the full system are **SSL-dominant joint training**, not pure self-supervised pretraining. Supervised losses remain active at 0.25×; masked-atom and coordinate-denoising objectives are dominant. Later training is supervised-dominant while SSL remains auxiliary.

Pure-supervised architecture/official baselines instead use 1.0× supervised loss from epoch zero.

## 4. Formal preflight

Before expensive runs:

```bash
python -m training.paper cache-rebuild-audit
python -m training.paper cache-sanity-audit
python -m training.paper split-duplicate-audit
python -m training.paper neighbor-symmetry-audit
python -m training.paper generator-contract-audit
```

All must pass.

## 5. Full/ablation campaign

Use seeds `2027,2028,2029,2030,2031` for every preregistered ablation. One example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.paper ablation \
  --ablation full --seed 2027
```

Or launch the fixed four-GPU campaign:

```bash
bash training/ablations/run_4gpu.sh
```

The paper summary requires all nine preregistered ablations and the common five-seed set.

## 6. Architecture and official baselines

Architecture launcher:

```bash
bash training/baselines/run_4gpu.sh
```

Official upstream backbones must run in their isolated pinned environments through:

```bash
python -m training.paper official --model <official_model> --seeds 2027
```

See `training/baselines/README.md` and `training/baselines/official/README.md`. Do not attribute the entire full-vs-baseline gap to equivariance; use the matched architecture track and causal ablations for component claims.

## 7. Metrics and statistics

Report at least:

- per-class precision/recall/F1/support, ROC-AUC and Average Precision;
- accuracy, balanced accuracy, macro F1, macro ROC-AUC and macro AP;
- calibrated ECE;
- NFE pseudo-score MAE/RMSE/Spearman/R²;
- high-class Precision/Recall/Enrichment at preregistered screening fractions;
- chemistry and cell-size OOD slices;
- prediction-blind frozen higher-fidelity NFE evidence subset for physical-state claims;
- strict five-seed × `Split_Group` paired bootstrap for planned model comparisons.

Pseudo labels are not independent physical ground truth. Physical NFE claims require real-space localization plus dispersion/effective-mass evidence in the frozen verified layer.

## 8. Prediction/checkpoint compatibility

Formal prediction rejects stale target/data/global/neighbor semantics, incompatible normalizers, changed code revisions and dirty-worktree checkpoints. v2.3 and older checkpoints are archival unless explicitly reproduced with their matching old code/data semantics.

For long jobs, resume only the same run directory's own `best.pt`; do not fork a checkpoint into another seed/ablation directory and continue it under a new identity.

## 9. Generator integration

The final paper configuration requires at least 15 Å vacuum because the generator cutoff is 12 Å and the predictor cutoff is 6 Å. Run `generator-contract-audit` before formal generation/screening.

Stop and diagnose NaN/Inf, OOM, cache/split/provenance/audit failures or representation-consistency drift before launching the full campaign.
