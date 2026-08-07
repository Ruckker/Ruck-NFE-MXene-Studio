# NFE predictor ablation suite

This directory runs controlled ablations of the repository's periodic equivariant NFE predictor.
The goal is to explain **why** the full system works, separately from the external/matched baseline
comparison in `training/baselines/`.

All experiments reuse the graph cache, fixed group-disjoint splits, train-only normalizers, DDP/AMP
training loop, validation checkpointing, temperature calibration, OOD embedding bank and final test
evaluation. The audited trainer also records dataset/split/Git provenance and removes any
`DistributedSampler` padding duplicates from validation/test metrics.

## Ablations

| Key | Removed component | What stays enabled |
|---|---|---|
| `full` | nothing | original model and losses |
| `no_vector` | equivariant vector message branch | scalar periodic message passing, global features, masked-atom SSL, supervised tasks |
| `no_global` | 11 global lattice/slab/vacuum invariants | equivariant local graph representation and all losses |
| `no_masked_pretrain` | masked-atom corruption/objective | coordinate denoising, supervised classification/regression |
| `no_denoise` | coordinate noise/denoising objective | masked-atom SSL, supervised classification/regression |
| `no_auxiliary_regression` | regression targets other than `NFE_Pseudo_Score` | NFE class, NFE score, both SSL objectives |
| `classification_only` | all regression and SSL objectives | NFE low/medium/high classification only |

### `no_auxiliary_regression` loss semantics

Zero-weight auxiliary regression targets are now removed from the regression-loss denominator as
well as the numerator. Positive-weight targets retain the historical production loss scaling. This
makes the experiment genuinely score-only instead of silently shrinking the score loss by counting
disabled targets in the denominator.

### Important interpretation of `no_vector`

Coordinate denoising requires an equivariant vector output. Therefore `no_vector` necessarily also
disables coordinate noise/denoising. It is **not** a pure one-factor vector ablation. For the current
suite, use:

- `full` vs `no_denoise` to discuss the denoising contribution;
- `no_denoise` vs `no_vector` to discuss the vector/equivariant branch under a no-denoise condition.

Do not attribute `full` vs `no_vector` solely to equivariance.

### Corruption is removed together with its objective

- `no_masked_pretrain`: no atom masking is applied;
- `no_denoise`: no coordinate noise is applied;
- `classification_only`: neither masking nor coordinate noise is applied.

This prevents a disabled auxiliary loss from still changing the supervised task through corrupted
inputs.

## Checkpoint contract and provenance

Ablation checkpoints use the distinct format:

```text
nfe-mxene-predictor-ablation-1.0
```

They no longer masquerade as production predictor checkpoints. Each checkpoint stores the base
`PeriodicNFEModel` constructor configuration, the explicit ablation configuration, architecture
name and provenance metadata. Standard `predict.py` continues to accept only the production
`nfe-mxene-predictor-1.0` format; benchmark full-system evaluation has an explicit loader for the
`full` ablation checkpoint.

Provenance includes the dataset-table SHA256, exact split-manifest SHA256, graph-cache settings and
Git commit when a checkout is available.

## One-GPU smoke test

```bash
python -m nfe_model.train_ablation \
  --config training/configs/nfe_predictor.yaml \
  --ablation no_global \
  --seed 2027 \
  --epochs 5 \
  --batch-size 48 \
  --patience 5
```

Outputs:

```text
runs/ablations/no_global/seed_2027/
  best.pt
  history.jsonl
  final_metrics.json
  validation_predictions.csv
  test_predictions.csv
```

The command surface for normal predictor training remains `training/entrypoints/train.py`; that
entrypoint now routes predictor training through the audited wrapper so production retraining also
gets provenance, DDP-safe evaluation and per-sample final predictions.

## Full five-seed experiment on four GPUs

```bash
SEEDS=2027,2028,2029,2030,2031 \
EPOCHS=220 \
BATCH_SIZE=96 \
PATIENCE=35 \
bash training/ablations/run_4gpu.sh
```

The launcher runs independent single-GPU experiments in batches of four. To run selected ablations:

```bash
ABLATIONS="full no_vector no_global" \
SEEDS=2027,2028,2029 \
bash training/ablations/run_4gpu.sh
```

## Manual multi-GPU run

```bash
torchrun --standalone --nproc-per-node=4 \
  -m nfe_model.train_ablation \
  --config training/configs/nfe_predictor.yaml \
  --ablation no_vector \
  --seed 2027
```

The audited evaluation layer deduplicates sampler padding by stable cache-record index before
metrics are calculated, so validation/test results remain correct even when split size is not
divisible by world size.

## Summarize

```bash
python training/ablations/summarize.py
```

Outputs:

- `training/ablations/results/ablation_per_seed.csv`;
- `training/ablations/results/ablation_summary.csv`;
- `training/ablations/results/ablation_paper_table.csv`.

`classification_only` reports NFE-score MAE/RMSE as `N/A` because that head is deliberately not
trained. Treat all ablation differences as conditional effects of the stated experiment rather than
as generic causal statements about one architectural primitive.
