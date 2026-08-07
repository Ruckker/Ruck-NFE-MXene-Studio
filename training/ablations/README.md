# NFE predictor ablation suite

This directory runs controlled ablations of the repository's main periodic equivariant NFE
predictor. The goal is to answer **why** the model works, separately from the external baseline
comparison in `training/baselines/`.

All experiments reuse the original graph cache, `Suggested_Split`, `Split_Group`, robust target
normalizers, DDP/AMP training loop, validation checkpointing, temperature calibration, OOD
embedding bank, and final test evaluation. Only the explicitly ablated component is changed.

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

### Important interpretation of `no_vector`

Coordinate denoising requires an equivariant vector output. Therefore the `no_vector` experiment also
disables coordinate noise/denoising as a mechanically necessary consequence. Its paper label is
recommended as **“− vector/equivariant branch”**, not as a pure one-line pooling ablation.

### Corruption is removed together with its objective

The original trainer feeds masked/noisy structures to the model. For a clean loss ablation, this
suite also removes the corresponding input corruption:

- `no_masked_pretrain`: no atom masking is applied;
- `no_denoise`: no coordinate noise is applied;
- `classification_only`: neither masking nor coordinate noise is applied.

This prevents an ablated auxiliary objective from continuing to affect the supervised task through
its input augmentation.

## One-GPU smoke test

Run a very short experiment first:

```bash
python -m nfe_model.train_ablation \
  --config training/configs/nfe_predictor.yaml \
  --ablation no_global \
  --seed 2027 \
  --epochs 5 \
  --batch-size 48 \
  --patience 5
```

Outputs are isolated from the production predictor:

```text
runs/ablations/
  no_global/
    seed_2027/
      best.pt
      history.jsonl
      final_metrics.json
```

The normal `training/entrypoints/train.py --task predictor` path is unchanged.

## Full five-seed experiment on four GPUs

```bash
SEEDS=2027,2028,2029,2030,2031 \
EPOCHS=220 \
BATCH_SIZE=96 \
PATIENCE=35 \
bash training/ablations/run_4gpu.sh
```

The launcher runs independent single-GPU experiments in batches of four, which is usually more
throughput-efficient for ablations than assigning all four GPUs to one experiment.

To run only selected ablations:

```bash
ABLATIONS="full no_vector no_global" \
SEEDS=2027,2028,2029 \
bash training/ablations/run_4gpu.sh
```

## Manual multi-GPU run for one ablation

The ablation entrypoint is compatible with the original DDP trainer:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m nfe_model.train_ablation \
  --config training/configs/nfe_predictor.yaml \
  --ablation no_vector \
  --seed 2027
```

Do not simultaneously run the four-GPU launcher and a four-rank DDP ablation on the same devices.

## Summarize

```bash
python training/ablations/summarize.py
```

Outputs:

- `training/ablations/results/ablation_per_seed.csv`;
- `training/ablations/results/ablation_summary.csv`;
- `training/ablations/results/ablation_paper_table.csv`.

The paper table contains mean ± sample standard deviation and two effect columns:

- `Δ macro F1 vs full` — negative values mean classification degradation;
- `Δ score MAE vs full` — positive values mean worse NFE-score regression.

`classification_only` reports NFE-score MAE/RMSE as `N/A` because those heads are deliberately not
trained.

## Recommended paper interpretation

Use the external baseline table to answer **“is this architecture competitive?”** and the ablation
table to answer **“which NFE-specific design choices create the gain?”**. Do not combine controlled
CGCNN/SchNet/ALIGNN/M3GNet-style baselines and internal ablations into one model family.

A useful minimum discussion is:

1. `full` vs `no_vector`: directional/equivariant representation contribution;
2. `full` vs `no_global`: slab/vacuum global-information contribution;
3. `full` vs `no_masked_pretrain` and `no_denoise`: self-supervised representation learning;
4. `full` vs `no_auxiliary_regression`: NFE mechanism/property multi-task contribution;
5. `full` vs `classification_only`: total contribution of continuous/physics-aware supervision.
