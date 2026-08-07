# NFE baseline benchmark suite

This directory provides audited, leakage-safe comparison tracks for the NFE predictor. Every run
reuses the same graph cache, fixed `Suggested_Split`, group-disjoint `Split_Group`, NFE target
semantics, train-only normalizers, and repository metric definitions.

## Why there are two tracks

A single table previously mixed architecture differences with differences in supervision and
training. The audited benchmark therefore separates two scientific questions.

### Track A: `architecture`

This track asks how the graph representation performs under matched supervision and a matched
nominal training budget.

| Name | Role | Supervision |
|---|---|---|
| `dummy` | class-prior + median-score lower bound | train labels/score statistics only |
| `xgboost` | structure-only classical baseline | NFE class + NFE pseudo-score |
| `cgcnn` | controlled CGCNN-style GNN | NFE class + NFE pseudo-score |
| `schnet` | controlled SchNet-style GNN | NFE class + NFE pseudo-score |
| `alignn` | controlled angle-aware GNN | NFE class + NFE pseudo-score |
| `m3gnet` | controlled state/directional-moment GNN | NFE class + NFE pseudo-score |
| `painn` | Ruck-NFE scalar/vector backbone, matched version | NFE class + NFE pseudo-score |

The neural architecture baselines use the same nominal defaults: 192 hidden channels, 6 layers,
220 epochs, batch 96, AdamW `3e-4`, 8-epoch warmup + cosine, patience 35, dropout 0.12 and label
smoothing 0.04. They do **not** use auxiliary electronic-property targets, masked-atom prediction,
or coordinate denoising. Exact scalar parameter counts still differ by architecture and are reported
in every result; this is matched supervision/training budget, not exact parameter-count matching.

The controlled CGCNN/SchNet/ALIGNN/M3GNet-style models remain compact in-repository
reimplementations, not vendored official upstream packages. Exact upstream adapters are a separate
follow-up experiment and must not be implied by the current labels.

### Track B: `full-system`

This track asks what the complete Ruck-NFE system achieves with its intended multi-task/SSL
training. It does not reuse one checkpoint under several seed labels. For every requested seed it
requires an independently trained full checkpoint at:

```text
runs/ablations/full/seed_<seed>/best.pt
```

The default paper summary requires five independent seeds (2027–2031).

## Leakage and data-quality gates

No benchmark may randomly resplit the table. Every load calls `assert_disjoint_split_groups` and
also enforces the same `max_cache_skip_fraction` hard gate used by the main predictor trainer.

XGBoost is constructed only from cached atomic numbers, elemental descriptors and geometry-only
global invariants. It does not read NFE candidate fields, DOS, work function, band gap, ELF,
charge-density quantities, Fermi energy, total energy or other DFT-derived electronic inputs.

Every result records:

- dataset-table SHA256;
- exact split-manifest SHA256;
- Git commit SHA when available;
- graph-cache schema, radius and `max_neighbors`;
- number of records and skipped cache rows.

Full-system checkpoints must match the current dataset and split hashes. A legacy checkpoint without
provenance is rejected by default; `--allow-unverified-checkpoint` exists only for explicit legacy
inspection and should not be used for a paper table.

## Install

```bash
python -m pip install -r training/baselines/requirements-classical.txt
# or
python -m pip install -e ".[baseline-classical]"
```

## Export the fixed split manifest

```bash
python training/baselines/export_manifest.py
```

The manifest now includes a stable `Record_Index` in addition to structure ID, split group, path,
class label and NFE pseudo-score.

## Run the architecture track

```bash
python training/baselines/run.py \
  --track architecture \
  --model all \
  --seeds 2027,2028,2029,2030,2031 \
  --device cuda
```

A short smoke test can use one model and a few epochs:

```bash
python training/baselines/run.py \
  --track architecture \
  --model painn \
  --seeds 2027 \
  --epochs 5 \
  --patience 5 \
  --device cuda
```

## Run the full-system track

First train an independent `full` ablation checkpoint for every seed. Then evaluate them with:

```bash
python training/baselines/run.py \
  --track full-system \
  --model ours_full \
  --ours-root runs/ablations/full \
  --seeds 2027,2028,2029,2030,2031 \
  --device cuda
```

## Four-GPU launcher

```bash
SEEDS=2027,2028,2029,2030,2031 \
EPOCHS=220 \
FULL_SYSTEM_ROOT=runs/ablations/full \
bash training/baselines/run_4gpu.sh
```

The launcher runs Dummy/XGBoost, distributes the four controlled GNN jobs across GPUs 0–3, runs
the matched PaiNN backbone, and finally evaluates the five independently trained full-system
checkpoints. Set `RUN_FULL_SYSTEM=0` if the full checkpoints have not yet been trained.

## Outputs

```text
training/baselines/results/
  architecture/
    <model>/seed_<seed>/
      best.pt                    # neural architecture models
      history.jsonl              # neural architecture models
      result.json
      validation_predictions.csv
      test_predictions.csv
  full-system/
    ours_full/seed_<seed>/
      result.json
      validation_predictions.csv
      test_predictions.csv
```

Each per-sample prediction file contains structure ID, split group, true/predicted label, all three
class probabilities, true/predicted NFE pseudo-score and absolute score error. These files preserve
paired information for later statistical/error analysis instead of saving only aggregate metrics.

## Summarize

```bash
python training/baselines/summarize.py
```

Outputs:

- `benchmark_per_seed.csv`;
- `benchmark_summary.csv`;
- `architecture_paper_table.csv`;
- `full_system_paper_table.csv`;
- `benchmark_paper_table.csv`.

The summarizer refuses to combine results with different dataset or split hashes. If full-system
results are present, its default paper mode also refuses to summarize fewer than five independent
full-system seeds.

## Main reported metrics

Because the NFE classes are imbalanced, do not rank models by accuracy alone. The current main table
reports macro F1, balanced accuracy, macro ROC-AUC, low/medium/high F1, low/high recall, calibrated
ECE and `NFE_Pseudo_Score` MAE/RMSE. The individual metrics, rather than the internal composite
`selection_score`, should carry the manuscript interpretation.

## Fair-comparison boundary

The architecture track answers the narrower question of representation quality under class+score
supervision and a common nominal training budget. The full-system track measures the complete
NFE-specific system. Do not use the gap between these two tracks as a pure architecture effect,
because the full system intentionally adds global slab information, auxiliary physics targets and
self-supervised objectives.
