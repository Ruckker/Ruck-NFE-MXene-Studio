# NFE baseline benchmark suite

This directory provides leakage-safe baselines for the NFE predictor. The benchmark reuses the
same graph cache, `Suggested_Split`, `Split_Group`, target semantics, and metric definitions as the
main model.

## Included baselines

| Name | Type | Input |
|---|---|---|
| `dummy` | class-prior + median-score lower bound | labels only from train split |
| `xgboost` | classical tree baseline | composition, elemental descriptors, lattice/slab geometry |
| `cgcnn` | controlled CGCNN-style gated crystal graph network | periodic crystal graph |
| `schnet` | controlled SchNet-style continuous-filter graph network | periodic distances |
| `alignn` | controlled angle-aware ALIGNN-style graph network | periodic graph + directional angular context |
| `m3gnet` | controlled M3GNet-style state/three-body graph network | periodic graph + directional moments + global state |
| `ours` | existing `PeriodicNFEModel` checkpoint | repository predictor |

The four neural baselines are compact in-repository architectural reproductions. They are designed
for a controlled comparison under exactly the same data loader and split protocol. They are **not**
vendored copies of the upstream CGCNN, SchNetPack, ALIGNN, or MatGL codebases. If a manuscript
requires exact upstream reproduction, export `benchmark_split_manifest.csv` and run the official
packages against that fixed manifest as an additional experiment.

## Leakage policy

All baselines use the existing `Suggested_Split` values and call
`assert_disjoint_split_groups`. No baseline may resplit rows randomly.

The XGBoost input construction deliberately excludes all electronic-structure-derived quantities,
including NFE candidate-band fields, DOS, work function, ELF, charge-density features, Fermi level,
band gap, and total energy. It is built only from the cached atomic numbers, elemental descriptors,
and geometry-only global invariants.

## Install

The graph baselines use the same PyTorch/pymatgen training environment as the main predictor.

For XGBoost, install the optional dependency in an isolated environment or into an existing
compatible training environment:

```bash
python -m pip install -r training/baselines/requirements-classical.txt
```

or, after installing the project:

```bash
python -m pip install -e ".[baseline-classical]"
```

## Export the fixed split manifest

```bash
python training/baselines/export_manifest.py
```

This writes `training/baselines/benchmark_split_manifest.csv` with structure ID, split group,
structure path, class label, and NFE pseudo-score.

## Run one baseline

```bash
python training/baselines/run.py --model dummy --seeds 2027

python training/baselines/run.py \
  --model xgboost \
  --seeds 2027,2028,2029,2030,2031

python training/baselines/run.py \
  --model cgcnn \
  --seeds 2027,2028,2029,2030,2031 \
  --device cuda

python training/baselines/run.py --model schnet --seeds 2027,2028,2029 --device cuda
python training/baselines/run.py --model alignn --seeds 2027,2028,2029 --device cuda
python training/baselines/run.py --model m3gnet --seeds 2027,2028,2029 --device cuda
```

Run the existing predictor checkpoint through the same evaluation wrapper:

```bash
python training/baselines/run.py \
  --model ours \
  --ours-checkpoint models/server/ruck_dp/nfe_predictor/best.pt \
  --seeds 2027 \
  --device cuda
```

Run all baselines. `ours` is included only when a checkpoint is supplied:

```bash
python training/baselines/run.py \
  --model all \
  --ours-checkpoint models/server/ruck_dp/nfe_predictor/best.pt \
  --seeds 2027,2028,2029 \
  --device cuda
```

`ours` is evaluation-only in this suite. Its seed is recorded for table compatibility; the
checkpoint itself is not retrained by `training/baselines/run.py`.

## Four-GPU launcher

On a four-GPU workstation/server, the convenience launcher puts one controlled graph architecture
on each visible GPU and runs Dummy/XGBoost before the GPU jobs:

```bash
OURS_CHECKPOINT=models/server/ruck_dp/nfe_predictor/best.pt \
SEEDS=2027,2028,2029,2030,2031 \
bash training/baselines/run_4gpu.sh
```

The four graph jobs are independent; each job iterates through its requested seeds on a single GPU.
This changes throughput only and does not change the fixed split protocol.

## Outputs

Each run writes:

```text
training/baselines/results/
  <model>/
    seed_<seed>/
      best.pt          # graph baselines only
      history.jsonl    # graph baselines only
      result.json
```

The JSON result uses a common schema and contains validation/test metrics, parameter count,
training time, calibration temperature, split sizes, and the number of skipped graph-cache rows.

Aggregate all finished runs:

```bash
python training/baselines/summarize.py
```

Outputs:

- `benchmark_per_seed.csv`
- `benchmark_summary.csv`
- `benchmark_paper_table.csv`

The paper table reports mean ± standard deviation across seeds for the main quantities. The
summarizer itself only requires NumPy/pandas and can be run without the training stack once the JSON
results already exist.

## Main comparison metrics

Do not rank models by accuracy alone because the NFE classes are imbalanced. The recommended main
table contains:

- macro F1;
- balanced accuracy;
- macro ROC-AUC;
- low/medium/high F1;
- low/high recall;
- calibrated ECE;
- `NFE_Pseudo_Score` MAE/RMSE.

Checkpoint selection for the controlled graph baselines uses the repository's existing
`selection_score`, while the manuscript should show the transparent individual metrics above.

## Fair-comparison notes

1. Keep the same split manifest across every seed and every model.
2. Seeds may change initialization, dropout, and batch order only; they must not change the split.
3. Tune hyperparameters on validation only. Do not use the test set for iterative tuning.
4. Report whether a baseline is the controlled in-repository version or an exact upstream package.
5. The main model has self-supervised pretraining and auxiliary tasks; therefore also report the
   model ablations separately when attributing gains to equivariance, global slab features, or
   multi-task learning.
