# Scientific validation and statistical evaluation

The main dataset uses electronic-structure-derived **pseudo labels**. Formal claims therefore separate:

1. Can a model reproduce/generalize the fixed pseudo-label rule?
2. Does it generalize to independently inspected NFE physics and chemistry/cell-size OOD cases?

## DFT/manual verified NFE subset

Copy `verified_nfe_template.csv`. `*_Reviewed` means the diagnostic was actually inspected. `*_Confirmed` records the outcome of that inspection. **Negative findings are valid ground-truth evidence and are not filtered out.** This separation prevents a positive-only verified test set.

At minimum, charge localization and parabolic dispersion must have been reviewed and assigned a parseable positive/negative result. `--require-effective-mass` additionally requires the effective-mass review to be completed, but it does not require that finding to be positive.

```bash
python training/evaluation/evaluate_verified_nfe.py \
  --predictions training/baselines/results/full-system/ours_full/seed_2027/test_predictions.csv \
  --verified my_verified_nfe.csv
```

The verified label must be assigned independently of `NFE_Pseudo_Score`.

## OOD strata

```bash
python training/evaluation/build_ood_manifest.py
python training/evaluation/evaluate_slices.py \
  --predictions <test_predictions.csv> \
  --manifest training/evaluation/ood_manifest.csv
```

Thresholds and seen chemistry sets are derived from **train only**. Undefined metrics in sparse slices are represented as JSON `null`, not fabricated chance/zero values.

## Paired block bootstrap

```bash
python training/evaluation/paired_bootstrap.py --a model_A.csv --b model_B.csv \
  --name-a A --name-b B --iterations 5000
```

Rows pair by audited `Record_Index` when available, with structure/group identity cross-checks. Resampling is performed at `Split_Group` level; missing groups fall back to one block per structure. Positive deltas mean model A is better; score MAE is sign-inverted to the same convention. Ties count as 0.5 when estimating `P(A better)`.

## Exact-supercell consistency

```bash
python training/evaluation/supercell_consistency.py \
  --checkpoint runs/ablations/full/seed_2027/best.pt structure.vasp
```

The graph/cache policy removes raw in-plane cell-size descriptors and keeps complete neighbor-distance shells at the soft cap. The audit accepts only current v2.1 full/production checkpoints and rejects legacy graph semantics.
