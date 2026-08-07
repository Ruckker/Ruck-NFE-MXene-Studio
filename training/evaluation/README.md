# Scientific validation and statistical evaluation

The main dataset uses electronic-structure-derived **pseudo labels**. Formal claims should therefore
separate two questions:

1. Can a model reproduce the fixed pseudo-label benchmark?
2. Does it generalize to independently inspected NFE physics and chemistry/cell-size OOD cases?

## DFT/manual verified NFE subset

Copy `verified_nfe_template.csv` and fill it only after inspecting band-decomposed partial charge
(or an equivalent real-space localization diagnostic). The default evidence gate requires both
surface/vacuum charge localization and parabolic dispersion. `--require-effective-mass` can make the
effective-mass check mandatory as well.

```bash
python training/evaluation/evaluate_verified_nfe.py \
  --predictions training/baselines/results/full-system/ours_full/seed_2027/test_predictions.csv \
  --verified my_verified_nfe.csv
```

The verified table is deliberately not generated from `NFE_Pseudo_Score`; otherwise it would not be
an independent validation set.

## OOD strata

```bash
python training/evaluation/build_ood_manifest.py
python training/evaluation/evaluate_slices.py \
  --predictions <test_predictions.csv> \
  --manifest training/evaluation/ood_manifest.csv
```

The manifest marks unseen metal pairs, termination pairs, X elements, any unseen element, a
train-derived cell-size OOD threshold, chemistry OOD and combined OOD. Thresholds are derived from
**train only**.

## Paired block bootstrap

```bash
python training/evaluation/paired_bootstrap.py --a model_A.csv --b model_B.csv \
  --name-a A --name-b B --iterations 5000
```

Sampling is performed at `Split_Group` level so related structures remain together. Positive deltas
mean model A is better; score MAE is sign-inverted to the same convention.

## Exact-supercell consistency

The graph/cache v2 policy removes raw in-plane cell size from global features and keeps complete
neighbor distance shells at the soft neighbor cap. Check a trained **v2** predictor directly:

```bash
python training/evaluation/supercell_consistency.py \
  --checkpoint runs/ablations/full/seed_2027/best.pt structure.vasp
```

A legacy checkpoint without `intensive-slab-v2` / `radius-shell-complete-v2` provenance is rejected,
because evaluating old weights with new feature semantics would be scientifically invalid.
