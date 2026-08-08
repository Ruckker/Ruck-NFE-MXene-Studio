# Scientific validation and statistical evaluation

The main benchmark targets are electronic-structure-derived **pseudo labels**. Final claims therefore separate:

1. Can a model reproduce/generalize the fixed computational pseudo-label definition?
2. Does the prediction agree with a prediction-blind, higher-fidelity NFE evidence subset and with chemistry/cell-size OOD tests?

For final paper work, use the aliases exposed by:

```bash
python -m training.paper <alias> ...
```

Direct `training.evaluation.*` scripts remain development/debug entrypoints and should not be substituted for the paper-ready guards.

## Verified NFE subset

The review protocol explicitly separates `*_Reviewed` from `*_Confirmed`. Negative reviewed findings are valid evidence; they are not filtered out.

Paper flow:

```bash
python -m training.paper verified-queue ...
python -m training.paper blind-verified ...
python -m training.paper freeze-verified ...
python -m training.paper verified-evaluate ...
```

Selection is generated without model predictions. The reviewer-facing sheet removes pseudo/model label cues. Freezing requires exact membership equality with the preregistered queue. The primary verified analysis uses all review-complete cases; reviewer-confidence cutoffs are sensitivity analyses only.

`Verified_NFE_Score` is optional. If it is used, one explicit score definition must be applied consistently to the reviewed set.

The verified subset is a higher-fidelity DFT/manual adjudication layer, not automatically an experimentally independent ground truth.

## OOD strata

Use the strict paper wrapper:

```bash
python -m training.paper ood-evaluate \
  --predictions /path/to/test_predictions.csv \
  --manifest /path/to/ood_manifest.csv
```

OOD chemistry thresholds/seen sets are derived from train only. Sparse slices may legitimately lack a class; undefined metrics remain null/NaN rather than being fabricated as zero/chance.

Unseen-element OOD still has access to fixed elemental descriptors and must not be described as zero-prior unknown chemistry.

## Five-seed paired uncertainty

Single-seed block bootstrap is useful diagnostically, but final model-vs-model inference uses the strict nested training-seed × `Split_Group` bootstrap:

```bash
python -m training.paper paired-bootstrap \
  --a A_seed2027/test_predictions.csv A_seed2028/test_predictions.csv A_seed2029/test_predictions.csv A_seed2030/test_predictions.csv A_seed2031/test_predictions.csv \
  --b B_seed2027/test_predictions.csv B_seed2028/test_predictions.csv B_seed2029/test_predictions.csv B_seed2030/test_predictions.csv B_seed2031/test_predictions.csv \
  --name-a A --name-b B
```

Each side must be one fixed track/model, use the same unique five-seed set, contain distinct checkpoint hashes and share the exact signed benchmark data identity.

## Prediction manifests

Bind every formal validation/test CSV to the adjacent result:

```bash
python -m training.paper sign-predictions --predictions /path/to/test_predictions.csv
```

The formal signer recomputes core metrics before writing the SHA256 content-addressed manifest. This detects accidental substitution/modification and wrong-result pairing; it is not a public-key digital authenticity signature.

## Representation consistency

```bash
python -m training.paper representation-audit \
  --checkpoint /path/to/best.pt structure1.vasp structure2.vasp
```

The v2.4 acceptance test covers atom permutation, exact in-plane supercells, equivalent unimodular in-plane basis changes and added vacuum. The graph is shell-complete and pair-symmetric.

## Current formal identity

Paper-ready evaluation accepts only the current contract:

- cache `nfe-mxene-cache-2.4`;
- global descriptors `intrinsic-slab-v3`;
- neighbor policy `radius-shell-complete-pair-symmetric-v3`;
- zero skipped cache rows;
- exact target/data/cache/normalizer/split/code provenance;
- clean final Git revision.

Final benchmark and ablation summaries are closed-set and reject omitted preregistered models/ablations.
