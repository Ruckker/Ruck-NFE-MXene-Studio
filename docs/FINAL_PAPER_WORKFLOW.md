# Final paper workflow — v2.4 pair-symmetric contract

This is the single canonical paper-ready workflow. Earlier v2.3 commands and lower-level scripts remain useful for development/archive reproduction, but final paper runs should use the dispatcher below so the pair-symmetric graph contract is installed **before** any trainer, baseline, summarizer or evaluator imports `data_v2` constants.

## Canonical dispatcher

```bash
python -m training.formal_v2_4 <alias> [arguments...]
```

Default formal configuration:

```text
training/configs/nfe_predictor_v2_4.yaml
cache/nfe_graphs_v2_4.pt
```

Final graph identity:
- cache schema: `nfe-mxene-cache-2.4`;
- global features: `intrinsic-slab-v3`;
- neighbor policy: `radius-shell-complete-pair-symmetric-v3`.

The neighbor budget is a soft kth-shell cap. The complete kth distance shell is retained, then the graph is closed under periodic edge reversal: `(j -> i, shift)` always has `(i -> j, -shift)`. Pair closure may increase an atom's realized degree above the nominal cap; no physical retained bond is dropped merely to enforce a hard tensor width.

## A. Before any expensive training

Run from a **clean Git worktree** at the final commit:

```bash
python -m training.formal_v2_4 cache-rebuild-audit
python -m training.formal_v2_4 cache-sanity-audit
python -m training.formal_v2_4 split-duplicate-audit
python -m training.formal_v2_4 neighbor-symmetry-audit
```

All must pass. In particular:
- fresh source rebuild must reproduce the persisted cache tensor identity;
- all tensors/edges/targets/weights must be finite and structurally valid;
- exact source-byte/model-input duplicates cannot cross fixed splits;
- every retained periodic edge must have its reverse counterpart;
- slab vacuum gap must exceed the graph cutoff.

Inspect the reported near-duplicate candidates from the split audit manually before launching the paper campaign.

## B. Full model and ablations

Full/ablation example:

```bash
python -m training.formal_v2_4 ablation \
  --ablation full --seed 2027
```

Use the same predeclared five seeds for every formal ablation. Repeat for the full ablation matrix. Do not copy/relabel checkpoints between seed directories.

Before using a full checkpoint for formal inference:

```bash
python -m training.formal_v2_4 checkpoint-audit \
  /path/to/seed_2027/best.pt /path/to/seed_2028/best.pt
```

## C. Architecture and official-upstream baselines

Controlled/matched baseline example:

```bash
python -m training.formal_v2_4 baseline \
  --track architecture --model painn \
  --seeds 2027,2028,2029,2030,2031 --device cuda
```

Official backbone example:

```bash
python -m training.formal_v2_4 official \
  --model schnet_official \
  --seeds 2027,2028,2029,2030,2031 --device cuda
```

Run each official package in its pinned isolated environment. CGCNN additionally requires the clean exact-commit upstream checkout. The official track uses project task heads/adapters and the common pair-symmetric graph; name results as official **backbones/operators**, not untouched upstream training pipelines.

## D. Representation consistency

For every representative full checkpoint/material class:

```bash
python -m training.formal_v2_4 representation-audit \
  --checkpoint /path/to/best.pt \
  sample1.vasp sample2.vasp
```

This checks site ordering, exact in-plane supercells, equivalent unimodular in-plane basis and added vacuum for the same Cartesian slab. Do not publish a supercell-invariance claim if this acceptance test fails.

## E. Bind prediction CSVs to their run results

For **both validation and test** prediction files of every formal run:

```bash
python -m training.formal_v2_4 sign-predictions \
  --predictions /path/to/test_predictions.csv

python -m training.formal_v2_4 sign-predictions \
  --predictions /path/to/validation_predictions.csv
```

The canonical signer recomputes core metrics from the CSV before writing the SHA256 content-addressed manifest. These manifests provide workflow integrity/consistency, not adversarial public-key authenticity.

## F. Verified physical-NFE subset

### F1. Select cases without reading model predictions

Choose one preregistered sampling design:

```bash
python -m training.formal_v2_4 verified-queue \
  --mode class-balanced-group-diverse --per-class 50 --seed 2027
```

or a simple random sample preserving expected test prevalence:

```bash
python -m training.formal_v2_4 verified-queue \
  --mode test-prevalence-random --total 150 --seed 2027
```

### F2. Blind reviewers to pseudo/model labels

```bash
python -m training.formal_v2_4 blind-verified \
  --queue training/evaluation/results/verified_review_queue.csv
```

### F3. Freeze the completed preregistered review

Only after reviewers finish:

```bash
python -m training.formal_v2_4 freeze-verified \
  --review-sheet /path/to/completed_blinded_sheet.csv \
  --selection-queue training/evaluation/results/verified_review_queue.csv \
  --selection-manifest training/evaluation/results/verified_review_queue.selection.json \
  --blinding-manifest /path/to/blinded_sheet.blinding.json \
  --confirm-reviewer-blinded-to-model-predictions
```

The paper freezer requires exact membership equality with the preregistered queue. If any `Verified_NFE_Score` is supplied, one explicit score definition must be used consistently.

### F4. Evaluate physical verification

```bash
python -m training.formal_v2_4 verified-evaluate \
  --predictions /path/to/test_predictions.csv \
  --verified /path/to/completed_blinded_sheet.csv \
  --paper-frozen-manifest /path/to/completed_blinded_sheet.paper_frozen.json
```

Primary analysis is all review-complete cases. Reviewer-confidence thresholds are sensitivity analyses, not post-hoc alternative primary datasets.

## G. OOD and paired statistics

OOD:

```bash
python -m training.formal_v2_4 ood-evaluate \
  --predictions /path/to/test_predictions.csv \
  --manifest /path/to/ood_manifest.csv
```

For model-vs-model paper inference across five seeds use the strict nested seed × `Split_Group` bootstrap:

```bash
python -m training.formal_v2_4 paired-bootstrap \
  --a A_seed2027/test_predictions.csv A_seed2028/test_predictions.csv A_seed2029/test_predictions.csv A_seed2030/test_predictions.csv A_seed2031/test_predictions.csv \
  --b B_seed2027/test_predictions.csv B_seed2028/test_predictions.csv B_seed2029/test_predictions.csv B_seed2030/test_predictions.csv B_seed2031/test_predictions.csv \
  --name-a A --name-b B
```

Each side must contain one fixed track/model, the same five unique training seeds, distinct checkpoint hashes and the same signed benchmark data identity.

## H. Aggregate only current formal results

```bash
python -m training.formal_v2_4 baseline-summary
python -m training.formal_v2_4 ablation-summary
```

Never aggregate v2.3 and v2.4 artifacts together.

## I. Last gate before paper tables

After all prediction CSVs are formally bound:

```bash
python -m training.formal_v2_4 paper-preflight \
  /path/to/run1/result.json \
  /path/to/run2/result.json \
  /path/to/ablation/final_metrics.json
```

Default paper-ready rules include:
- current v2.4 graph semantics;
- exact cache tensor, target, structure, split and train-normalizer identity;
- clean current Git revision equal to artifact revision;
- zero skipped cache records;
- untampered validation/test prediction manifests;
- prediction run identity equals result identity;
- recomputed CSV metrics equal reported metrics.

Do not use `--allow-cache-skips` for the final table unless every skipped row and the resulting selection effect are explicitly reported and justified.

## J. Scientific wording

Before drafting Results/Discussion, follow:
- `docs/PAPER_CLAIM_BOUNDARIES.md`
- `docs/STATISTICAL_ANALYSIS_PLAN.md`
- `docs/PREDICTION_MANIFEST_SECURITY.md`

The pseudo-label benchmark establishes generalization of a computational NFE definition. It does **not** by itself prove a physical NFE state. Physical NFE claims require the independently frozen verified-evidence layer.
