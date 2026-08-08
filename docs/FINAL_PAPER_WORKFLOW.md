# Final paper workflow — pair-symmetric v2.4

This document defines the **paper-ready** path. Use:

```bash
python -m training.paper <alias> [arguments...]
```

`training.paper` is the only final-paper dispatcher. It installs the v2.4 pair-symmetric graph contract before importing trainers/evaluators, requires a clean Git revision, fixes the registered scientific budget, requires one CUDA process per independent training run, forbids arbitrary module passthrough, and routes final summaries through closed-set guards.

`python -m training.formal_v2_4 ...` is retained only for smoke tests, debugging, abbreviated runs and archival/development work. Results produced with altered budgets must not be admitted to final paper tables.

## Immutable paper identity

The dispatcher fixes:

```text
training/configs/nfe_predictor_v2_4_paper_ready.yaml
cache/nfe_graphs_v2_4.pt
```

Final graph semantics:

- cache schema: `nfe-mxene-cache-2.4`;
- global features: `intrinsic-slab-v3`;
- neighbor policy: `radius-shell-complete-pair-symmetric-v3`;
- graph cutoff: 6 Å;
- neighbor limit: shell-complete soft cap at 36, followed by exact reverse-edge closure;
- zero skipped cache records for paper-ready data;
- generator minimum vacuum: 15 Å, greater than both predictor (6 Å) and generator (12 Å) cutoffs.

The pair closure means every retained `(j -> i, shift)` has `(i -> j, -shift)`. Realized degree may exceed the nominal soft cap when a degenerate shell or reverse edge must be retained.

## A. Preflight before expensive training

Run from the final **clean** commit:

```bash
python -m training.paper cache-rebuild-audit
python -m training.paper cache-sanity-audit
python -m training.paper split-duplicate-audit
python -m training.paper neighbor-symmetry-audit
python -m training.paper generator-contract-audit
```

All must pass. These checks cover fresh-cache reproducibility, finite/index-valid tensors, target/weight sanity, slab-vacuum adequacy, exact duplicate leakage, pair-symmetric periodic edges and generator/predictor vacuum compatibility.

Review any reported near-duplicate split candidates manually before starting the formal campaign.

## B. Full model and ablations

The paper optimization protocol is **one Python process / one GPU / batch 96 per independent run**. Use multiple GPUs to run independent seeds/models concurrently; do not use DDP inside a paper run.

Run the full model through the `full` ablation so the same five-seed machinery is used for all system comparisons:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.paper ablation --ablation full --seed 2027
CUDA_VISIBLE_DEVICES=1 python -m training.paper ablation --ablation full --seed 2028
```

Continue with the preregistered seed set `2027,2028,2029,2030,2031` and all nine ablations:

```text
full
no_vector
no_global
no_masked_pretrain
no_denoise
no_self_supervision
no_auxiliary_regression
matched_supervision
classification_only
```

For interrupted jobs, resume only the checkpoint belonging to the same ablation/seed run directory. Do not copy/relabel checkpoints between seeds or ablations.

## C. Controlled/matched architecture baselines

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.paper baseline \
  --track architecture --model painn --seeds 2027
```

Run all preregistered architecture models and all five seeds:

```text
dummy
xgboost
cgcnn_controlled
schnet_controlled
angle_moment
state_threebody
painn
```

The paper dispatcher injects the registered neural budget and CUDA device. `dummy` and XGBoost remain non-neural reference baselines.

## D. Official-upstream backbone track

Run each package in its isolated pinned environment. Example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.paper official \
  --model schnet_official --seeds 2027
```

Required official set:

```text
cgcnn_official
schnet_official
alignn_official
m3gnet_official
```

Pinned audited dependencies include SchNetPack 2.2.0, ALIGNN 2026.5.20 with DGL 2.1.0, and MatGL 4.0.3. CGCNN additionally requires a clean exact-commit upstream checkout. These are official backbones/operators adapted to the common project graph/task heads; do not describe them as untouched upstream training pipelines.

## E. Representation and checkpoint audits

For representative full checkpoints:

```bash
python -m training.paper representation-audit \
  --checkpoint /path/to/best.pt sample1.vasp sample2.vasp
```

The acceptance test covers site permutation, exact in-plane supercells, equivalent unimodular in-plane basis changes and added vacuum.

Audit checkpoint internals before formal inference:

```bash
python -m training.paper checkpoint-audit \
  /path/to/seed_2027/best.pt /path/to/seed_2028/best.pt
```

## F. Bind prediction CSVs to run identity

For validation and test predictions of every formal run:

```bash
python -m training.paper sign-predictions \
  --predictions /path/to/test_predictions.csv
```

The signer recomputes core metrics from the CSV and reconciles them with the adjacent result before writing a SHA256 content-addressed manifest. This is a reproducibility/integrity manifest, not a public-key authenticity signature.

## G. Verified physical-NFE subset

### G1. Prediction-blind selection

Choose one preregistered design before reading model predictions:

```bash
python -m training.paper verified-queue \
  --mode class-balanced-group-diverse --per-class 50 --seed 2027
```

or:

```bash
python -m training.paper verified-queue \
  --mode test-prevalence-random --total 150 --seed 2027
```

### G2. Blind the reviewer sheet

```bash
python -m training.paper blind-verified \
  --queue training/evaluation/results/verified_review_queue.csv
```

### G3. Freeze exact preregistered membership

```bash
python -m training.paper freeze-verified \
  --review-sheet /path/to/completed_blinded_sheet.csv \
  --selection-queue training/evaluation/results/verified_review_queue.csv \
  --selection-manifest training/evaluation/results/verified_review_queue.selection.json \
  --blinding-manifest /path/to/blinded_sheet.blinding.json \
  --confirm-reviewer-blinded-to-model-predictions
```

If `Verified_NFE_Score` is used, one explicit score definition must be applied to the entire reviewed set.

### G4. Evaluate the frozen subset

```bash
python -m training.paper verified-evaluate \
  --predictions /path/to/test_predictions.csv \
  --verified /path/to/completed_blinded_sheet.csv \
  --paper-frozen-manifest /path/to/completed_blinded_sheet.paper_frozen.json
```

All review-complete cases are the primary analysis. Reviewer-confidence thresholds are sensitivity analyses only.

## H. OOD and paired statistics

OOD:

```bash
python -m training.paper ood-evaluate \
  --predictions /path/to/test_predictions.csv \
  --manifest /path/to/ood_manifest.csv
```

For model-vs-model paper inference use the strict seed × `Split_Group` bootstrap over the same five independent seeds:

```bash
python -m training.paper paired-bootstrap \
  --a A_seed2027/test_predictions.csv A_seed2028/test_predictions.csv A_seed2029/test_predictions.csv A_seed2030/test_predictions.csv A_seed2031/test_predictions.csv \
  --b B_seed2027/test_predictions.csv B_seed2028/test_predictions.csv B_seed2029/test_predictions.csv B_seed2030/test_predictions.csv B_seed2031/test_predictions.csv \
  --name-a A --name-b B
```

Each side must be one fixed model/track, use the same unique seed set and distinct checkpoint hashes, and share the exact signed data identity.

## I. Closed-set final summaries

Only after all preregistered runs exist:

```bash
python -m training.paper baseline-summary
python -m training.paper ablation-summary
```

The paper wrappers reject incomplete model/ablation sets. A failed or unfavorable preregistered baseline cannot be silently omitted. They also retain the existing provenance, seed, protocol and checkpoint-independence guards.

Never aggregate v2.3 and v2.4 artifacts together.

## J. Last artifact gate

After prediction manifests are written:

```bash
python -m training.paper paper-preflight \
  /path/to/run1/result.json \
  /path/to/run2/result.json \
  /path/to/ablation/final_metrics.json
```

Paper-ready rules include current v2.4 graph semantics, exact cache/target/structure/split/normalizer identity, clean code revision, zero cache skips, untampered prediction manifests and metric reconciliation.

## K. Scientific wording

Follow:

- `docs/PAPER_CLAIM_BOUNDARIES.md`
- `docs/STATISTICAL_ANALYSIS_PLAN.md`
- `docs/PREDICTION_MANIFEST_SECURITY.md`
- `docs/OOD_INTERPRETATION.md`

The pseudo-label benchmark tests generalization of a computational NFE definition; it does not by itself prove a physical NFE state. Strong physical claims require the independently frozen higher-fidelity evidence subset.
