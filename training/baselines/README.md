# NFE baseline benchmark suite

All formal tracks use the same fixed `Suggested_Split` / `Split_Group`, target definition, exact v2.3 cached tensor identity, train-normalizer identity, graph budget, clean-Git provenance and validation-only checkpoint selection. Random row resplitting is prohibited.

## Three comparison tracks

### 1. `architecture`

Answers: **what does architecture alone contribute under matched pure supervision?**

Neural models use NFE class + NFE pseudo-score only: no auxiliary electronic targets, no masked-atom objective and no coordinate denoising. The nominal budget is matched: 192 hidden channels, 6 layers where applicable, 220 epochs, AdamW 3e-4, 8-epoch LR warmup and validation-only early stopping. Because these models have no SSL objective, their supervised objective has a constant **1.0× factor from epoch zero**; the full system's SSL-specific 0.25× early supervised factor is not imposed on them.

Keys:
- `cgcnn_controlled` — compact CGCNN-style control;
- `schnet_controlled` — compact SchNet-style control;
- `angle_moment` — internal angle-moment control, **not ALIGNN**;
- `state_threebody` — internal state/three-body-moment control, **not M3GNet**;
- `painn` — Ruck-NFE scalar/vector backbone under class+score supervision, without global slab features, auxiliary targets or SSL;
- Dummy and structure-only XGBoost lower bounds.

`painn` is the architecture-only Ruck-NFE comparator. The `matched_supervision` ablation is not substituted for it because that ablation retains the full model's global branch and heteroscedastic head machinery and preserves the full-system early supervised schedule for causal isolation.

`state_threebody` receives zeroed global slab features in this track so it does not receive an input channel absent from other architecture controls. XGBoost tree count is reported as tree complexity, not mislabeled as neural trainable parameters.

### 2. `official-upstream`

Answers: **does the gain remain relative to recognized upstream backbones under the same task protocol?**

Official CGCNN, SchNetPack SchNet, ALIGNN and MatGL M3GNet message-passing backbones are used with project NFE heads/adapters. These are named **official backbones**, not untouched official training pipelines. To avoid graph-budget confounding, all consume the same `nfe-mxene-cache-2.3` / `radius-shell-complete-v2` periodic edge list. ALIGNN constructs a real line graph; MatGL M3GNet builds its native three-body graph from those bonds.

Formal upstream identities are pinned:
- clean exact-commit `txie-93/cgcnn` checkout;
- `schnetpack==2.2.0`;
- `alignn==2026.5.20`;
- `matgl==4.0.3`.

Official models run in isolated environments. The cache tensor SHA256 and train-normalizer SHA256—not merely package names—must match across formal results. See `official/README.md`.

### 3. `full-system`

Answers: **what performance does the complete NFE-specific system reach?**

This track evaluates independently trained `full` checkpoints. Five seeds are required by default. The summarizer rejects duplicated checkpoint hashes, mismatched seeds, changed cache tensors or normalizers, mixed target/graph semantics, different Git revisions, dirty-worktree runs and mixed full-model training protocols.

The full model's first 35 epochs are an **SSL-dominant joint-training window** with 0.25× supervised loss, followed by supervised-dominant joint training. This schedule is part of the full system; it is not copied into the pure-supervised architecture/official tracks.

## Formal data and graph contract

Current graph semantics are:
- `nfe-mxene-cache-2.3`;
- `intrinsic-slab-v3` global descriptors;
- `radius-shell-complete-v2` neighbor policy.

Formal provenance locks:
- dataset table SHA256;
- actual referenced structure-file bytes;
- target ordering/transforms;
- exact cached graph/feature/target tensors;
- train-fitted normalizer tensors;
- fixed split manifest;
- graph cutoff and neighbor policy;
- clean Git revision and training protocol.

Slabs must have an atom-free normal vacuum gap **strictly larger than the graph cutoff** so 3D PBC cannot introduce artificial cross-vacuum neighbors.

## Formal result contract

Formal result schema is `nfe-baseline-result-2.2`. Older result schemas are intentionally excluded from paper aggregation.

Every neural run saves validation/test per-sample predictions. Stochastic models included in one formal comparison must use the same seed set, and every seed must correspond to a distinct checkpoint SHA256. Screening Precision/Recall/Enrichment metrics are tie-invariant, so Dummy/all-equal scores do not depend on CSV row order.

Do not describe controlled models as official implementations. Do not mix legacy cache/checkpoint/result files with v2.3 paper tables.
