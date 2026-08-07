# NFE baseline benchmark suite

All formal tracks use the same fixed `Suggested_Split` / `Split_Group`, target definition, v2.1 structure manifest, graph budget, clean-Git provenance, and validation-only checkpoint selection. Random row resplitting is prohibited.

## Three comparison tracks

### 1. `architecture`

Answers: **what does architecture alone contribute under matched supervision?**

Neural models use NFE class + NFE pseudo-score only, no auxiliary electronic targets, no masked-atom objective and no coordinate denoising. The nominal budget is matched: 192 hidden channels, 6 layers where applicable, 220 epochs, AdamW 3e-4, 8-epoch LR warmup, validation-only early stopping, and the same first-35-epoch 0.25× supervised weighting window used by `matched_supervision`. Parameter counts are always reported; matched budget does not mean identical parameter count.

Keys:
- `cgcnn_controlled` — compact CGCNN-style control;
- `schnet_controlled` — compact SchNet-style control;
- `angle_moment` — internal angle-moment control, **not ALIGNN**;
- `state_threebody` — internal state/three-body-moment control, **not M3GNet**;
- `painn` — Ruck-NFE scalar/vector backbone under class+score supervision;
- Dummy and structure-only XGBoost lower bounds.

`state_threebody` receives zeroed global slab features in this track so it does not obtain an extra input channel absent from the other architecture controls.

### 2. `official-upstream`

Answers: **does the gain remain relative to recognized upstream backbones?**

The official CGCNN, SchNetPack SchNet, ALIGNN and MatGL M3GNet message-passing implementations are used with project task heads/adapters. To avoid graph-budget confounding, all consume the same `radius-shell-complete-v2` periodic edge list. ALIGNN constructs a real DGL line graph from those bonds; MatGL M3GNet constructs its normal internal three-body line graph. These runs belong in isolated dependency environments; see `official/README.md`.

### 3. `full-system`

Answers: **what performance does the complete NFE-specific system reach?**

This track evaluates independently trained `full` checkpoints. Five seeds are required by default. The summarizer rejects duplicated checkpoint hashes, mismatched checkpoint seeds, different dataset/structure/split/graph provenance, different Git revisions, and dirty-worktree runs.

## Formal result contract

Every neural run saves validation/test per-sample predictions and records dataset-table SHA256, structure-file-manifest SHA256, split-manifest SHA256, graph schema, cutoff/neighbor budget, Git commit and clean/dirty state. The formal summarizer requires the same seed set for all stochastic models that appear together.

Do not describe controlled models as official implementations. Do not mix legacy cache/checkpoint results with v2.1 paper tables.
