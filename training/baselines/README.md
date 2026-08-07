# NFE baseline benchmark suite

For final paper work use only:

```bash
python -m training.paper baseline ...
python -m training.paper official ...
python -m training.paper baseline-summary
```

The lower-level `training.baselines.run` / `training.formal_v2_4` entrypoints remain useful for development and smoke tests, but altered-budget results are not paper-ready.

All final tracks share the same fixed `Suggested_Split` / `Split_Group`, pseudo-target definition, exact v2.4 cache tensor identity, train-normalizer identity, graph budget, clean-Git provenance and validation-only checkpoint selection. Random row resplitting is prohibited.

## Comparison tracks

### `architecture`

Question: **what does backbone architecture contribute under matched pure supervision?**

Neural models use NFE class + NFE pseudo-score only: no auxiliary electronic targets, no masked-atom objective and no coordinate denoising. The registered budget is 192 hidden channels, 6 layers where applicable, 220 epochs, AdamW 3e-4, 8-epoch LR warmup, batch 96 and validation-only early stopping. Because these models have no SSL objective, supervised loss is 1.0× from epoch zero.

Preregistered set:

- `dummy` — class prior / score median;
- `xgboost` — structure-only tree baseline;
- `cgcnn_controlled` — compact CGCNN-style control;
- `schnet_controlled` — compact SchNet-style control;
- `angle_moment` — internal angle-moment control, **not ALIGNN**;
- `state_threebody` — internal state/three-body-moment control, **not M3GNet**;
- `painn` — Ruck-NFE local scalar/vector backbone under class+score supervision without global slab features or SSL.

`painn` is the architecture-only Ruck-NFE comparator. `matched_supervision` is an ablation of the full architecture and is not substituted for it.

### `official-upstream`

Question: **does the result remain competitive with recognized upstream backbones under the common task/data protocol?**

Preregistered set:

- `cgcnn_official`;
- `schnet_official`;
- `alignn_official`;
- `m3gnet_official`.

All consume the same v2.4 pair-symmetric common periodic bond graph. ALIGNN constructs a real DGL line graph; MatGL M3GNet builds its native three-body representation from those bonds. CGCNN uses upstream operator weights/BatchNorm/nonlinearities on a ragged real-edge scatter implementation because the original dense padding path has no padding mask.

Pinned audited identities:

- clean exact-commit `txie-93/cgcnn` checkout;
- `schnetpack==2.2.0`;
- `alignn==2026.5.20` and `dgl==2.1.0`;
- `matgl==4.0.3`.

These are official backbones/operators with project task heads/adapters, not untouched upstream training pipelines.

### `full-system`

Question: **what performance does the complete NFE-specific system reach?**

This track evaluates five independently trained `full` ablation checkpoints using seeds 2027–2031. Checkpoint hashes must be distinct and internal seed/provenance must match the evaluation row.

The full system uses a 35-epoch SSL-dominant joint-training window with 0.25× supervised loss, followed by supervised-dominant joint training. This schedule is intentionally not imposed on pure-supervised architecture/official models.

## Final data/graph identity

Paper-ready semantics:

- cache schema `nfe-mxene-cache-2.4`;
- global descriptor schema `intrinsic-slab-v3`;
- neighbor policy `radius-shell-complete-pair-symmetric-v3`;
- radius 6 Å;
- shell-complete soft neighbor cap 36 followed by reverse-edge closure;
- zero skipped cache rows;
- slab normal vacuum strictly greater than the graph cutoff.

Formal provenance locks the CSV, referenced structure bytes, target ordering/transforms, actual graph/feature/target tensors, normalizer tensors, split manifest, graph semantics, clean Git revision and training protocol.

## Final aggregation

The paper summary is closed-set: all seven architecture models, all four official-upstream backbones and `ours_full` must be present. Every stochastic model must use the same five-seed set and one distinct checkpoint per seed. A failed or unfavorable preregistered baseline cannot be silently omitted.

```bash
python -m training.paper baseline-summary
```

Do not mix v2.3/v2.4 results, controlled and official model identities, or development-budget runs in a paper table.
