# Official-upstream baseline adapters

These adapters run in isolated environments because upstream dependency requirements differ. Final paper runs must enter through:

```bash
python -m training.paper official --model <official_model> --seeds <seed-list> [...]
```

Pinned audited identities:

- original `txie-93/cgcnn` checkout at a recorded clean exact Git commit;
- `schnetpack==2.2.0`;
- `alignn==2026.5.20` with `dgl==2.1.0`;
- `matgl==4.0.3` with PyTorch Geometric.

Use the Python version supported by each upstream package; do not force all backbones into one environment. The runtime adapter hard-checks the pinned primary package versions, and result provenance records package/source identity.

## What is official and what is project-specific

Message-passing operators/backbones come from the named upstream projects. The fixed NFE split, class+score task, optimizer/scheduler, calibration, common graph adapter and task heads are project code. Name results `CGCNN (official backbone)`, `SchNetPack SchNet (official backbone)`, etc.; do not claim an untouched upstream training pipeline.

The architecture/official comparison is pure supervised: NFE class + NFE pseudo-score use a constant 1.0× supervised schedule from epoch zero. The full system's SSL-specific early schedule is not imposed on official baselines.

## Common v2.4 graph fairness

All official backbones consume the same paper-ready periodic bond identity:

- cache `nfe-mxene-cache-2.4`;
- neighbor policy `radius-shell-complete-pair-symmetric-v3`;
- radius 6 Å;
- complete kth distance shell retained at the soft neighbor cap;
- every retained periodic edge has its exact reverse counterpart;
- slab vacuum gap exceeds the cutoff.

Adapters:

- SchNetPack consumes the common pair vectors directly;
- CGCNN uses the upstream embedding/ConvLayer parameters, BatchNorms, nonlinearities, pooling and head machinery on **ragged real common edges**. The original dense neighbor tensor has no padding mask, so fake zero-index padding is not used;
- ALIGNN builds the actual DGL atom graph and line graph from the common pair-symmetric bonds;
- MatGL M3GNet receives the common bonds in its center/source convention and builds its native three-body representation.

Official architecture adapters do not receive the full system's extra global-information branch.

## Provenance across isolated environments

Formal reproducibility locks actual scientific inputs rather than requiring identical package stacks for all backbones. Results compare:

- exact cached graph/feature/target tensor SHA256;
- target contract SHA256;
- train-fitted normalizer SHA256;
- dataset/structure/split provenance;
- common v2.4 graph semantics;
- clean project Git revision;
- model-specific package/source identity and protocol fingerprint.

If rebuilding in another environment produces different cache tensors, the results cannot aggregate into one formal table.

## Unseen-element OOD

The fixed input vocabulary covers Z=1..118. An element absent from the train split can therefore be evaluated without redefining the input space. This does **not** mean the model has no elemental prior: project elemental descriptors remain available, and some backbones carry untrained identity embeddings for unseen elements. Describe this as unseen-element-identity/composition OOD, not zero-shot unknown chemistry.

## CGCNN source checkout

`--cgcnn-repo` must point to a clean checkout of the original repository. Its exact commit is part of the model protocol. The historical `--cgcnn-atom-init` option is compatibility-only; the audited adapter uses the project's common 14-D elemental descriptors.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.paper official \
  --model schnet_official --seeds 2027
```

CGCNN:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.paper official \
  --model cgcnn_official \
  --cgcnn-repo /path/to/clean/txie-93-cgcnn \
  --seeds 2027
```

Run the same preregistered seed set `2027–2031` for every official backbone. Final `baseline-summary` refuses an incomplete official set.
