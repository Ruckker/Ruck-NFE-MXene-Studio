# Official-upstream baseline adapters

These adapters are intentionally isolated from the main project environment because upstream dependency requirements differ.

Pinned formal identities:
- original `txie-93/cgcnn` checkout at a recorded, clean exact Git commit;
- `schnetpack==2.2.0`;
- `alignn==2026.5.20`;
- `matgl==4.0.3` with PyTorch Geometric.

Use Python versions supported by each upstream package; do not force all four into one environment merely for installation convenience. One `--model` is run per isolated environment.

## What is official and what is project-specific

The message-passing backbone comes from the upstream project. The common NFE dual task (three-class logits + NFE pseudo-score), fixed split, optimizer/scheduler, calibration, graph adapter and task heads are project code. Results must therefore be named `CGCNN (official backbone)`, `SchNetPack SchNet (official backbone)`, etc., rather than claiming an untouched upstream training pipeline.

The architecture/official comparison is pure supervised: class + NFE pseudo-score have a constant 1.0× supervised factor from epoch zero. The full model's SSL-specific 0.25× early supervised window is not imposed on official backbones.

## Graph fairness and provenance

All upstream backbones consume the same audited `nfe-mxene-cache-2.3` / `radius-shell-complete-v2` periodic bond list:
- SchNetPack uses common pair vectors directly;
- CGCNN maps common bonds into the upstream fixed-neighbor tensor format;
- ALIGNN constructs the DGL atom graph and official line graph / bond-angle features;
- MatGL M3GNet constructs a PyG graph from the common bonds and then its native three-body graph.

The cache uses `intrinsic-slab-v3` global semantics for the project full model, but official architecture adapters do not receive an extra full-system global branch.

`max_neighbors` is a soft kth-shell cap, retaining the complete degenerate distance shell. Formal slab data must also have a normal vacuum gap larger than the graph cutoff, so the shared 3D-PBC graph cannot contain artificial cross-vacuum bonds.

Because the four upstream models may require isolated environments, formal reproducibility does not rely only on package versions. Results additionally record and compare:
- exact cached graph/feature/target tensor SHA256;
- target contract SHA256;
- train-fitted normalizer SHA256;
- dataset/structure/split provenance;
- clean project Git revision;
- model-specific upstream package/source identity.

Thus separate environments may consume one immutable cache, but any actual change to tensors or normalizers prevents formal aggregation.

## CGCNN neighbor width

CGCNN requires a fixed neighbor slot width. A single width is derived from the **maximum realized degree in the train split only**. This avoids validation/test feature transduction and keeps a structure's prediction independent of its batch companions. If a validation/test structure exceeds the train-derived width, evaluation fails explicitly rather than truncating bonds.

## Unseen-element OOD

MatGL's element vocabulary is fixed to Z=1..118 as input-space metadata; it is not fitted from test labels. An element unseen during training therefore retains an untrained embedding but can still be evaluated as unseen-element OOD instead of crashing. SchNetPack likewise uses a 119-row nuclear embedding for the project Z<=118 contract.

## CGCNN source checkout

`--cgcnn-repo` points to a clean checkout of the original repository. Its exact commit is included in the model-protocol fingerprint. The older `--cgcnn-atom-init` argument is retained only for command compatibility; v2.3 adapters use the project's common 14-D elemental descriptors.

Example:

```bash
python -m training.baselines.official.run \
  --model schnet_official \
  --seeds 2027,2028,2029,2030,2031 \
  --device cuda
```

For CGCNN:

```bash
python -m training.baselines.official.run \
  --model cgcnn_official \
  --cgcnn-repo /path/to/clean/txie-93-cgcnn \
  --seeds 2027,2028,2029,2030,2031 \
  --device cuda
```
