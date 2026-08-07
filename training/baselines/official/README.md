# Official-upstream baseline adapters

These adapters are intentionally isolated from the main project environment because current package requirements differ substantially.

Pinned targets used by the audit:
- original `txie-93/cgcnn` checkout;
- `schnetpack==2.2.0`;
- `alignn==2026.5.20`;
- `matgl==4.0.3` with PyTorch Geometric.

Use Python versions supported by each upstream package; do not force all four into one environment merely to simplify installation.

## What is official and what is project-specific

The message-passing backbone comes from the upstream project. The common NFE dual task (three-class logits + NFE pseudo-score), optimizer/split/calibration protocol and data adapter are project code. Results must therefore be named `CGCNN (official backbone)`, `SchNetPack SchNet (official backbone)`, etc., rather than claiming an untouched upstream training pipeline.

## Graph fairness

Previous audit revisions let ALIGNN/M3GNet/CGCNN rebuild native graphs from structure files every epoch. That had two problems: large I/O overhead and different neighbor semantics. In particular ALIGNN's k-nearest builder may expand the cutoff when a site has fewer requested neighbors, which is inappropriate for large-vacuum MXene slabs.

The current adapter therefore consumes the same v2.1 periodic bond list used by the project benchmark:

- SchNetPack: uses the common pair vectors directly;
- CGCNN: maps common bonds into the upstream fixed-neighbor tensor format; padding follows the original CGCNN convention;
- ALIGNN: creates the DGL atom graph from common bonds and then constructs the official line graph / bond-angle features;
- MatGL M3GNet: creates a PyG graph from common bonds; M3GNet then builds its normal internal three-body graph.

The configured `max_neighbors` is a soft kth-shell cap. CGCNN's tensor slot count is therefore set to the maximum realized degree of the v2.1 cache, not blindly to 36.

## Unseen-element OOD

MatGL's element vocabulary is fixed to Z=1..118 as input-space metadata. Test labels are never used to build the vocabulary. Elements unseen during training therefore have untrained embeddings but can still be evaluated instead of crashing the OOD benchmark. SchNetPack likewise uses a 119-row nuclear embedding so the project Z<=118 contract is respected.

## CGCNN

`--cgcnn-repo` points to the original repository checkout. The older `--cgcnn-atom-init` argument is accepted only for command compatibility; v2.1 uses the project's common 14-D elemental descriptors so all architecture inputs remain aligned.
