# Official-upstream baseline adapters

This track complements, rather than renames, the lightweight controlled baselines. Each adapter
uses the upstream model implementation while preserving this project's fixed `Split_Group`, class +
NFE-score supervision, validation-only checkpoint selection and final metrics.

- `cgcnn_official`: original `txie-93/cgcnn` `CrystalGraphConvNet`; the adapter constructs CGCNN
  neighbor tensors and replaces only the final scalar regression layer by a four-output NFE head.
- `schnet_official`: `schnetpack.representation.SchNet` from SchNetPack 2.2.0 + project dual head.
- `alignn_official`: `alignn.models.alignn.ALIGNN` with its real DGL line graph and four outputs.
- `m3gnet_official`: `matgl.models.M3GNet` (MatGL 4.x / PyG) configured for four outputs.

The common NFE output is `[logit_low, logit_medium, logit_high, normalized_score]`. This is an
**official upstream backbone comparison**, not a claim that the upstream package natively implements
NFE classification.

Keep these environments separate from the main project because current upstream Python/DGL/PyG
requirements differ. See the per-backend requirement files. CGCNN additionally needs a checkout of
`https://github.com/txie-93/cgcnn` and its `atom_init.json`.

Example:

```bash
python -m training.baselines.official.run \
  --model schnet_official --seeds 2027,2028,2029,2030,2031 --device cuda
```

For CGCNN:

```bash
python -m training.baselines.official.run \
  --model cgcnn_official \
  --cgcnn-repo /path/to/cgcnn \
  --cgcnn-atom-init /path/to/cgcnn/data/sample-regression/atom_init.json \
  --device cuda
```
