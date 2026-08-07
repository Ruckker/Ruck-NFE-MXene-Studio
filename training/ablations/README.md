# NFE predictor ablation suite

For final paper runs use the canonical dispatcher or the fixed 4-GPU launcher:

```bash
python -m training.paper ablation --ablation full --seed 2027
bash training/ablations/run_4gpu.sh
```

Do not add epoch/batch/config overrides to paper runs. Shortened smoke tests belong to `training.formal_v2_4`, not the paper table.

All paper ablations reuse the fixed v2.4 pair-symmetric cache/split, exact cache/normalizer provenance, calibration and final test protocol.

| Key | Removed / retained |
|---|---|
| `full` | complete model |
| `no_vector` | removes vector/directional information while retaining capacity-matched interaction/readout parameters; vector-dependent denoising is necessarily disabled |
| `no_global` | replaces the 11 `intrinsic-slab-v3` global channels by zeros while retaining global encoder/readout capacity |
| `no_masked_pretrain` | no atom masking/objective |
| `no_denoise` | no coordinate noise/denoising |
| `no_self_supervision` | removes masked-atom + denoising; supervised regression remains |
| `no_auxiliary_regression` | class + NFE score only, SSL remains |
| `matched_supervision` | class + NFE score only and no SSL; full vector/global architecture remains |
| `classification_only` | class supervision only |

Representation ablations retain matching parameter/readout capacity so removal of vector/global information is not simultaneously a parameter-count reduction.

## Schedule rule

The full system uses a 35-epoch **SSL-dominant joint-training window** with supervised losses multiplied by 0.25, followed by supervised-dominant joint training. This is not pure self-supervised pretraining.

Within the causal ablation matrix, removing SSL retains the same 35-epoch supervised weighting boundary. Otherwise `full vs no_self_supervision` would change both SSL and supervised optimization strength. External architecture/official tracks instead use a constant 1.0× supervised factor from epoch zero.

## Correct causal comparisons

Do not use `full vs no_vector` as a pure vector effect because vector coordinate denoising is also unavailable in `no_vector`.

Use:

- `full vs no_denoise` → denoising contribution;
- `no_denoise vs no_vector` → vector/directional information contribution under matched no-denoise conditions;
- `full vs no_self_supervision` → total SSL contribution under the retained full-system supervised schedule;
- `no_self_supervision vs matched_supervision` → auxiliary supervised-property contribution with SSL absent;
- `full vs no_global` → intrinsic global-information contribution with capacity retained.

`matched_supervision` is not the architecture-only comparator against external backbones. That role belongs to the `painn` model in the architecture track.

## Final graph/provenance contract

Paper-ready ablations require:

- `nfe-mxene-cache-2.4`;
- `intrinsic-slab-v3`;
- `radius-shell-complete-pair-symmetric-v3`;
- zero skipped cache rows;
- identical dataset-table, structure-byte, target, exact cache tensor, train-normalizer and split identities;
- one clean Git revision;
- one training protocol per ablation across seeds;
- one distinct seed-specific experiment protocol/checkpoint per run.

The full preregistered nine-ablation set and the same five seeds `2027–2031` are required. The paper summary rejects missing ablations, seed-set drift and checkpoint reuse across ablation/seed rows.

```bash
python -m training.paper ablation-summary
```

Deltas versus `full` are paired by seed rather than subtracting unrelated means.
