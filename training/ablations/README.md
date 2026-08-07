# NFE predictor ablation suite

Ablations reuse the fixed v2.3 cache/split, audited DDP evaluation, exact cache/normalizer provenance, calibration and final test protocol. They answer **which component of the full NFE-specific system causes a gain?**

| Key | Removed / retained |
|---|---|
| `full` | complete model |
| `no_vector` | removes vector/directional information while retaining capacity-matched interaction/readout parameters; vector-dependent denoising is necessarily disabled |
| `no_global` | replaces the 11 `intrinsic-slab-v3` global channels by zeros while retaining global encoder/readout capacity |
| `no_masked_pretrain` | no atom masking/objective |
| `no_denoise` | no coordinate noise/denoising |
| `no_self_supervision` | removes masked-atom + denoising; all supervised regression remains |
| `no_auxiliary_regression` | class + NFE score only, SSL remains |
| `matched_supervision` | class + NFE score only and no SSL; **full vector/global architecture remains** |
| `classification_only` | class supervision only |

Disabled objectives also lose their associated input corruption. Representation ablations are capacity preserving so removing vector/global information does not simultaneously shrink the matching interaction/readout capacity.

## Critical schedule rule

The full model uses a 35-epoch **SSL-dominant joint-training window** in which supervised losses are multiplied by 0.25, followed by supervised-dominant joint training. This is not pure self-supervised pretraining.

Within the **ablation matrix**, removing SSL retains that same first-35-epoch supervised weighting boundary. Otherwise `full vs no_self_supervision` would change both SSL and supervised optimization strength. `matched_supervision`, `classification_only` and no-SSL ablations therefore keep `pretrain_epochs=35` as a schedule boundary even if no SSL loss is active.

This rule is specific to causal ablation against the full system. The external `architecture` and `official-upstream` tracks are pure-supervised comparisons and correctly use a constant 1.0× supervised factor from epoch zero.

## Correct causal comparisons

Do **not** use `full vs no_vector` as a pure vector effect because `no_vector` also cannot perform vector coordinate denoising.

Use:
- `full vs no_denoise` → denoising contribution;
- `no_denoise vs no_vector` → vector/directional information contribution under the same no-denoise condition, with capacity-matched interaction/readout parameters;
- `full vs no_self_supervision` → total SSL contribution at the same full-system supervised schedule;
- `no_self_supervision vs matched_supervision` → auxiliary supervised-property contribution with SSL absent;
- `full vs no_global` → intrinsic global-information contribution with global/readout capacity retained.

Do **not** use `matched_supervision` as the pure architecture comparator against CGCNN/SchNet/ALIGNN/M3GNet. It retains the full model's global-information branch and heteroscedastic head machinery. The architecture-only comparator is `painn` in `training/baselines/run.py`, trained on class + NFE score only without global features, auxiliary targets or SSL.

## Formal aggregation

By default every ablation requires the same five seeds as `full`. Formal aggregation requires:
- one distinct checkpoint SHA256 for every ablation/seed row, including no reuse across different ablations;
- identical dataset-table, structure-byte, target, exact cached-tensor, train-normalizer, split and graph provenance;
- current `nfe-mxene-cache-2.3` / `intrinsic-slab-v3` semantics;
- one clean Git revision;
- one training protocol per ablation across seeds and distinct seed-specific experiment protocols.

Deltas versus full are paired by seed rather than subtracting unrelated means.

```bash
SEEDS=2027,2028,2029,2030,2031 EPOCHS=220 BATCH_SIZE=96 \
  bash training/ablations/run_4gpu.sh

python training/ablations/summarize.py
```
