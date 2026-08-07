# NFE predictor ablation suite

Ablations reuse the fixed v2.1 cache/split, audited DDP evaluation, structure-file provenance, calibration and final test protocol. They answer **which component of the full NFE-specific system causes a gain?**

| Key | Removed / retained |
|---|---|
| `full` | complete model |
| `no_vector` | scalar message passing only; vector-dependent denoising is necessarily disabled |
| `no_global` | removes the 11 intensive slab/global descriptors |
| `no_masked_pretrain` | no atom masking/objective |
| `no_denoise` | no coordinate noise/denoising |
| `no_self_supervision` | removes masked-atom + denoising; all supervised regression remains |
| `no_auxiliary_regression` | class + NFE score only, SSL remains |
| `matched_supervision` | class + NFE score only and no SSL; full vector/global architecture retained |
| `classification_only` | class supervision only |

Disabled objectives also lose their associated input corruption; a zero loss weight is not allowed to keep feeding corrupted inputs.

## Critical schedule rule

The full model uses a 35-epoch **SSL-dominant joint-training window** in which supervised losses are multiplied by 0.25, followed by supervised-dominant joint training. This is not pure self-supervised pretraining.

When SSL is removed, the first-35-epoch **supervised weighting schedule is retained**. Otherwise `full vs no_self_supervision` would change both SSL and supervised optimization strength. `matched_supervision`, `classification_only`, and the no-SSL ablations therefore keep `pretrain_epochs=35` as a schedule boundary even though no SSL loss may be active. Architecture/official-upstream neural baselines use the same supervised schedule.

## Correct causal comparisons

Do **not** use `full vs no_vector` as a pure vector effect because `no_vector` also cannot perform vector coordinate denoising.

Use:
- `full vs no_denoise` → denoising contribution;
- `no_denoise vs no_vector` → vector representation contribution under the same no-denoise condition;
- `full vs no_self_supervision` → total SSL contribution at the same supervised schedule;
- `no_self_supervision vs matched_supervision` → auxiliary supervised-property contribution with SSL absent;
- `matched_supervision` vs architecture/official-upstream neural baselines → architecture comparison under class+score supervision and the same supervised weighting schedule;
- `full vs no_global` → intensive slab/global information contribution.

## Formal aggregation

By default every ablation requires the same five seeds as `full`. Each row must have a distinct checkpoint SHA256 and matching dataset-table, structure-manifest, split, graph, and clean Git provenance. Deltas versus full are paired by seed rather than subtracting unrelated means.

```bash
SEEDS=2027,2028,2029,2030,2031 EPOCHS=220 BATCH_SIZE=96 \
  bash training/ablations/run_4gpu.sh

python training/ablations/summarize.py
```
