# NFE predictor ablation suite

Ablations reuse the fixed cache/split, audited DDP evaluation, provenance, calibration and final test
protocol. They answer **which part of the full NFE-specific system causes a gain?**

| Key | Removed / retained |
|---|---|
| `full` | complete model |
| `no_vector` | scalar message passing only; denoising is necessarily disabled |
| `no_global` | removes the 11 intensive slab/global descriptors |
| `no_masked_pretrain` | no atom masking/objective |
| `no_denoise` | no coordinate noise/denoising |
| `no_self_supervision` | removes both masked-atom and denoising; all supervised regression remains |
| `no_auxiliary_regression` | class + NFE score only, but SSL remains |
| `matched_supervision` | class + NFE score only **and no SSL**; full vector/global architecture retained |
| `classification_only` | class supervision only |

Disabled objectives also lose their input corruption; a zero loss weight is not allowed to keep
feeding corrupted inputs.

## Correct causal comparisons

Do **not** use `full vs no_vector` as a pure vector effect because `no_vector` also cannot perform
vector coordinate denoising.

Use:
- `full vs no_denoise` → denoising contribution;
- `no_denoise vs no_vector` → vector representation contribution under the same no-denoise condition;
- `full vs no_self_supervision` → total SSL contribution;
- `no_self_supervision vs matched_supervision` → auxiliary supervised property contribution when SSL
  is absent;
- `matched_supervision` vs architecture-track baselines → architecture comparison under class+score
  only supervision;
- `full vs no_global` → intensive slab/global information contribution.

## Training phases

The first configured 35 epochs of the full model are **SSL-dominant joint training**, not pure
self-supervised pretraining: supervised classification/regression remains active at reduced weight.
Afterward training becomes supervised-dominant while SSL remains active at reduced weight.

## Run

```bash
SEEDS=2027,2028,2029,2030,2031 EPOCHS=220 BATCH_SIZE=96 \
  bash training/ablations/run_4gpu.sh

python training/ablations/summarize.py
```

Formal ablation aggregation rejects mixed dataset hashes, split hashes, cache schemas, global-feature
schemas or neighbor policies.
