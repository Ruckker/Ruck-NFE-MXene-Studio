# NFE baseline benchmark suite

This suite uses the same fixed `Suggested_Split` / `Split_Group`, graph/cache semantics and target
definitions as the NFE predictor. Random row resplitting is prohibited.

## Three comparison tracks

### 1. `architecture`

Answers: **what does architecture alone contribute under matched supervision?**

All neural models use NFE class + NFE pseudo-score only, no auxiliary electronic targets, no masked
atom objective and no coordinate denoising. The nominal budget is matched (192 hidden channels,
6 layers where applicable, 220 epochs, AdamW 3e-4, 8-epoch warmup, validation-only early stopping).
Parameter counts are always reported; “matched budget” does not mean exactly identical parameter
counts.

Keys:
- `cgcnn_controlled` — compact CGCNN-style control;
- `schnet_controlled` — compact SchNet-style control;
- `angle_moment` — internal angle-moment control. **Not ALIGNN**;
- `state_threebody` — internal state/three-body-moment control. **Not M3GNet**;
- `painn` — Ruck-NFE scalar/vector backbone under the same class+score supervision;
- Dummy and structure-only XGBoost lower bounds.

### 2. `official-upstream`

Uses actual upstream backbones through adapters in `training/baselines/official/`: original CGCNN,
SchNetPack SchNet, ALIGNN with its line graph, and MatGL M3GNet. Each upstream backbone receives the
same fixed split and common four-output NFE objective. The NFE head/data adapter is project code, so
report these as **official upstream backbone + NFE adapter**, not as an upstream package's native NFE
task.

Keep those dependencies in isolated environments; see `official/README.md`.

### 3. `full-system`

Answers: **what does the complete NFE-specific system achieve?** It evaluates five independently
trained audited `full` checkpoints and verifies internal seed, checkpoint SHA256, dataset hash, split
hash, cache schema, global-feature schema and neighbor policy. Reusing one checkpoint under five seed
labels is rejected.

## Graph/cache v2

Formal runs use:
- `nfe-mxene-cache-2.0`;
- `intensive-slab-v2` global features (no raw in-plane `a/b` or `N_atoms` shortcut);
- `radius-shell-complete-v2` neighbor selection (the `max_neighbors` value is a soft cap and a
  degenerate distance shell is never cut in half).

An old cache is rebuilt automatically. Old checkpoints must not be mixed into the formal v2 tables.

## Outputs and metrics

Every run saves `result.json`, validation/test predictions and graph-model checkpoints/history where
applicable. Prediction CSVs include structure ID, `Split_Group`, true/predicted class probabilities,
true/predicted NFE score and absolute score error.

The paper tables include macro F1, balanced accuracy, ROC-AUC, Average Precision, high-class
Precision/Recall/Enrichment at top fractions, score MAE/RMSE/Spearman/R² and calibrated ECE.
Accuracy alone is not a primary metric because the medium class dominates.

## Commands

```bash
python training/baselines/export_manifest.py --rebuild-cache
python training/baselines/run.py --track architecture --model painn \
  --seeds 2027,2028,2029,2030,2031 --device cuda

SEEDS=2027,2028,2029,2030,2031 bash training/baselines/run_4gpu.sh

python training/baselines/run.py --track full-system --model ours_full \
  --seeds 2027,2028,2029,2030,2031 --device cuda

python training/baselines/summarize.py
```

For official models, see `training/baselines/official/README.md`.

## Leakage policy

XGBoost consumes only composition/element descriptors and structural geometry. It must never receive
NFE candidate fields, DOS, work function, band gap, ELF, charge density, Fermi level or any target
column. Normalizers and vocabularies are fitted on train only. Dataset/split provenance is mandatory
for formal aggregation.

## Statistical and physical validation

Do not interpret pseudo-label benchmark performance as independent proof of a real NFE state. Use
`training/evaluation/` for DFT/manual verified NFE cases, chemistry/cell-size OOD slices, paired
`Split_Group` bootstrap and exact-supercell consistency.
