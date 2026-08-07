# Benchmark audit v2.1

This document defines the formal benchmark contract after the 2026-08 repeated audit.

## Non-negotiable contracts

1. Dataset metadata is fail-fast: `Structure_Name` must be unique/non-empty, `File_Path` and `Split_Group` must be non-empty, and `Suggested_Split` must resolve explicitly to train/validation/test. Unknown split labels are never silently moved into train.
2. `Split_Group` cannot cross train/validation/test.
3. Formal results require matching dataset-table SHA256, **actual structure-file manifest SHA256**, and split-manifest SHA256.
4. Graph semantics are `nfe-mxene-cache-2.1`, `intensive-slab-v2`, and `radius-shell-complete-v2`.
5. Formal aggregated runs must come from one resolvable, clean Git commit. Dirty/unknown worktrees are retained only for debugging, not paper tables.
6. Predictor continuation (`--resume`) requires the same seed-specific experiment-protocol hash. Multi-seed aggregation requires one common seed-independent training-protocol hash for each model/ablation.
7. Validation selects checkpoints; test is evaluated only after selection. Temperature calibration is fitted on validation after checkpoint selection.
8. Per-sample predictions are retained so model comparisons can be paired.
9. Full-system and ablation mean±std require independently trained checkpoints, not repeated evaluation of one checkpoint under different seed labels.
10. Neural architecture/official-upstream comparisons use the same class+NFE-score supervision, optimizer schedule, nominal hidden/layer budget, and early supervised weighting window. Parameter counts are still reported because exact scalar parameter equality is not assumed across different backbones.

## Why cache v2.1 is incompatible with legacy weights

The legacy global vector encoded raw in-plane cell lengths and atom count, so exact 1×1 → 3×3 replication changed model inputs. The v2 family uses only intensive slab/cell statistics. Neighbor truncation is a soft cap that retains the complete kth distance shell. v2.1 additionally fingerprints the bytes of every referenced POSCAR/CIF; editing a structure while leaving the CSV unchanged now invalidates the cache and formal provenance.

These changes alter learned input semantics. Legacy predictor weights and v2.0 weights are archival artifacts and must not enter v2.1 formal comparisons.

## Public entrypoints and internal cores

Use `python -m nfe_model.train` (or `training/entrypoints/train.py`) and `python -m nfe_model.predict` (or `training/entrypoints/predict.py`) for formal work. These install/enforce the v2.1 data, provenance, protocol and prediction guards.

`train_core.py` and `predict_core.py` are implementation modules, not supported formal entrypoints. `predict_core.py` refuses direct CLI execution. Any archival direct execution of `train_core.py` is outside the formal benchmark contract and its output must not be admitted by the v2.1 provenance/summarizer gates.

## Ablation interpretation

`no_vector` and `no_global` are capacity-preserving information ablations: disabled information channels are replaced by zeros while the matching parameter/readout capacity is retained. Because vector coordinate denoising is impossible when vector information is removed, use `no_denoise vs no_vector`—not `full vs no_vector`—to isolate the vector/directional contribution under a common no-denoise condition.

`matched_supervision` is a **full-architecture supervision ablation**. It retains the full model's global-information branch and heteroscedastic head machinery, so it is not the pure architecture comparator. The architecture-only Ruck-NFE comparator is `painn` in the architecture track.

## Uncertainty terminology

The historical trainer called the validation residual quantile a `conformal_score_radius`. That name is not statistically justified because validation also participates in checkpoint selection. Audited v2.1 artifacts relabel it as `empirical_validation_score_radius` and record:

- `score_interval_method = validation-residual-plus-mc-normal-heuristic`;
- `score_interval_coverage_guarantee = false`.

The prediction interval is a screening heuristic, not a split-conformal coverage guarantee. The configured nominal confidence level controls the normal multiplier used for the MC/aleatoric component and is explicitly written to prediction output.

## Claim hierarchy

- Pseudo-label benchmark: reproduction/generalization of the fixed electronic-structure pseudo-label definition.
- Controlled architecture benchmark: architecture-only comparison under matched class+score supervision and schedule.
- Official-upstream benchmark: official CGCNN/SchNet/ALIGNN/M3GNet backbones adapted to the **same v2.1 periodic edge list**. ALIGNN still builds a real line graph; MatGL M3GNet still builds its internal three-body line graph. CGCNN uses a fixed neighbor tensor width derived from the training split only and fails rather than truncates if a later structure exceeds it.
- Verified-NFE subset: independent manual/DFT labels; reviewed negative cases are retained rather than filtered out.
- OOD slices: chemistry and cell-size extrapolation.
- Paired `Split_Group` bootstrap: uncertainty on model-to-model differences.

A high pseudo-label score alone is not evidence that a material hosts a real NFE state.