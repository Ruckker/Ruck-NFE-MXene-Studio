# Benchmark audit v2.1

This document defines the formal benchmark contract after the 2026-08 repeated audit.

## Non-negotiable contracts

1. Dataset rows use the fixed group-aware split; `Split_Group` cannot cross train/validation/test.
2. Formal results require matching dataset-table SHA256, **actual structure-file manifest SHA256**, and split-manifest SHA256.
3. Graph semantics are `nfe-mxene-cache-2.1`, `intensive-slab-v2`, and `radius-shell-complete-v2`.
4. Formal aggregated runs must come from one resolvable, clean Git commit. Dirty/unknown worktrees are retained only for debugging, not paper tables.
5. Validation selects checkpoints; test is evaluated only after selection. Temperature calibration is fitted on validation after checkpoint selection.
6. Per-sample predictions are retained so model comparisons can be paired.
7. Full-system and ablation mean±std require independently trained checkpoints, not repeated evaluation of one checkpoint under different seed labels.
8. Neural architecture/official-upstream comparisons use the same class+NFE-score supervision, optimizer schedule, and the same early supervised weighting window as the `matched_supervision` ablation. Parameter counts are still reported because exact capacity equality is not assumed.

## Why cache v2.1 is incompatible with legacy weights

The legacy global vector encoded raw in-plane cell lengths and atom count, so exact 1×1 → 3×3 replication changed model inputs. The v2 family uses only intensive slab/cell statistics. Neighbor truncation is a soft cap that retains the complete kth distance shell. v2.1 additionally fingerprints the bytes of every referenced POSCAR/CIF; editing a structure while leaving the CSV unchanged now invalidates the cache and formal provenance.

These changes alter learned input semantics. Legacy predictor weights and v2.0 weights are archival artifacts and must not enter v2.1 formal comparisons.

## Public entrypoints

`python -m nfe_model.train` and the launcher under `training/entrypoints/train.py` both install the audited v2.1 contract. `python -m nfe_model.predict` and the production prediction launcher both enforce v2.1 checkpoint/data/graph provenance. The historical implementations live in explicitly named internal modules (`train_core.py`, `predict_core.py`) so accidental CLI bypass is not possible.

## Claim hierarchy

- Pseudo-label benchmark: reproduction/generalization of the fixed electronic-structure pseudo-label definition.
- Controlled architecture benchmark: architecture-only comparison under matched supervision and schedule.
- Official-upstream benchmark: official CGCNN/SchNet/ALIGNN/M3GNet backbones adapted to the **same v2.1 periodic edge list**. ALIGNN still builds a real line graph; MatGL M3GNet still builds its internal three-body line graph.
- Verified-NFE subset: independent manual/DFT labels; reviewed negative cases are retained rather than filtered out.
- OOD slices: chemistry and cell-size extrapolation.
- Paired `Split_Group` bootstrap: uncertainty on model-to-model differences.

A high pseudo-label score alone is not evidence that a material hosts a real NFE state.
