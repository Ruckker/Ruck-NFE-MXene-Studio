# Benchmark audit v2.3

This document defines the formal benchmark contract after the 2026-08 repeated audit.

## Non-negotiable contracts

1. Dataset metadata is fail-fast: `Structure_Name` is unique/non-empty, `File_Path` and `Split_Group` are non-empty, and `Suggested_Split` resolves explicitly to train/validation/test. Unknown splits are never mapped to train.
2. `Split_Group` cannot cross train/validation/test. The canonical fixed split must contain all three NFE classes and a finite primary `NFE_Pseudo_Score` for every retained row. OOD/verified slices may legitimately miss classes and use NaN-aware metrics.
3. Formal results lock the dataset CSV, actual structure-file bytes, exact cached graph/feature/target tensors, target specification, split manifest, train-fitted normalizer tensors, graph budget, source-code data contract and clean Git revision.
4. Current graph semantics are `nfe-mxene-cache-2.3`, `intrinsic-slab-v3`, and `radius-shell-complete-v2`.
5. `max_neighbors` is a soft kth-shell cap: the complete degenerate distance shell is retained. Formal slab graphs additionally require the atom-free normal vacuum gap to be strictly larger than the graph cutoff, preventing artificial cross-vacuum neighbors under 3D PBC.
6. The 11 global descriptors are intrinsic slab/composition quantities. They are designed to be invariant to exact in-plane supercell replication, atom ordering, equivalent unimodular in-plane basis choices and added vacuum for the same Cartesian slab. These claims are covered by model-level regression tests and the representation-consistency acceptance tool.
7. Formal aggregation requires one resolvable clean Git commit. Production inference requires runtime code to equal checkpoint training code. Predictor continuation (`--resume`) additionally requires the same seed-specific experiment-protocol hash.
8. The actual train normalizer tensors are fingerprinted. Isolated official-backbone environments may read the same immutable cache, but results cannot be mixed if their resulting normalization tensors differ.
9. Validation selects checkpoints; test is evaluated only after selection. Temperature calibration is fitted on validation after checkpoint selection. Per-sample validation/test predictions are retained for paired analysis.
10. Full-system and ablation mean±std require independently trained checkpoints with distinct checkpoint hashes. Copied/relabelled checkpoints are rejected.
11. Architecture and official-upstream neural baselines are **pure supervised** class+NFE-score models and use a constant 1.0 supervised factor from epoch zero. They share the same optimizer/scheduler, nominal hidden/layer budget and five-seed matrix. Exact scalar parameter equality is not assumed, so parameter counts are reported.
12. The full system uses the historical 35-epoch SSL-dominant joint window (0.25× supervised factor) followed by supervised-dominant joint training. No-SSL/matched-supervision **ablations retain that window only for causal isolation against full**; it is not imposed on external architecture baselines.

## Why v2.3 is incompatible with older formal weights

Older global vectors encoded computational-cell choices such as raw lengths, atom count, lattice angles or vacuum/cell-height information. Those features can change for an equivalent material representation. v2.3 replaces them with `intrinsic-slab-v3`, fingerprints the exact cache tensors and normalization tensors, validates target ordering/transforms, and rejects slab cells whose cutoff crosses the vacuum gap.

These changes alter learned input semantics. v1/v2.0/v2.1/v2.2 predictor weights are archival artifacts and must not enter v2.3 formal comparisons.

## Public entrypoints and internal cores

Use `python -m nfe_model.train` (or `training/entrypoints/train.py`) and `python -m nfe_model.predict` (or `training/entrypoints/predict.py`) for formal work. The legacy `nfe_model.train_audited` path is only a compatibility alias to the same formal trainer.

`train_core.py` and `predict_core.py` are implementation modules, not supported formal entrypoints. Their outputs must not bypass v2.3 provenance/result gates.

## Architecture and official-upstream tracks

The architecture track answers: **what does the backbone contribute under common pure-supervised class+score training?** Internal controlled models are named as controlled reimplementations, not as official ALIGNN/M3GNet.

The official-upstream track uses upstream message-passing backbones with project task heads/adapters and the same audited periodic edge list. Formal package/source identities are pinned: SchNetPack 2.2.0, ALIGNN 2026.5.20, MatGL 4.0.3, and a clean exact-commit `txie-93/cgcnn` checkout. CGCNN's fixed neighbor tensor width is derived from train only; validation/test structures exceeding it fail instead of being truncated.

`painn` in the architecture track is the architecture-only Ruck-NFE comparator. `matched_supervision` is a full-architecture supervision ablation and is **not** substituted for this external architecture comparison because it retains the full global branch and heteroscedastic head machinery.

## Ablation interpretation

`no_vector` and `no_global` are capacity-preserving information ablations: disabled information is zeroed while matching interaction/readout capacity remains. Because vector denoising is impossible without vector information:

- `full vs no_denoise` → coordinate-denoising contribution;
- `no_denoise vs no_vector` → vector/directional information contribution under a common no-denoise condition;
- `full vs no_self_supervision` → combined SSL contribution at the same full-system supervised schedule;
- `no_self_supervision vs matched_supervision` → auxiliary supervised-property contribution with SSL absent;
- `full vs no_global` → intrinsic global-information contribution with global/readout capacity retained.

Do not interpret `full vs no_vector` as a pure vector effect.

## Metrics and statistics

Formal classification/ranking reporting includes macro F1, balanced accuracy, ROC-AUC, average precision and high-class Precision/Recall/Enrichment at fixed screening fractions. Fixed-budget screening metrics are tie-invariant: boundary ties use expected random tie breaking rather than row-order-dependent truncation.

Score reporting includes MAE, RMSE, Spearman correlation and R². Model-to-model uncertainty is estimated with paired `Split_Group` block bootstrap; prediction files must contain the same samples and truth. Bootstrap classification macro metrics are excluded for resamples that lose a class, and the valid-iteration fraction is reported.

## Verified NFE and OOD claims

The pseudo-label benchmark measures reproduction/generalization of a fixed electronic-structure-derived pseudo-label definition. NFE class and NFE pseudo-score are mathematically coupled targets and must not be described as independent physical ground truths.

The verified-NFE subset is a separate manual/DFT evidence layer. Review completion is distinct from whether evidence supports NFE, so reviewed negative cases remain eligible. Formal verified evaluation requires complete prediction coverage unless an explicitly exploratory partial-coverage flag is used.

OOD chemistry slices are defined from train-only metal-pair, termination-pair, X-element and element vocabularies. The large-`N_Atoms` slice is a **representation-size stress test**, not a chemical OOD claim; exact representation invariance is audited separately.

A high pseudo-label score alone is not evidence that a material hosts a real NFE state.

## Uncertainty terminology

The historical validation residual quantile was called `conformal_score_radius`, but validation also participates in checkpoint selection. v2.3 therefore records it as `empirical_validation_score_radius` with:

- `score_interval_method = validation-residual-plus-mc-normal-heuristic`;
- `score_interval_coverage_guarantee = false`.

It is a screening heuristic, not a split-conformal coverage guarantee.
