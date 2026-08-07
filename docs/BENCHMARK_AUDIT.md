# Benchmark audit v2

This document defines the formal benchmark contract after the 2026-08 audit.

## Non-negotiable contracts

1. Dataset rows use the fixed group-aware split; `Split_Group` cannot cross train/validation/test.
2. Formal results require matching dataset SHA256 and split-manifest SHA256.
3. Graph semantics are `nfe-mxene-cache-2.0`, `intensive-slab-v2`, and
   `radius-shell-complete-v2`.
4. Validation selects checkpoints; test is evaluated only after selection. Temperature calibration
   is fitted on validation after checkpoint selection.
5. Per-sample predictions are retained so model comparisons can be paired.
6. Full-system mean±std requires independently trained checkpoints, not repeated evaluation of one
   checkpoint with different seed labels.

## Why cache v2 is incompatible with legacy weights

The old global vector encoded raw in-plane lattice lengths and `log(N_atoms)`, so an exact 1×1 →
3×3 replication changed model inputs. The v2 vector contains only intensive slab/cell statistics.
Neighbor truncation also changed from a hard top-k to a soft cap that retains the complete k-th
distance shell. Both changes are scientifically desirable but alter the learned input semantics.
Therefore old predictor weights are archival artifacts and must not enter new formal comparisons.

## Claim hierarchy

- Pseudo-label benchmark: measures reproduction/generalization of the fixed electronic-structure
  pseudo-label definition.
- Official-backbone benchmark: asks whether gains remain relative to upstream CGCNN/SchNet/ALIGNN/
  M3GNet backbones under a common NFE task adapter.
- Verified-NFE subset: tests independently inspected surface/vacuum localization and dispersion
  evidence.
- OOD slices: test chemistry and cell-size extrapolation.
- Paired Split_Group bootstrap: quantifies uncertainty of model-to-model differences on the fixed
  test set.

A high pseudo-label score alone is not sufficient evidence that a material hosts a real NFE state.
