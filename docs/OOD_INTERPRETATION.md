# OOD interpretation boundaries

The formal OOD slices are defined relative to the fixed training split. Their names describe **which categorical identity/combination was absent from training**, not an absence of all chemical prior information.

## Unseen element

`OOD_Unseen_Element` means that an atomic-number identity present in the test structure does not occur in the training structures used by that model run.

However, project graph models also receive fixed 14-D elemental descriptors derived from periodic-table properties. Therefore an unseen element can still be represented through those descriptors even when its learned identity embedding has never been updated by training.

Safe wording:

> generalization to structures containing element identities absent from the training set, with fixed periodic-table descriptors available as side information.

Unsafe wording:

> zero-shot prediction for completely unknown chemistry with no prior information.

For official MatGL/SchNet adapters, unseen identity embeddings may remain untrained/random, while the project adapter/input still exposes the common fixed elemental descriptors where applicable. Report this input contract explicitly.

## Unseen metal pair / termination pair

These slices mean the encoded pair combination is absent from training. They do not imply that the constituent metals/terminations are individually unseen. Performance therefore measures **combinatorial extrapolation**, not necessarily elemental extrapolation.

If the dataset encoding treats top/bottom ordering as distinct, retain that definition consistently in the OOD manifest and describe it. Do not silently reinterpret an ordered Janus pair as an unordered chemical set after results are observed.

## Unseen X element

`OOD_Unseen_X_Element` is specific to the MXene core X-site identity as encoded by the dataset metadata. It is a chemically motivated subgroup of unseen/combinatorial extrapolation, not an independent statistical test if it overlaps substantially with the general unseen-element slice.

## Large-cell representation stress

`OOD_Large_Cell_Representation` is based on train-only `N_Atoms` quantiles. It tests sensitivity to representation size/cell complexity. It is not a chemical OOD claim and should not be pooled with chemistry-OOD categories as if they were independent datasets.

Exact in-plane supercell/basis/vacuum invariance is a separate deterministic representation audit.

## Overlapping slices

A single test structure may belong to several OOD slices. Report per-slice support and do not sum supports as if slices were disjoint. If many correlated slices are tested, treat the analysis as secondary/descriptive unless a multiple-comparison plan was predeclared.
