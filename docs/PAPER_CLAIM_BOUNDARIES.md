# Publication claim boundaries

This note defines the strongest claims that the audited benchmark can support without overstating the evidence.

## 1. Pseudo-label prediction is not physical NFE proof

`NFE_Pseudo_Score` and the low/medium/high NFE class are electronic-structure-derived pseudo-targets. The class is partly defined from the score, so classification and score regression are correlated tasks rather than independent ground truths.

Safe claim: the model reproduces/generalizes the fixed pseudo-label definition on held-out chemistry/structures.

Unsafe claim: high predicted class/score alone proves a real near-free-electron state.

Physical NFE claims require the independently reviewed/DFT-verified subset, with evidence such as vacuum/surface localization, band-resolved charge density, parabolic dispersion and effective-mass analysis as appropriate.

## 2. Architecture track is a matched backbone-family comparison, not a mathematical isolation of architecture

The `architecture` track matches the task, split, optimizer/scheduler, nominal hidden/layer budget, graph budget and pure-supervised class+score objective. Different backbones still retain architecture-native parameterization, radial/angular bases, embedding choices and different exact parameter counts.

Safe claim: under the matched benchmark protocol, the Ruck-NFE matched PaiNN-style backbone outperforms or underperforms the compared backbone families by the reported metrics/paired statistics.

Unsafe claim: every observed delta is caused solely by one abstract architectural primitive.

Report parameter counts and model-specific protocol metadata alongside performance.

## 3. Official-upstream track uses official backbones/operators with project adapters

CGCNN, SchNetPack SchNet, ALIGNN and MatGL M3GNet use upstream message-passing backbones/operators, but the project supplies the fixed split, common graph adapter, NFE heads, optimizer protocol and calibration.

CGCNN additionally uses the upstream ConvLayer parameters/BatchNorms/nonlinearities on a ragged common-edge scatter implementation to eliminate the original unmasked padding-neighbor artifact.

Safe names: `CGCNN (official backbone)`, `SchNetPack SchNet (official backbone)`, `ALIGNN (official backbone)`, `MatGL M3GNet (official backbone)`.

Unsafe wording: `official untouched implementation/training pipeline`.

## 4. Full-system vs baseline is a system-level comparison

The full model differs from pure-supervised baselines not only in backbone details but also in auxiliary regression, SSL objectives and the SSL-dominant joint-training schedule.

Safe claim: the complete NFE-specific system reaches the reported performance relative to the baseline systems.

Unsafe claim: the full-system delta proves that vector features, SSL, auxiliary properties or global slab descriptors individually caused the gain.

Individual component claims must use the paired ablation matrix with the documented causal comparisons.

## 5. Correct ablation interpretations

Use:
- `full vs no_denoise` for coordinate-denoising contribution;
- `no_denoise vs no_vector` for vector/directional information under a common no-denoise condition;
- `full vs no_self_supervision` for combined SSL contribution at the same full-system supervised schedule;
- `no_self_supervision vs matched_supervision` for auxiliary supervised-property contribution with SSL absent;
- `full vs no_global` for intrinsic global-information contribution while retaining global/readout capacity.

Do not call `full vs no_vector` a pure vector effect because vector denoising is necessarily removed as well.

## 6. OOD terminology

`OOD_Unseen_Metal_Pair`, `OOD_Unseen_Termination_Pair`, `OOD_Unseen_X_Element` and `OOD_Unseen_Element` are chemistry OOD slices defined only relative to the training split.

`OOD_Large_Cell_Representation` is a representation-size stress slice based on train-only `N_Atoms` quantiles. It is not, by itself, a chemistry OOD test. Exact supercell/basis/vacuum invariance is audited separately.

## 7. Screening metrics and statistical claims

Precision/Recall/Enrichment at K are screening-budget metrics. Boundary ties use expected random tie breaking so all-equal or tied scores do not depend on CSV row order.

Mean ± standard deviation across five independent seeds describes run-to-run variability. Model-to-model inferential claims should use paired per-sample/`Split_Group` analyses where appropriate; do not infer significance from overlapping/non-overlapping standard deviations alone.

Paired bootstrap comparisons require the same signed prediction data identity and the same samples/truth. Classification macro statistics exclude bootstrap resamples that lose a class and report the valid iteration fraction.

## 8. Verified-NFE review should be independent of the model prediction

To minimize confirmation bias, verified NFE labels/evidence should be assigned before reviewers inspect the model's predicted class/probabilities/score for those structures. The model prediction should be joined only after the verification table is frozen/versioned.

If the review was not blinded to model predictions, disclose this limitation rather than calling the subset an independent validation set.

## 9. Prediction files are scientific artifacts

Formal verified/OOD/paired statistics should use signed prediction CSVs. The prediction manifest binds the CSV bytes to dataset/structure/target/cache/normalizer/split/Git/run identity. Editing or substituting the CSV invalidates the manifest.

Unsigned prediction files may be useful for exploratory debugging but should not be used for formal paper tables or statistical claims.
