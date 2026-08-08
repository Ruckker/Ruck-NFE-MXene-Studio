# Publication claim boundaries

This note defines the strongest claims that the audited benchmark can support without overstating the evidence.

## 1. Pseudo-label prediction is not physical NFE proof

`NFE_Pseudo_Score` and the low/medium/high NFE class are electronic-structure-derived pseudo-targets. The class is partly defined from the score, so classification and score regression are correlated tasks rather than independent ground truths.

Safe claim: the model reproduces/generalizes the fixed pseudo-label definition on held-out chemistry/structures.

Unsafe claim: high predicted class/score alone proves a real near-free-electron state.

Physical NFE claims require the independently reviewed/DFT-verified subset, with evidence such as vacuum/surface localization, band-resolved charge density, parabolic dispersion and effective-mass analysis as appropriate. For the paper-frozen verified layer, those physical evidence files must themselves exist under the declared evidence root and be SHA256-bound into the frozen review manifest. A reviewer label or spreadsheet entry without the corresponding hash-bound evidence bytes is not sufficient physical proof.

## 2. Architecture track is a matched backbone-family comparison, not a mathematical isolation of architecture

The `architecture` track matches the task, split, optimizer/scheduler, nominal hidden/layer budget, graph budget and pure-supervised class+score objective. Different backbones still retain architecture-native parameterization, radial/angular bases, embedding choices and different exact parameter counts.

Safe claim: under the matched benchmark protocol, the Ruck-NFE matched PaiNN-style backbone outperforms or underperforms the compared backbone families by the reported metrics/paired statistics.

Unsafe claim: every observed delta is caused solely by one abstract architectural primitive.

Report parameter counts and model-specific protocol metadata alongside performance. Formal preflight additionally requires the result identity to agree with the fitted checkpoint bytes and its internal track/model/seed/protocol identity.

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

Formal ablation paper tables use validation-temperature-calibrated classification probabilities for the registered probability/ranking metrics. Raw-probability metrics may be retained as diagnostics but must not be substituted silently for the calibrated paper estimand.

## 6. OOD terminology

`OOD_Unseen_Metal_Pair`, `OOD_Unseen_Termination_Pair`, `OOD_Unseen_X_Element` and `OOD_Unseen_Element` are chemistry OOD slices defined only relative to the training split.

`OOD_Large_Cell_Representation` is a representation-size stress slice based on the preregistered train-only `N_Atoms` quantile. It is not, by itself, a chemistry OOD test. Exact supercell/basis/vacuum invariance is audited separately. Paper mode fixes the large-cell quantile and representation-drift tolerances before inspecting results.

## 7. Screening metrics and statistical claims

Precision/Recall/Enrichment at K are screening-budget metrics. Boundary ties use expected random tie breaking so all-equal or tied scores do not depend on CSV row order.

Mean ± standard deviation across five independent seeds describes run-to-run variability. Model-to-model inferential claims should use paired per-sample/`Split_Group` analyses where appropriate; do not infer significance from overlapping/non-overlapping standard deviations alone.

Formal nested paired bootstrap comparisons require the same signed prediction data identity, matched training seeds and the same samples/truth. Comparison direction is preregistered. The observed point estimate is the mean of complete seed-level paired deltas, and High-class Enrichment@5% is included with the other registered primary endpoints. Classification macro statistics exclude bootstrap resamples that lose a class and report the valid iteration fraction.

Formal probability/ranking statistics use the validation-fitted temperature-calibrated probabilities. Calibration is fitted without test labels; it is not an after-the-fact test-set optimization.

## 8. Verified-NFE review should be independent of the model prediction

To minimize confirmation bias, verified NFE labels/evidence should be assigned before reviewers inspect the model's predicted class/probabilities/score for those structures. The model prediction should be joined only after the verification table is frozen/versioned.

If the review was not blinded to model predictions, disclose this limitation rather than calling the subset an independent validation set.

The frozen review binds selection membership, blinding protocol, reviewer outputs and the actual declared physical-evidence file hashes. This supports provenance of the verification process; it does not by itself guarantee that the physical interpretation is scientifically correct, which still depends on the quality and appropriateness of the reviewed DFT evidence.

## 9. Prediction and fitted-model files are scientific artifacts

Formal verified/OOD/paired statistics should use signed prediction CSVs. The prediction manifest binds the CSV bytes to dataset/structure/target/cache/normalizer/split/Git/run identity. Editing or substituting the CSV invalidates the manifest.

The formal signing/preflight path recomputes the registered paper metrics from the signed CSV and requires agreement with the sibling result rather than trusting metric JSON alone. This includes High AP, Enrichment@5%, ECE and score MAE/RMSE/Spearman/R².

Neural paper results must bind to the fitted checkpoint bytes and internal run identity. Formal XGBoost results persist the classifier/regressor boosters as UBJ files; preflight independently hashes those files and reconstructs the fitted-state identity from the stored bytes.

`paper_preflight_strict` is a whole-campaign gate. A valid single result cannot produce a legitimate `paper_ready: true`; the complete preregistered model/ablation roster and seed matrix must be present under one compatible scientific data identity.

Unsigned prediction files or unbound fitted-model files may be useful for exploratory debugging but should not be used for formal paper tables or statistical claims.

## 10. Conditional generator outputs are candidates, not physical discoveries

The formal surface generator is trained from the same audited v2 data/cache contract. Validation/test generation evaluation uses templates drawn only from the training catalog; an evaluation topology unsupported by that training catalog is rejected rather than borrowing a template from validation/test. Distributed evaluation uses exact non-padded shards so duplicated padding samples cannot improve or distort reported validation/test metrics.

The formal generator checkpoint records benchmark provenance, split/cache/code/Git identity, the generator protocol fingerprint and its train-only template-source policy. Formal generation also requires compatible audited predictor provenance and records the generator and predictor checkpoint SHA256 identities in `run_info.json`.

Safe claim: the generator proposes structures that pass the declared structural filters and receive the reported predictor-based screening scores under the bound model/checkpoint identities.

Unsafe claim: a generated structure is physically stable, synthesizable or a true NFE material merely because generation succeeded or the predictor assigned a high score. Those claims require the corresponding higher-fidelity relaxation/electronic-structure/experimental evidence.

## 11. CLI and Windows inference share one scientific graph/checkpoint contract

Canonical prediction, formal generation and the Windows backend use the same pair-symmetric periodic graph/data-v2 and guarded predictor-checkpoint contract. The packaged Windows runtime may lack a Git worktree, so only the requirement to compare against a live runtime Git checkout is relaxed in that packaged context; the checkpoint's recorded training Git identity, data/protocol provenance and feature-builder environment checks remain part of the scientific contract.

Therefore Windows/GUI output may be described as using the same audited inference semantics as the canonical application path when the packaged dependencies/checkpoints satisfy those guards. It should not be described as an independent validation implementation.
