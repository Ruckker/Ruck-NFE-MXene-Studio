# Statistical analysis plan for the NFE benchmark

This plan should be frozen before the final five-seed paper campaign so metric/reporting choices are not selected after seeing test results.

## Fixed data and runs

- Use the audited fixed train/validation/test split and one final clean Git commit.
- Use the same five predeclared training seeds `2027, 2028, 2029, 2030, 2031` for every stochastic model in a formal comparison.
- Independent GPU processes may execute a registered seed subset through `training.paper ... --seeds <subset>`; this is execution sharding only and does not change the preregistered five-seed estimand.
- Never choose a checkpoint, hyperparameter or model variant from test performance.
- Paper-ready results require zero skipped cache rows.
- Cross-split exact/source duplicates are forbidden. Any coarse near-duplicate candidate must have a fingerprint-keyed reviewed disposition before the split audit can pass.

## Primary endpoints

Use a small set of prespecified endpoints for headline comparisons:

1. **Macro average precision** — primary imbalanced three-class discrimination/ranking endpoint.
2. **Macro F1** — primary thresholded three-class classification endpoint.
3. **High-class enrichment at 5%** — primary screening-use endpoint.
4. **NFE pseudo-score MAE** — primary continuous pseudo-score endpoint for models trained on the score.

These endpoints answer different operational questions but are not independent physical ground truths. In particular NFE class and NFE pseudo-score are mathematically coupled pseudo-targets.

For formal paper tables, signed predictions and paired inference, probability/ranking metrics use the **validation-fitted temperature-calibrated probabilities**. Raw-probability metrics may be retained as diagnostics, but they are not interchangeable with the calibrated paper estimand. Score-regression metrics are unaffected by classification temperature scaling.

## Secondary/descriptive endpoints

Report as secondary or diagnostic metrics:
- balanced accuracy;
- macro ROC-AUC;
- per-class F1/recall/AP;
- Precision/Recall/Enrichment at 1%, 5%, 10%;
- NFE score RMSE, Spearman and R²;
- calibrated test ECE;
- parameter count, training time and runtime/environment metadata;
- support plus observed target/prediction ranges for bounded auxiliary regression properties.

Do not promote a secondary metric to the headline endpoint because it happens to give the most favorable result.

## Seed variability

For each model, report mean ± sample standard deviation across the five independent training seeds. This describes training-run variability; it is not a confidence interval over the test population.

Do not infer statistical significance from whether two mean±SD intervals overlap.

## Paired model comparisons

For a formal A-vs-B claim, use matched seed identities and the same signed test samples. Comparison direction is preregistered and is part of the estimand; A-minus-B must not be normalized into an unordered pair.

Preferred inference:
- `formal_multiseed_bootstrap.py`: nested resampling of training seed and `Split_Group` chemistry blocks;
- compute the observed point estimate as the mean of complete **seed-level paired deltas**, rather than pooling samples across independently trained seeds first;
- include all primary endpoints, including High-class Enrichment@5%;
- report the observed mean paired improvement and 95% bootstrap interval;
- report the valid-bootstrap fraction for macro classification metrics.

The older single-seed paired bootstrap is diagnostic only for within-seed sample uncertainty.

## Planned pairwise comparisons

Predeclare the comparisons that answer the scientific questions rather than testing every possible pair:

- Full system vs matched Ruck-NFE `painn` backbone — system-level gain from the NFE-specific training/supervision package, not a single-component attribution.
- Matched Ruck-NFE `painn` vs each controlled/official backbone — backbone-family comparison under the matched pure-supervised protocol.
- Full vs `no_denoise` — coordinate-denoising contribution.
- `no_denoise` vs `no_vector` — vector/directional-information contribution at common no-denoise condition.
- Full vs `no_self_supervision` — combined SSL contribution at the same full-system supervised schedule.
- `no_self_supervision` vs `matched_supervision` — auxiliary supervised-property contribution with SSL absent.
- Full vs `no_global` — intrinsic global-information contribution with capacity retained.

`classification_only` is a sanity endpoint with a different available checkpoint-selection objective; do not present its delta as a clean causal estimate of one isolated component.

## Multiple comparisons

If binary significance language is used across several planned baselines for the same primary endpoint, control family-wise error (for example Holm correction) or state clearly that intervals/effect sizes are descriptive. Avoid running dozens of post-hoc pairwise tests and reporting only favorable ones.

## Signed artifact and whole-campaign closure

Formal prediction CSVs are not trusted merely because a sibling JSON reports plausible metrics. The signing/preflight path recomputes the registered paper metrics from the CSV and requires agreement with the sibling result within the fixed tolerance, including High AP, Enrichment@5%, ECE, score MAE/RMSE/Spearman/R².

Neural result identity must agree with the fitted checkpoint bytes and its internal track/model/seed/protocol identity. XGBoost formal runs persist the fitted classifier/regressor boosters as UBJ artifacts; paper preflight hashes those files and reconstructs the fitted-state SHA from their bytes.

`paper_preflight_strict` is a **whole-campaign gate**. `paper_ready: true` is valid only when the complete preregistered baseline/official/full-system/ablation roster and seed matrix are present with one common scientific data identity. A valid single result is never sufficient.

Interrupted main-predictor training must resume from `last.pt`, which contains optimizer/scheduler/scaler/early-stopping state and per-rank Python/NumPy/Torch/CUDA RNG state. `best.pt` remains the model-selection artifact and is not the exact-interruption resume point.

## Verified-NFE subset

Primary verified analysis should use **all review-complete cases**. Reviewer-confidence thresholds such as 0.6/0.8/0.9 are sensitivity analyses, not alternative primary datasets chosen after seeing performance.

Generate the verified review queue without reading model predictions, supply reviewers a blinded sheet that omits pseudo-label/model prediction columns, freeze the completed sheet, and only then join predictions. Paper mode fixes the review selection mode, size and seed rather than choosing them after inspection.

The frozen physical review must also bind the bytes of every declared `Evidence_File` (for example band-decomposed charge density, band/effective-mass evidence as appropriate) under the declared evidence root. A review label without the corresponding hash-bound evidence files is not sufficient for the physical-NFE layer.

If the verified queue is deliberately class-balanced, overall accuracy/F1 describes performance on that designed sample, not natural real-world class prevalence. Always report class support and the queue selection protocol.

## OOD and representation analyses

Chemistry OOD slices and the large-cell representation stress slice are secondary generalization analyses. Do not redefine thresholds after inspecting errors. `N_Atoms` stress is not itself chemical OOD.

Paper mode fixes the large-cell train quantile at `0.95`. Representation/supercell consistency uses the registered probability/score drift tolerances rather than tuning them after seeing the stress-test result.

## Physical interpretation

Pseudo-label benchmark statistics quantify reproduction/generalization of the fixed computational pseudo-label definition. Only the independently reviewed/DFT-verified evidence layer can support claims about a real NFE state. Neither a low p-value nor a narrow bootstrap interval converts a pseudo-label into physical ground truth.
