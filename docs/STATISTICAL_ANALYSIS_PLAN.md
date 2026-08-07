# Statistical analysis plan for the NFE benchmark

This plan should be frozen before the final five-seed paper campaign so metric/reporting choices are not selected after seeing test results.

## Fixed data and runs

- Use the audited fixed train/validation/test split and one final clean Git commit.
- Use the same five predeclared training seeds for every stochastic model in a formal comparison.
- Never choose a checkpoint, hyperparameter or model variant from test performance.
- Paper-ready results require zero skipped cache rows unless every skipped row is explicitly disclosed and justified.

## Primary endpoints

Use a small set of prespecified endpoints for headline comparisons:

1. **Macro average precision** — primary imbalanced three-class discrimination/ranking endpoint.
2. **Macro F1** — primary thresholded three-class classification endpoint.
3. **High-class enrichment at 5%** — primary screening-use endpoint.
4. **NFE pseudo-score MAE** — primary continuous pseudo-score endpoint for models trained on the score.

These endpoints answer different operational questions but are not independent physical ground truths. In particular NFE class and NFE pseudo-score are mathematically coupled pseudo-targets.

## Secondary/descriptive endpoints

Report as secondary or diagnostic metrics:
- balanced accuracy;
- macro ROC-AUC;
- per-class F1/recall/AP;
- Precision/Recall/Enrichment at 1%, 5%, 10%;
- NFE score RMSE, Spearman and R²;
- calibrated test ECE;
- parameter count, training time and runtime/environment metadata.

Do not promote a secondary metric to the headline endpoint because it happens to give the most favorable result.

## Seed variability

For each model, report mean ± sample standard deviation across the five independent training seeds. This describes training-run variability; it is not a confidence interval over the test population.

Do not infer statistical significance from whether two mean±SD intervals overlap.

## Paired model comparisons

For a formal A-vs-B claim, use matched seed identities and the same signed test samples.

Preferred inference:
- `formal_multiseed_bootstrap.py`: nested resampling of training seed and `Split_Group` chemistry blocks;
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

## Verified-NFE subset

Primary verified analysis should use **all review-complete cases**. Reviewer-confidence thresholds such as 0.6/0.8/0.9 are sensitivity analyses, not alternative primary datasets chosen after seeing performance.

Generate the verified review queue without reading model predictions, supply reviewers a blinded sheet that omits pseudo-label/model prediction columns, freeze the completed sheet, and only then join predictions.

If the verified queue is deliberately class-balanced, overall accuracy/F1 describes performance on that designed sample, not natural real-world class prevalence. Always report class support and the queue selection protocol.

## OOD analyses

Chemistry OOD slices and the large-cell representation stress slice are secondary generalization analyses. Do not redefine thresholds after inspecting errors. `N_Atoms` stress is not itself chemical OOD.

## Physical interpretation

Pseudo-label benchmark statistics quantify reproduction/generalization of the fixed computational pseudo-label definition. Only the independently reviewed/DFT-verified evidence layer can support claims about a real NFE state. Neither a low p-value nor a narrow bootstrap interval converts a pseudo-label into physical ground truth.
