# Formal evaluation workflow

For **final paper artifacts**, use the canonical dispatcher documented in `docs/FINAL_PAPER_WORKFLOW.md`:

```bash
python -m training.paper <alias> [arguments...]
```

Do not use the individual evaluator modules below as substitutes for the paper-ready path. They remain available for debugging/development, but they do not collectively replace the immutable budget, clean-Git, v2.4 pair-symmetric graph, closed-set summary and artifact gates enforced by `training.paper`.

## Paper-ready evaluation aliases

Run from one clean final Git commit:

```bash
python -m training.paper cache-rebuild-audit
python -m training.paper cache-sanity-audit
python -m training.paper split-duplicate-audit
python -m training.paper neighbor-symmetry-audit
python -m training.paper generator-contract-audit
```

Current formal semantics are:

- cache schema `nfe-mxene-cache-2.4`;
- global descriptor schema `intrinsic-slab-v3`;
- neighbor policy `radius-shell-complete-pair-symmetric-v3`;
- zero cache skips for paper-ready data;
- exact data/structure/target/cache/normalizer/split/code provenance.

After each formal run, bind both prediction CSVs to the adjacent result identity:

```bash
python -m training.paper sign-predictions \
  --predictions /path/to/test_predictions.csv

python -m training.paper sign-predictions \
  --predictions /path/to/validation_predictions.csv
```

The formal signer recomputes the relevant metrics before writing the SHA256 content-addressed manifest. Editing, substituting or pairing the CSV with the wrong result invalidates the workflow-integrity checks.

For the prediction-blind verified-NFE protocol, use the `verified-queue`, `blind-verified`, `freeze-verified` and `verified-evaluate` aliases exactly as described in `docs/FINAL_PAPER_WORKFLOW.md`.

OOD analysis:

```bash
python -m training.paper ood-evaluate \
  --predictions /path/to/test_predictions.csv \
  --manifest /path/to/ood_manifest.csv
```

Paper model-vs-model inference uses the strict five-seed nested seed × `Split_Group` bootstrap:

```bash
python -m training.paper paired-bootstrap \
  --a A_seed2027/test_predictions.csv A_seed2028/test_predictions.csv A_seed2029/test_predictions.csv A_seed2030/test_predictions.csv A_seed2031/test_predictions.csv \
  --b B_seed2027/test_predictions.csv B_seed2028/test_predictions.csv B_seed2029/test_predictions.csv B_seed2030/test_predictions.csv B_seed2031/test_predictions.csv \
  --name-a A --name-b B
```

Representation consistency:

```bash
python -m training.paper representation-audit \
  --checkpoint /path/to/best.pt structure1.vasp structure2.vasp
```

Final artifact reconciliation:

```bash
python -m training.paper paper-preflight \
  /path/to/run1/result.json \
  /path/to/run2/result.json \
  /path/to/run3/final_metrics.json
```

Final summaries must use:

```bash
python -m training.paper baseline-summary
python -m training.paper ablation-summary
```

These summary aliases require the complete preregistered model/ablation sets and therefore cannot silently omit a failed or unfavorable baseline.

## Development-only lower-level entrypoints

`training.formal_v2_4` and `training.evaluation.*` modules may be used for smoke tests, debugging and exploratory analysis. If their budget, cache-skip rule, runtime mode or result set differs from the paper-ready contract, their outputs are not eligible for final paper tables.
