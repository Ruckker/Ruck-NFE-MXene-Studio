# Formal evaluation workflow

Use this sequence for paper-ready benchmark artifacts. The lower-level evaluator scripts remain useful for debugging, but formal statistics should pass through the signed/provenance-aware steps below.

## 1. Audit the fixed split for duplicate leakage

```bash
python -m training.evaluation.audit_split_duplicates \
  --config training/configs/nfe_predictor.yaml
```

This hard-fails if identical source-file bytes or exact model-input tensors occur across train/validation/test. Coarser representation-invariant signatures are also reported for manual near-duplicate review.

## 2. Train/evaluate from one clean final Git commit

Run all formal seeds/models from the same clean project commit. Formal provenance records the dataset table, referenced structure bytes, exact cached tensors, target contract, train normalizers, split manifest, graph semantics and training/runtime code identity.

Development runs may tolerate a configured cache skip fraction, but paper-ready results should have **zero skipped cache rows**.

## 3. Sign each prediction CSV immediately after the run

For every result directory:

```bash
python -m training.evaluation.sign_predictions \
  --predictions /path/to/run/test_predictions.csv

python -m training.evaluation.sign_predictions \
  --predictions /path/to/run/validation_predictions.csv
```

The signer auto-detects the sibling `result.json` or `final_metrics.json` and writes:

- `test_predictions.manifest.json`
- `validation_predictions.manifest.json`

The manifest hashes the CSV bytes and binds them to dataset/structure/target/cache/normalizer/split/Git/run identity. Editing or substituting the CSV invalidates the manifest.

## 4. Use signed formal evaluation entrypoints

Verified NFE:

```bash
python -m training.evaluation.formal_verified_nfe \
  --predictions /path/to/test_predictions.csv \
  --verified training/evaluation/verified_nfe_template.csv
```

OOD slices:

```bash
python -m training.evaluation.formal_evaluate_slices \
  --predictions /path/to/test_predictions.csv \
  --manifest training/evaluation/ood_manifest.csv
```

Paired Split_Group bootstrap:

```bash
python -m training.evaluation.formal_paired_bootstrap \
  --a /path/to/model_a/test_predictions.csv \
  --b /path/to/model_b/test_predictions.csv \
  --name-a A --name-b B
```

The paired formal wrapper additionally requires the two signed prediction files to share the same benchmark data identity.

## 5. Run representation consistency on full checkpoints

```bash
python -m training.evaluation.supercell_consistency \
  --checkpoint /path/to/best.pt \
  structure1.vasp structure2.vasp
```

This uses production checkpoint/graph guards and tests exact in-plane supercells, atom reordering, an equivalent unimodular in-plane basis and added vacuum for the same Cartesian slab. Drift beyond the configured threshold exits nonzero.

## 6. Final paper-ready gate

After signing predictions:

```bash
python -m training.evaluation.paper_preflight \
  /path/to/run1/result.json \
  /path/to/run2/result.json \
  /path/to/run3/final_metrics.json
```

Default requirements include:
- current v2.3 target/data/global/graph semantics;
- exact cache-tensor and train-normalizer identity;
- zero skipped cache records;
- clean current Git commit equal to the artifact commit;
- valid, untampered validation/test prediction manifests;
- one common dataset/cache/normalizer/split identity across the supplied result set.

`--allow-cache-skips` exists only for explicitly exploratory analysis; do not use it for the final paper table without reporting and justifying every skipped row.

## 7. Verified-NFE review independence

Freeze/version the manual/DFT verification table **before** exposing reviewers to model predictions whenever possible. If reviewers saw model predictions while assigning verified labels, disclose that the verified subset is not fully blinded and do not describe it as an independent blinded validation set.
