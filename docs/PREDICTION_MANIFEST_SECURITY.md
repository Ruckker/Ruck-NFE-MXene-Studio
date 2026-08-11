# Prediction manifest integrity model

The `sign_predictions*` command names are historical shorthand. The generated `*.manifest.json` files are **SHA256 content-addressed integrity manifests, not public-key digital signatures**.

They are designed to catch research-workflow mistakes such as:
- editing a prediction CSV after metrics were computed;
- copying the wrong model's CSV into another run directory;
- pairing predictions from different dataset/cache/normalizer/split identities;
- replacing a result file after a formal prediction manifest was produced;
- mixing artifacts from different Git revisions or training protocols.

`sign_predictions_formal.py` additionally recomputes core metrics from the CSV before binding it to the sibling run result. `paper_preflight_strict.py` later rechecks CSV bytes, manifest hashes, run identity, result metrics and current formal provenance.

This is an **integrity and consistency** mechanism for an accidental-error/reproducibility threat model. It is not an authenticity mechanism against an adversary who can deliberately rewrite the CSV, result and manifest together and recompute every unkeyed hash.

If adversarial provenance/authenticity is ever required, add a real signing layer (for example signed Git tags/releases plus a detached public-key signature over the final artifact manifest). Do not describe the current SHA256 manifests as cryptographic digital signatures in a paper or software claim.
