# `data/` — data contracts, codebook evidence check, and re-analysis inputs

This directory holds **contracts and aggregates**, not raw crash data. The raw Alaska extract is used
locally under an owner grant and is never placed here; the de-identified modelling table is written to
a git-ignored `_local_data/` path (`research/DATA_LICENSE_NOTE.md`).

| File | What it is |
|---|---|
| `DATA_MANIFEST.md` | Gate-0 manifest: provenance, unit/keys, evidence-checked target, real row-flow, features, privacy. |
| `schema.json` | Machine-readable schema: required columns, target domain, numeric sentinels (built by `data_audit/build_schema_contract.py`). |
| `target_mapping.yml` | The fail-closed KABCO→ordinal contract, incl. evidence-checked `blank_maps_to: O` (consumed by `crashsev/target.py`). |
| `codebook_verification_09_12.md`, `codebook_evidence_09_12.json` | Evidence-checked target mapping — an internal check (NOT an official source-agency codebook validation) — that blank severity = property-damage-only (0.0% injury/fatality contamination). |
| `feature_availability_ledger.csv` | All 100 columns classified pre-event / at-event / temporal / post-outcome / identifier; 66 allowed. |
| `modeling_table_audit.md` | Aggregate audit of the de-identified modelling table (shape, columns, per-column summaries) — no per-crash rows. |
| `confusion_matrices_from_paper.json` | The four confusion matrices transcribed from the prior report's screenshots. **These are the ORIGINAL, contaminated-protocol results**, input to the honest re-analysis. |

To reproduce the empirical run with a licensed copy of the extract, follow **`research/REPRODUCE.md`**:
```bash
python -m crashsev.cli validate-data    --data "<path>/Crash Level 09-12 (1).xlsx"
python -m crashsev.build_modeling_table  --data "<path>/Crash Level 09-12 (1).xlsx"
python -m crashsev.cli develop --data _local_data/modeling_table_09_12.csv --config configs/route_a_09_12.yml
```
