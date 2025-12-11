# ML Partner Integration (Peyton) – Adapter Layer + Engine Wiring

This document describes the two-phase integration of Peyton's original
ML + data-cleaning code into the `alaska_crash_analysis` Django backend.

- **Phase 1 (Part 1):** Add a *read-only* snapshot of Peyton's repo under
  `peyton_original/` and introduce a non-interactive adapter layer
  `ml_partner_adapters/` that exposes Peyton's config, helpers, and model
  defaults in a Django-safe way. :contentReference[oaicite:0]{index=0}
- **Phase 2 (Part 2):** Wire the existing engine (`analysis.ml_core`) and tests
  to consume those adapters, so the backend behavior stays aligned with Peyton’s
  canonical configuration and model defaults. :contentReference[oaicite:1]{index=1}

`peyton_original/` and `ml_partner_adapters/` are treated as **frozen** inputs
for this merge; all runtime integration now flows through the adapter package.

---

## Phase 1 – Adapter Layer (`ml_partner_adapters/`)

Phase 1 is **purely additive**: no existing files are modified. It introduces a
new top-level package that wraps the Peyton snapshot in non-interactive,
web-safe helpers. :contentReference[oaicite:2]{index=2}

### New files

- `ml_partner_adapters/__init__.py`  
  Package initializer that exposes a stable, non-interactive adapter layer
  on top of the frozen `peyton_original` snapshot.

- `ml_partner_adapters/config_bridge.py`  
  Re-exports Peyton's global cleaning configuration:

  - `UNKNOWN_THRESHOLD`
  - `YES_NO_THRESHOLD`
  - `UNKNOWN_STRINGS`

  from `peyton_original.DataCleaning.config`, so the rest of the codebase has a
  single source of truth for these values.

- `ml_partner_adapters/unknown_bridge.py`  
  Thin, non-interactive wrapper around  
  `peyton_original.DataCleaning.unknown_discovery.discover_unknown_placeholders`,
  suitable for web/worker use. This keeps the original discovery heuristics but
  removes CLI prompts.

- `ml_partner_adapters/severity_mapping_bridge.py`  
  Non-interactive severity-mapping helper that reuses Peyton's numeric and
  text-based heuristics from `severity_mapping_utils` while skipping all
  manual mapping / `input()` steps.

- `ml_partner_adapters/leakage_bridge.py`  
  Adapter for Peyton's leakage utilities that:

  - Combines name-based suggestions and near-perfect-predictor checks into a
    pure function returning a set of leakage columns.
  - Exposes a wrapper for `warn_suspicious_importances` that returns the list of
    suspicious feature names without any interactivity.

- `ml_partner_adapters/model_configs.py`  
  Frozen dictionaries capturing the **default hyperparameters** for Peyton's:

  - Decision Tree
  - EBM
  - Multilevel Random Forest (MRF)
  - XGBoost

  as used in their original training scripts. These dicts are the canonical
  source of truth for model defaults.

- `ml_partner_adapters/mrf_ordinal.py`  
  Standalone adapter module containing a copy of Peyton's
  `MultiLevelRandomForestClassifier` implementation, with minimal import
  adjustments so it can be imported directly from Django code.

- `docs/ml_partner_integration_overview_part1.md`  
  Short design note explaining:

  - `peyton_original/` as a frozen snapshot.
  - `ml_partner_adapters/` as a non-interactive bridge to Peyton's config +
    helpers.
  - How Phase 2 will wire `analysis.ml_core` and tests to use this adapter
    layer.

---

## Phase 2 – Engine + Tests Wiring

Phase 2 changes **only** the engine (`analysis/ml_core`) and tests
(`analysis/tests`), plus an overview doc. `peyton_original/**` and
`ml_partner_adapters/**` remain read-only. :contentReference[oaicite:3]{index=3}

### Modified engine files

#### `analysis/ml_core/cleaning.py`

- Imports canonical config values from the adapter:

  ```python
  from ml_partner_adapters.config_bridge import (
      UNKNOWN_THRESHOLD,
      YES_NO_THRESHOLD,
      UNKNOWN_STRINGS as PEYTON_UNKNOWN_STRINGS,
  )
