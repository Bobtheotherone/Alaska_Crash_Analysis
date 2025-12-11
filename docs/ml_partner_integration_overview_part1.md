# ML partner integration – Part 1 (adapter layer)

This repository already contains a frozen snapshot of Peyton's original
ML + data-cleaning code under the `peyton_original/` package. That snapshot
is treated as read-only and is wired into Git attributes so that it does not
generate noisy diffs.

Part 1 of the integration introduces a small, Django-friendly adapter layer
that sits on top of `peyton_original/`:

* `ml_partner_adapters/config_bridge.py` re-exports the canonical cleaning
  configuration (thresholds and unknown strings) from
  `peyton_original.DataCleaning.config`.
* `ml_partner_adapters/unknown_bridge.py` exposes a non-interactive wrapper
  around Peyton's unknown-placeholder discovery helper.
* `ml_partner_adapters/severity_mapping_bridge.py` wraps Peyton's severity
  mapping utilities, keeping the automatic heuristics but skipping all
  `input()`-based prompts.
* `ml_partner_adapters/leakage_bridge.py` provides non-interactive helpers
  for data-leakage detection and post-hoc importance warnings.
* `ml_partner_adapters/model_configs.py` captures the default
  hyperparameters used by the Decision Tree, EBM, Multilevel Random Forest,
  and XGBoost training scripts as plain Python dictionaries.
* `ml_partner_adapters/mrf_ordinal.py` contains a copy of Peyton's
  `MultiLevelRandomForestClassifier` so the engine can import it from a
  clean, stable module path.

In **Part 2**, the `analysis.ml_core` modules and tests will be updated to
consume this adapter layer instead of reaching into `peyton_original/`
directly. No existing files are modified in Part 1; all changes are
additive and isolated to new modules.
