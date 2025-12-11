# ML Partner Integration – Part 2 (Engine wiring)

This document describes the *engine-side* changes that were made in
Part 2 of the ML partner integration.

## Overview

* `peyton_original/` remains a frozen, read‑only snapshot of the
  upstream ML+cleaning repository.
* `ml_partner_adapters/` (added in Part 1) exposes a stable, non‑interactive
  API for this Django service to consume Peyton's canonical cleaning,
  leakage detection and model configuration logic.

Part 2 wires the existing engine to those adapters:

* `analysis/ml_core/cleaning.py` now imports configuration values
  (`UNKNOWN_THRESHOLD`, `YES_NO_THRESHOLD`, and `UNKNOWN_STRINGS`) from
  `ml_partner_adapters.config_bridge`.  Unknown discovery, severity
  mapping and leakage detection delegate to the corresponding adapter
  modules so that behaviour stays aligned with Peyton's code while
  remaining safe for web/worker execution.

* `analysis/ml_core/models.py` now:
  * uses `ml_partner_adapters.model_configs` for all default
    hyperparameters, and
  * trains the MRF model using the copied
    `MultiLevelRandomForestClassifier` implementation from
    `ml_partner_adapters.mrf_ordinal`.

* Tests under `analysis/tests/` assert that:
  * engine configuration values are imported *via* the adapters and
    match Peyton's `DataCleaning.config`, and
  * model registry defaults are sourced from the adapter model configs.

With these changes, future updates to Peyton's ML repository can be
absorbed by touching only the `ml_partner_adapters/` package, keeping
the Django app's core engine and tests stable.
