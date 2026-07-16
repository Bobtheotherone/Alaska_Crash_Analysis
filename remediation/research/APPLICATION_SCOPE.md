# Application scope: canonical research layer vs. legacy demo (AA2-018)

To remove the false impression that the original Django web application is the study's
execution layer, this remediation draws an explicit boundary.

## The canonical research execution layer is `crashsev`

Every scientific claim in the paper is produced by the `crashsev` package through the governed
CLI (`validate-data → develop → freeze-experiment → evaluate-final`). The contracts, target
mapping, split, models, metrics, calibration, uncertainty, and immutable run bundle are all
here. **Reproduction uses only `crashsev`** (plus a licensed copy of the raw extract). Nothing
in the paper depends on the web application.

## The Django web application (`akCrashData` / `capstone`) is a legacy demonstration

The original project shipped a Django app that let a user upload a CSV, pick a target, and train
one of several models in-browser. It is retained in the archive as a **legacy demonstration of a
user interface**, not as the research pipeline. It is explicitly **out of scope** for the
scientific claims because:

* it fits preprocessing with `pd.get_dummies` / median-fill on the full uploaded frame before a
  random `train_test_split` (the leakage this study corrects);
* it label-encodes the chosen target generically (no fail-closed KABCO contract, no quarantine);
* it uses a user-set random split (default 0.2) with `random_state=42`, not a temporal,
  crash-grouped, held-out (out-of-time) final test.

## Security fixes: specified, not claimed as executed

The first-stage review specified fixes for the app's multi-user isolation, upload gating, and
transaction handling (SEC-003 / SW-001 / SW-007). Those patches are **documented but not
executed here**, because verifying them requires a live PostGIS database and Django test
harness that this environment does not provide. This ledger does **not** claim the app is fixed
or production-safe; it claims only that the *research* is performed by `crashsev`, which the app
does not touch. Presenting the app as "the corrected pipeline" would be a false-completion claim
and is avoided.

## Consequence for the reader

If you want to *reproduce the science*, ignore the web app entirely and follow
`research/REPRODUCE.md`. If you want to *see the original UI*, the legacy app is in the archive,
clearly labelled as such and not part of the evaluated result.
