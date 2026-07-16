# Data access, license, and disclosure note

## Grant of access

The owner of this project and of the local archive granted explicit, full permission to
inspect, comb through, and use for this study all contents of the local backup at
`%USERPROFILE%\Downloads\BACKUP\OLD_BACKUP` (a personal licensed backup; local username redacted),
including the raw crash extracts and the prior
student work. This study relies on that grant for **inspection and analysis**.

## What that grant does and does not authorize

* **Authorizes:** reading the raw extracts, deriving a de-identified modelling table locally,
  computing and reporting **aggregate** results (metrics, counts, distributions), and shipping
  the *code* and *aggregate artifacts* in the portfolio.
* **Does not, by itself, authorize public redistribution of the raw records.** Alaska
  police-reported crash data contain precise `Latitude`/`Longitude`, `Officer ID`, free-text
  `Street`/`Intersecting Street`, and person-level attributes. These are treated as
  **restricted** regardless of the personal-use grant.

## Disclosure controls enforced by this repository (privacy-by-construction)

1. **No raw records are committed.** The raw workbook stays at its archive path; the derived
   modelling table is written **outside the git tree** (see `.gitignore` and
   `crashsev/build_modeling_table.py`, which writes to a local, ignored `_local_data/` path).
2. **No coordinates, ids, or free-text location** enter the modelling table or any committed
   artifact. `Latitude`, `Longitude`, `Street`, `Intersecting Street`, `Officer ID`,
   `Report ID`, `CDS Number`, `Route` are dropped by the feature-availability ledger
   (IDENTIFIER / privacy tier) and never persisted to a committed file.
3. **Only aggregates are committed.** Committed data artifacts (`data/*.json`, `data/*.csv`,
   `data/*.md`) contain column metadata and class/marginal counts, never per-crash rows.
4. **A reviewer without the raw data** can still read the full method, the contracts, the
   split/target audits, and the aggregate results, and can re-run everything the moment they
   point the pipeline at a licensed copy of the extract (`--data <path>`).

If this project is ever published beyond the owner's personal portfolio, the raw-data terms of
the source agency (Alaska DOT&PF / the crash-records custodian) govern and must be re-checked;
nothing here should be read as a public data-release license.
