# Data provenance and relationship to prior work

Establishes where the modelling data comes from, how it relates to the original capstone's
data, and why this study models the **raw 2009–2012 extract** rather than the prior group's
cleaned CSVs. Based on a read-only audit of the licensed archive
(a personal licensed backup, `%USERPROFILE%\Downloads\BACKUP\OLD_BACKUP`; local username redacted);
file/line citations are in the audit trail
`research_audit/PRIOR_WORK_LINEAGE.md`.

## Three generations of prior code (kept distinct)

| Generation | What it is | Temporal columns |
|---|---|---|
| **S23** (Spring 2023) | Combine + clean notebooks that read the two raw xlsx and write a combined/cleaned pickle | retains Year/Month/Time (per V1↔V2 field map) |
| **F24** (Fall 2024) | The capstone's XGBoost/DT/RF/SVM/multilevel models + the Django app; reads two cleaned CSVs | **Year dropped** in the cleaned CSVs |
| **2025** | The remediation-era scripts (poster viz, a profiling cleaner) | n/a |

## The modelling source and why

* **Used:** `Crash Level 09-12 (1).xlsx` — the raw 2009–2012 crash-level extract, **physically
  present** in the archive (`OLD_BACKUP\2025\`; 25,391,525 bytes;
  **SHA-256 withheld from public release** (retained in the restricted reproduction log; custodian
  permission to publish this source identifier has not been confirmed under the NDA/data-use
  agreement), recorded
  2026-07-11), 50,543 crashes, 100 columns, with `Year`, `Date`, `DateTime`, and unique
  `Crash Number`. This is the only present source with an intact temporal axis and full column
  provenance, so it is the study's data. Rebuilding the modelling table from this byte-hashed file
  reproduces the frozen study input byte-identically and the governed content hash `059559cd…`
  exactly (`DATA_AUTHORITY_AND_ACCESS.md` §2).
* **Not used for modelling:** `cleaned_test_data.csv` (58,454 rows) and
  `new_test_data_oct_7.csv` (48,964 rows). Both match the paper's class balance, but:
  1. **Their producing scripts are not preserved** — they were Colab notebooks reading/writing
     `/content/drive/MyDrive/crash_data/`. Every reference to them in the archive is a
     *read*, never a *write*. Their exact cleaning (including how/why `Year` was dropped) is
     therefore **inference**, not a recovered artifact — unacceptable provenance for the modelling table.
  2. **Both lack any `Year`/`Date`/`DateTime` column**, which precludes the charter-preferred
     chronological final test.
* **`Crash Level 13-17 (1).xlsx`** (the 2013–2017 "V2" extract) is **referenced in code but not
  present** in the archive (only a stale LibreOffice lock file proves it once existed). A
  combined 2009–2017 dataset survives only as a *derived* pickle
  (`Cleaned Combined Crash Data 38 Fields.pkl`, S23 cleaning of unknown full provenance, a
  different 38-field Title-Case schema). Extending the study to 2013–2017 is documented as
  future work (`research/ESTIMAND_AND_SCOPE.md` §Why the 2009–2012 window); it is **not**
  fabricated from a derived pickle.

## Continuity with the original target definition (corroboration)

The prior F24 3-class collapse is **identical** to this study's:

```
# prior F24 (xgboost_three_classification):        # this study (KABCO):
No Apparent Injury           -> 0                   O (blank, verified PDO)      -> 0
Suspected Minor, Possible    -> 1                   B (Non-Incapacitating), C (Possible) -> 1
Suspected Serious, Fatal     -> 2                   A (Incapacitating), K (Fatal)        -> 2
Unknown, Died Prior to Crash -> dropped             Unknown, Not Reported, Null value    -> quarantined
```

The prior data used MMUCC text labels with an explicit "No Apparent Injury" at ~64%; this
extract uses old-style ABC labels with **blank = property-damage-only** at ~63% — the same
class-0 phenomenon under two label vintages, reaching the same ~0.68/0.28/0.04 balance. This
is independent corroboration of the evidence-checked blank=O decision, not a coincidence to hide.

## The evaluation defects this study corrects (from the prior code)

The prior modelling — the motivation for this remediation — exhibited exactly the defects the
leakage-controlled protocol and the leakage factorial target:

* **Random, non-temporal splits**, with **different test fractions per model** (XGBoost 0.20;
  Decision Tree / Random Forest / SVM 0.30; multilevel 5-fold with no held-out test; the web
  app a user-set default of 0.20). Different models were thus scored on different rows, and no
  split respected time.
* **`random_state=42` everywhere**, with no repeated-seed variability reported.
* **No feature-availability control** was documented for the CSV feature sets (e.g.
  `serious_with_fatalities`, `fatalities`, `serious`, `minor` appear as columns in
  `new_test_data_oct_7.csv` — outcome-derived fields that must be excluded).

## Sentinel handling — prior art validating this study's rules

The S23 clean neutralised **AADT `< 0 → None`** (the mechanism that catches the int32-min
`-2147483648` sentinel), and folded string placeholders (`Null value`, `Missing`) into
"Unknown" buckets; the Django app mapped a regex of placeholders plus NaN to `Unknown`. This
study's schema-driven sentinel rules (`data/schema.json`) generalise that handling and add the
Temperature (`-460`/`999`) and Distance-from-intersection sentinels the prior work did not
address. No prior Milepoint cleaning existed; this study drops Milepoint (identifier/privacy).

## Provenance caveats (stated, not hidden)

1. Exact producing scripts for the two cleaned CSVs are unrecovered; statements about their
   column drops are inference from source-schema docs + the CSV headers.
2. Two author identities appear in the archive (an **S23 contributor**, from Spring-2023 Colab
   paths; **`rnmercado`** — the applicant, from 2025 LibreOffice locks); the raw 09-12 xlsx
   predates both remediation touches. (The third-party S23 handle and local OS usernames are
   redacted here; the lineage facts are unchanged.)
