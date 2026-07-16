# Request for source-owner confirmation of crash-severity field semantics (Gate 3)

**From:** Radames Naythan Mercado-Barbosa, University of Alaska Anchorage (graduate research portfolio, Alaska Car Crash Analysis, Iteration IV)
**To:** Alaska DOT&PF crash-records custodian (or delegated data steward)
**Regarding:** licensed extract `Crash Level 09-12 (1).xlsx` (crash-level, 50,543 rows, 2009–2012)
**Date prepared:** 2026-07-15 (manuscript completion is NOT blocked on this request; the study
treats the items below as open construct evidence and says so explicitly)

## What we ask you to confirm (or correct)

1. **Blank `Crash Severity`.** In the 2009–2012 crash-level extract, the `Crash Severity` field
   is blank for 32,046 of 50,543 rows; every blank-severity row carries explicit zeros (not
   blanks) in `Number of Fatalities`, `Number of Injuries with Fatalities`, and `Number of
   Injuries without Fatailites` (sic). Does a blank in `Crash Severity` officially denote a
   no-apparent-injury / property-damage-only crash in this extract's vintage, or can it also
   mean "not recorded"? Is there a written data dictionary or ETL rule that governs this?
2. **`Unknown` / `Not Reported` / `Null value` tokens** (3,699 rows). What distinguishes these
   from a blank at data entry? (Our study quarantines them and maps blanks to PDO; both
   populations show all-zero injury counts, so the counts alone cannot discriminate.)
3. **Injury-count field derivation.** Are the three count fields entered independently, or
   derived from person-level records (or from `Crash Severity` itself)? We observe
   `Number of Injuries with Fatalities` positive for 1,511 of 1,512 `Incapacitating` crashes
   that have zero fatalities, and column-identical to `Number of Injuries without Fatailites`
   on most rows — which suggests a mislabelled or duplicated export column.
4. **Field recording times.** For the 49 report fields listed in the enclosed ledger
   (`paper/FEATURE_TIMING_EVIDENCE.md`): who records each value, at which workflow stage
   (scene / initial entry / supervisor review / later amendment), and which are revised after
   the injury outcome is known?
5. **Agency-keyed re-supply for dependence analysis.** The licensed extract contained
   `Officer Agency`, `Reporting Agency`, and `Detachment` columns; we removed them, with all
   identifier-tier fields, during de-identification, and the raw file is no longer retained in
   the analysis environment under our data-handling policy. Could a crash-number-keyed agency
   identifier (a pseudonymous code is sufficient — no agency names needed) be licensed for
   research, so a paired agency-cluster dependence analysis can be run against the frozen
   evaluation rows?

## Documents that would fully answer this request

- The **Alaska Motor Vehicle Collision Report Form 12-200 Instruction Manual (Revised)** — a
  copy is catalogued by NHTSA ("Alaska Motor Vehicle Collision Report Form 12-200 –
  Instruction Manual, Revised", nhtsa.gov document catalogue; automated retrieval was blocked,
  so an official copy is requested directly).
- The DOT&PF crash-database **data dictionary / ETL specification** for the crash-level extract
  vintage used here (2009–2012).
- Any **FHWA/NHTSA state injury-code conversion sheet** for Alaska covering these years.

## Why it matters

The blank→PDO mapping constitutes **all** of class 0 in this study (32,046 of 46,844 usable
rows). The manuscript therefore labels its target *researcher-defined recorded severity*,
reports the internal evidence for and against the mapping, and defers the official semantics to
this request (roadmap Phase 1, item 1). A one-paragraph custodian confirmation would resolve
the study's single largest construct uncertainty.

*Prepared as part of the 2026-07 portfolio revision; see `experiment/target_reporting_process_audit.json` for the machine evidence behind every count quoted above.*
