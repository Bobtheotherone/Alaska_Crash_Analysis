"""
crashsev.schema — data contract for the Alaska police-reported crash dataset.

Everything in this module is *recovered from the project's own source code*, not from
the raw data (which is absent; see DATA-001). Specifically:

* The set of columns is taken from ``peyton_original/DataCleaning/Column Analysis.txt``,
  which enumerates the columns of the 2009-2012 and 2013-2017 Alaska crash extracts.
* The target column name ("Crash Severity") is the common severity column in that file.
* The KABCO code set {K, A, B, C, O} is taken from ``crashdata/importer.py``
  (``_coerce_severity`` normalises to exactly this set).

Because the codebook itself is not supplied, the *semantic* KABCO ordering below is the
standard national KABCO convention (see references in the paper). It is marked as an
ASSUMPTION that must be confirmed against the source agency codebook before any empirical
claim (DATA-002 acceptance criterion).

The feature-availability ledger classifies every known source variable by *when the
information becomes available relative to the crash outcome*. This is the core artifact
that prevents outcome-derived leakage (METH-001): variables that are only known *because*
of the crash outcome (injury and fatality counts) must never be used to predict severity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# Target: KABCO -> 3-level ordinal severity
# ---------------------------------------------------------------------------

TARGET_COLUMN = "Crash Severity"

# The 5 KABCO codes the importer recognises (crashdata/importer.py::_coerce_severity).
KABCO_CODES: List[str] = ["O", "C", "B", "A", "K"]

# Standard national KABCO ordering, most-severe last. ASSUMPTION pending source codebook.
#   K = Fatal injury
#   A = Suspected serious / incapacitating injury
#   B = Suspected minor / non-incapacitating injury
#   C = Possible injury
#   O = No apparent injury / property damage only (PDO)
KABCO_SEVERITY_RANK: Dict[str, int] = {"O": 0, "C": 1, "B": 2, "A": 3, "K": 4}

# The 3-level ordinal collapse used by the study (matches the text keyword collapse in
# peyton_original/severity_mapping_utils.py::map_text_severity):
#   class 0 = None / PDO                     {O}
#   class 1 = Minor / possible injury        {C, B}
#   class 2 = Serious or fatal injury        {A, K}
# This collapse is DOCUMENTED and REVERSIBLE (raw KABCO retained separately). It is a
# defensible clinical/operational grouping but is an ASSUMPTION until the codebook and a
# stakeholder confirm it (DATA-002).
KABCO_TO_ORDINAL: Dict[str, int] = {
    "O": 0,
    "C": 1,
    "B": 1,
    "A": 2,
    "K": 2,
}

ORDINAL_CLASS_LABELS: Dict[int, str] = {
    0: "none/PDO (O)",
    1: "minor/possible injury (B,C)",
    2: "serious/fatal injury (A,K)",
}

N_CLASSES = 3


# ---------------------------------------------------------------------------
# Feature-availability ledger (METH-001 / RQ-001)
# ---------------------------------------------------------------------------


class Availability(str, Enum):
    """When is a variable's value determined, relative to the crash outcome?"""

    PRE_EVENT = "pre_event"        # roadway / traffic context, known before the crash
    AT_EVENT = "at_event"          # circumstances of the crash itself (weather, collision type)
    TEMPORAL = "temporal"          # calendar/time features (note: Year is a source-version proxy)
    POST_OUTCOME = "post_outcome"  # known only *because of* the outcome -> PROHIBITED for prediction
    IDENTIFIER = "identifier"      # keys / ids -> dropped (not predictive, grouping only)
    TARGET = "target"              # the outcome itself
    UNKNOWN = "unknown"            # timing not established -> must be resolved before use


# Policy: which availability tiers may enter the model, keyed by use case.
USE_CASE_ALLOWED: Dict[str, Set[Availability]] = {
    # Post-crash severity classification / data completion (the most defensible use case
    # given the available features): everything except outcome-derived, identifiers, target.
    "post_crash_triage": {
        Availability.PRE_EVENT,
        Availability.AT_EVENT,
        Availability.TEMPORAL,
    },
    # Pre-crash / prospective risk (stricter): only information available before the crash.
    "pre_crash_prevention": {
        Availability.PRE_EVENT,
        Availability.TEMPORAL,
    },
}

# NEVER allowed under any prospective/predictive use case.
PROHIBITED_TIERS: Set[Availability] = {
    Availability.POST_OUTCOME,
    Availability.IDENTIFIER,
    Availability.TARGET,
    Availability.UNKNOWN,
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    availability: Availability
    note: str = ""


# Classification of the known Alaska crash variables. Columns are drawn from
# peyton_original/DataCleaning/Column Analysis.txt (common + per-extract columns).
# POST_OUTCOME entries are the leakage guardrail: they are *mechanically determined by the
# outcome* and are excluded from every predictive model.
FEATURE_LEDGER: List[FeatureSpec] = [
    # --- Target ---
    FeatureSpec(TARGET_COLUMN, Availability.TARGET, "3-level ordinal outcome"),
    # --- Outcome-derived (PROHIBITED) ---
    FeatureSpec("Number of Fatalities", Availability.POST_OUTCOME, "> 0 iff K; direct leak"),
    FeatureSpec("Number of Serious Injuries", Availability.POST_OUTCOME, "defines A; direct leak"),
    FeatureSpec("Number of Serious Injuries with Fatalities", Availability.POST_OUTCOME, "leak"),
    FeatureSpec("Number of Minor Injuries", Availability.POST_OUTCOME, "defines B/C; leak"),
    FeatureSpec("Number of Injuries with Fatalities", Availability.POST_OUTCOME, "leak"),
    FeatureSpec("Number of Injuries without Fatailites", Availability.POST_OUTCOME, "leak (sic)"),
    FeatureSpec("Number of Injuries without Fatalities", Availability.POST_OUTCOME, "leak"),
    FeatureSpec("Causal Unit Driver Injury", Availability.POST_OUTCOME, "person injury outcome"),
    FeatureSpec("Unit 1 Person 1 Injury", Availability.POST_OUTCOME, "person injury outcome"),
    FeatureSpec("Unit 1 Person 1 Transported", Availability.POST_OUTCOME, "EMS disposition"),
    FeatureSpec("Unit 1 Person 1 Transported To", Availability.POST_OUTCOME, "EMS disposition"),
    FeatureSpec("Unit 1 Person 1 Transported By", Availability.POST_OUTCOME, "EMS disposition"),
    FeatureSpec("Unit 1 Person 1 Extricated", Availability.POST_OUTCOME, "post-crash response"),
    FeatureSpec("Unit 1 Person 1 Ejected", Availability.POST_OUTCOME, "outcome-correlated"),
    FeatureSpec("Unit 1 Person 1 NFR", Availability.POST_OUTCOME, "post-crash"),
    FeatureSpec("Causal Unit Extent of Damage", Availability.POST_OUTCOME, "damage assessment"),
    FeatureSpec("Unit 1 Damage", Availability.POST_OUTCOME, "damage assessment"),
    FeatureSpec("Unit 1 Undercarriage Damage", Availability.POST_OUTCOME, "damage assessment"),
    FeatureSpec("Unit 1 Primary Damage Location", Availability.POST_OUTCOME, "damage assessment"),
    FeatureSpec("Non Vehicle Damage", Availability.POST_OUTCOME, "damage assessment"),
    FeatureSpec("Arrest", Availability.POST_OUTCOME, "post-crash enforcement"),
    FeatureSpec("Causal Unit Driver Charges", Availability.POST_OUTCOME, "post-crash enforcement"),
    FeatureSpec("Causal Unit Driver Violations", Availability.POST_OUTCOME, "post-crash enforcement"),
    FeatureSpec("Unit 1 Person 1 Violations", Availability.POST_OUTCOME, "post-crash enforcement"),
    # --- Identifiers (dropped; used only for grouping) ---
    FeatureSpec("Crash Number", Availability.IDENTIFIER, "crash grouping key"),
    FeatureSpec("CDS Number", Availability.IDENTIFIER, "crash id"),
    FeatureSpec("Report ID", Availability.IDENTIFIER, "report id"),
    FeatureSpec("Officer ID", Availability.IDENTIFIER, "reporter id"),
    FeatureSpec("Officer Agency", Availability.IDENTIFIER, "reporter agency"),
    # --- Temporal (Year flagged as source-version proxy; ablate) ---
    FeatureSpec("Year", Availability.TEMPORAL, "source-version proxy; ablate (A4)"),
    FeatureSpec("Month", Availability.TEMPORAL, ""),
    FeatureSpec("Day of the Week", Availability.TEMPORAL, ""),
    FeatureSpec("Day of Month", Availability.TEMPORAL, "day-of-month has no plausible mechanism"),
    FeatureSpec("Time of Day", Availability.TEMPORAL, ""),
    FeatureSpec("Week of the Year", Availability.TEMPORAL, ""),
    # --- Pre-event roadway / traffic context ---
    FeatureSpec("AADT", Availability.PRE_EVENT, "annual average daily traffic"),
    FeatureSpec("Functional Class", Availability.PRE_EVENT, ""),
    FeatureSpec("NHS System", Availability.PRE_EVENT, ""),
    FeatureSpec("AHS System", Availability.PRE_EVENT, ""),
    FeatureSpec("Route", Availability.PRE_EVENT, ""),
    FeatureSpec("Milepoint", Availability.PRE_EVENT, ""),
    FeatureSpec("Posted Speed", Availability.PRE_EVENT, ""),
    FeatureSpec("At Intersection", Availability.PRE_EVENT, ""),
    FeatureSpec("Junction", Availability.PRE_EVENT, ""),
    FeatureSpec("Roadway Junction", Availability.PRE_EVENT, ""),
    FeatureSpec("Rural Urban", Availability.PRE_EVENT, ""),
    FeatureSpec("Urban-Rural", Availability.PRE_EVENT, ""),
    FeatureSpec("Region", Availability.PRE_EVENT, ""),
    FeatureSpec("Census Area", Availability.PRE_EVENT, ""),
    FeatureSpec("City", Availability.PRE_EVENT, ""),
    FeatureSpec("Borough", Availability.PRE_EVENT, ""),
    FeatureSpec("County-Borough", Availability.PRE_EVENT, ""),
    FeatureSpec("Latitude", Availability.PRE_EVENT, "location; privacy-sensitive (DATA-007)"),
    FeatureSpec("Longitude", Availability.PRE_EVENT, "location; privacy-sensitive (DATA-007)"),
    FeatureSpec("Maintenance Category", Availability.PRE_EVENT, ""),
    FeatureSpec("Maintenance Responsibility", Availability.PRE_EVENT, ""),
    # (AUTH-001) "Functional Class" is declared once above at its first occurrence; the duplicate
    # entry that used to sit here was removed.
    # --- At-event crash circumstances ---
    FeatureSpec("Weather", Availability.AT_EVENT, ""),
    FeatureSpec("Road Surface", Availability.AT_EVENT, ""),
    FeatureSpec("Pavement", Availability.AT_EVENT, ""),
    FeatureSpec("Lighting", Availability.AT_EVENT, ""),
    FeatureSpec("Manner of Collision", Availability.AT_EVENT, ""),
    FeatureSpec("Crash Type", Availability.AT_EVENT, ""),
    FeatureSpec("First Harmful Event", Availability.AT_EVENT, ""),
    FeatureSpec("Causal Unit 1st Event", Availability.AT_EVENT, ""),
    FeatureSpec("Causal Unit Most Harmful Event", Availability.AT_EVENT, ""),
    FeatureSpec("Number of Motorized Units", Availability.AT_EVENT, ""),
    FeatureSpec("Number of Units Involved", Availability.AT_EVENT, ""),
    FeatureSpec("Alcohol Suspected", Availability.AT_EVENT, "suspected at scene (not adjudicated)"),
    FeatureSpec("Drugs Suspected", Availability.AT_EVENT, "suspected at scene"),
    FeatureSpec("Causal Unit Driver Alcohol Suspected", Availability.AT_EVENT, ""),
    FeatureSpec("Causal Unit Action", Availability.AT_EVENT, ""),
    FeatureSpec("Causal Unit Direction of Travel", Availability.AT_EVENT, ""),
    FeatureSpec("Causal Unit Body Type", Availability.AT_EVENT, ""),
    FeatureSpec("Environmental Conditions 1", Availability.AT_EVENT, ""),
    FeatureSpec("Environmental Conditions 2", Availability.AT_EVENT, ""),
]


def ledger_by_name() -> Dict[str, FeatureSpec]:
    """Map lower-cased column name -> FeatureSpec (last spec wins on duplicates)."""
    return {spec.name.strip().lower(): spec for spec in FEATURE_LEDGER}


def prohibited_columns() -> Set[str]:
    """Column names (as written) that must never enter a predictive model."""
    return {
        spec.name
        for spec in FEATURE_LEDGER
        if spec.availability in PROHIBITED_TIERS
    }


def allowed_columns(use_case: str = "post_crash_triage") -> Set[str]:
    """Column names allowed under a given use case."""
    tiers = USE_CASE_ALLOWED[use_case]
    return {spec.name for spec in FEATURE_LEDGER if spec.availability in tiers}


def classify_columns(columns, use_case: str = "post_crash_triage"):
    """
    Classify an arbitrary list of dataframe columns against the ledger.

    Returns a dict with keys: allowed, prohibited, unknown_timing (columns not in the
    ledger -> must be reviewed, treated as UNKNOWN and excluded by default).
    """
    ledger = ledger_by_name()
    allowed_tiers = USE_CASE_ALLOWED[use_case]
    allowed, prohibited, unknown = [], [], []
    for col in columns:
        spec = ledger.get(str(col).strip().lower())
        if spec is None:
            unknown.append(col)
        elif spec.availability in allowed_tiers:
            allowed.append(col)
        else:
            prohibited.append(col)
    return {"allowed": allowed, "prohibited": prohibited, "unknown_timing": unknown}
