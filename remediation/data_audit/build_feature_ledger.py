"""Build the feature-availability ledger for the real 2009-2012 extract (AA2-005).

The ledger is the leakage guardrail: every one of the extract's columns is classified by
*when its value becomes known relative to the crash's injury outcome*, so that outcome-derived
fields can never enter a predictive model. This script encodes an explicit, reviewable
classification and asserts it covers **exactly** the columns present in the extract (no column
silently unclassified, no phantom column) before writing
``data/feature_availability_ledger.csv``.

Tiers
-----
* ``target``        — the outcome itself.
* ``post_outcome``  — known only *because of* the outcome (injury/fatality counts, EMS
                      response, damage extent, enforcement). **Prohibited** for prediction.
* ``identifier``    — keys, point-location, and free-text location. Dropped (grouping/privacy).
* ``temporal``      — calendar features. ``Year`` is withheld (split axis / source-version).
* ``pre_event``     — roadway / traffic / geographic context known before the crash.
* ``at_event``      — crash circumstances, dynamics, vehicle & occupant attributes observed
                      at the scene that are *not* harm measurements.
* ``constant``      — single-valued in this extract; dropped.

Allowed for the ``post_crash_triage`` use case: ``pre_event`` + ``at_event`` + ``temporal``
(minus ``Year``). Everything else is excluded. A small set of ``at_event`` occupant-kinematic
features (ejection, restraint, seat, damage location) are flagged ``high_coupling`` so the study
can ablate them and show the result is not carried by near-outcome proxies.

Run: ``python data_audit/build_feature_ledger.py``  (reads ``data/column_inventory_09_12.csv``)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"

TARGET = ["Crash Severity"]

POST_OUTCOME = {
    "Number of Fatalities": "injury/fatality count — defines K; direct leak",
    "Number of Injuries with Fatalities": "injury count — direct leak",
    "Number of Injuries without Fatailites": "injury count (sic) — direct leak",
    "Unit 1 Person 1 Injury": "the person's injury code — IS the outcome",
    "Unit 1 Person 1 Transported": "EMS transport — post-crash response to injury",
    "Unit 1 Person 1 Transported By": "EMS transport mode — post-crash response",
    "Unit 1 Person 1 Transported To": "destination facility — post-crash response",
    "Unit 1 Person 1 Extricated": "extrication — post-crash response to entrapment/injury",
    "Unit 1 Person 1 NFR": "ambiguous post-report field — excluded conservatively",
    "Unit 1 Damage": "vehicle damage extent — co-measurement of harm",
    "Unit 1 Undercarriage Damage": "rollover/undercarriage damage extent — co-measurement of harm",
    "Unit 1 Towed": "tow disposition — post-crash outcome of damage",
    "Arrest": "post-crash enforcement",
    "Unit 1 Person 1 Violations": "post-crash adjudication",
}

IDENTIFIER = {
    "Crash Number": "crash id — row id AND group key (one row per crash)",
    "CDS Number": "location/segment id",
    "Report ID": "report id",
    "Officer ID": "reporting-officer id (PII)",
    "Officer Agency": "reporting agency id",
    "Reporting Agency": "reporting agency id",
    "Detachment": "agency detachment id",
    "Maintenance Station": "operational station id (location proxy)",
    "Route": "route id — fine location, re-identifying",
    "Street": "free-text street — fine location, re-identifying",
    "Intersecting Street": "free-text cross street — fine location, re-identifying",
    "Latitude": "point coordinate — restricted (privacy)",
    "Longitude": "point coordinate — restricted (privacy)",
    "Milepoint": "continuous location along route — re-identifying",
    "Date": "timestamp — used to derive the split, not a feature",
    "DateTime": "timestamp — split ordering key, not a feature",
}

TEMPORAL = {
    "Year": "split axis / source-version proxy — WITHHELD as a feature",
    "Month": "calendar month — seasonality",
    "Day of Month": "day-of-month — weak; retained, no strong mechanism",
    "Day of the Week": "weekday/weekend pattern",
    "Time of Day": "hour band — lighting/traffic-exposure proxy",
    "Week of the Year": "week-of-year — seasonality (correlated with Month)",
}

PRE_EVENT = {
    "AHS System": "Alaska Highway System class",
    "At Intersection": "intersection-related location",
    "Borough": "coarse administrative geography",
    "Census Area": "coarse administrative geography",
    "City": "city/place",
    "Election District": "administrative polygon (not a point)",
    "Functional Class": "roadway functional class",
    "Maintenance Category": "roadway maintenance category",
    "Maintenance Responsibility": "maintenance responsibility",
    "NHS System": "National Highway System indicator",
    "Posted Speed": "posted speed limit",
    "Region": "DOT region (coarse geography)",
    "Roadway Characteristics": "roadway geometry/character",
    "Roadway Junction": "junction type",
    "Rural Urban": "rural/urban context",
    "AADT": "annual average daily traffic (numeric; sentinel-cleaned)",
    "Distance From Intersection": "distance to nearest intersection (numeric)",
}

AT_EVENT = {
    "Direction": "travel direction context",
    "First Sequence Event": "first harmful event / manner",
    "First Sequence Location": "location of first event (roadway/shoulder/…)",
    "Hit and Run": "hit-and-run flag (scene fact)",
    "Lighting": "lighting condition",
    "Non Vehicle Damage": "struck non-vehicle property (collision configuration)",
    "Number of Commercial Vehicles Involved": "count of CVs involved",
    "Number of Pedestrians Involved": "count of pedestrians involved",
    "Number of Persons Involved": "crash size (persons)",
    "Number of Units Involved": "count of units (vehicles) involved",
    "Pavement": "pavement type/condition",
    "Road Surface": "road surface condition (snow/ice/wet…)",
    "Temperature": "ambient temperature at crash time",
    "Temperature Range": "binned temperature at crash time",
    "Weather": "weather condition",
    "Unit 1 Action": "primary unit's action/maneuver",
    "Unit 1 Commercial Vehicle (CV)": "is unit a commercial vehicle",
    "Unit 1 CV Configuration": "CV configuration",
    "Unit 1 CV Haz-Mat Released": "haz-mat release flag",
    "Unit 1 CV Issuing Authority": "CV issuing authority",
    "Unit 1 CV Placard": "CV placard",
    "Unit 1 Direction of Travel": "unit direction of travel",
    "Unit 1 Model Year": "vehicle model year (age proxy)",
    "Unit 1 Non-CV Configuration": "non-CV body configuration",
    "Unit 1 Occupants": "number of occupants in unit",
    "Unit 1 Primary Contributing Circumstance": "primary contributing circumstance (cause)",
    "Unit 1 Primary Damage Location": "impact location on vehicle (collision config)",
    "Unit 1 Road Circumstance": "road circumstance contributing",
    "Unit 1 Secondary Sequence of Events": "secondary events in the crash sequence",
    "Unit 1 Traffic Control": "traffic control present",
    "Unit 1 Person 1 Age Range": "driver age band",
    "Unit 1 Person 1 Alcohol or Drug Use Suspected": "impairment suspected at scene",
    "Unit 1 Person 1 Circumstance": "person contributing circumstance (behavior)",
    "Unit 1 Person 1 Environment Circumstance": "environmental circumstance",
    "Unit 1 Person 1 Gender": "driver gender",
    "Unit 1 Person 1 Insurance Coverage": "insurance coverage (scene fact)",
    "Unit 1 Person 1 Raw Age": "driver age (raw)",
    "Unit 1 Person 1 Residence State": "resident vs visitor (scene fact)",
    "Unit 1 Person 1 Test Given": "impairment test administered at scene",
    "Unit 1 Person 1 Type": "person type (driver/…)",
    "Unit 1 License Plate State": "plate state (local vs out-of-state)",
    # --- high-coupling occupant kinematics: allowed but ablatable ---
    "Unit 1 Person 1 Ejected": "ejection (kinematics) — HIGH-COUPLING with injury",
    "Unit 1 Person 1 Restraint": "restraint use — HIGH-COUPLING with injury",
    "Unit 1 Person 1 Seat Location": "seat position — HIGH-COUPLING with injury",
}

HIGH_COUPLING = {
    "Unit 1 Person 1 Ejected",
    "Unit 1 Person 1 Restraint",
    "Unit 1 Person 1 Seat Location",
}

CONSTANT = {
    "Location Type": "single-valued in extract — dropped",
    "Unit 1 Make": "single-valued in extract — dropped",
}

ALLOWED_TIERS = {"pre_event", "at_event", "temporal"}
# Year is temporal but withheld:
WITHHELD_FROM_FEATURES = {"Year"}


def main() -> None:
    inv = pd.read_csv(DATA_DIR / "column_inventory_09_12.csv")
    cols = set(inv["column"])

    tiers: dict[str, str] = {}
    notes: dict[str, str] = {}
    for c in TARGET:
        tiers[c], notes[c] = "target", "3-level ordinal outcome"
    for grp, tier in [
        (POST_OUTCOME, "post_outcome"), (IDENTIFIER, "identifier"),
        (TEMPORAL, "temporal"), (PRE_EVENT, "pre_event"),
        (AT_EVENT, "at_event"), (CONSTANT, "constant"),
    ]:
        for c, note in grp.items():
            if c in tiers:
                raise SystemExit(f"column classified twice: {c!r} ({tiers[c]} and {tier})")
            tiers[c], notes[c] = tier, note

    classified = set(tiers)
    missing = cols - classified          # in data but unclassified -> fail closed
    phantom = classified - cols          # classified but not in data -> typo
    if missing:
        raise SystemExit(f"{len(missing)} column(s) in the extract are UNCLASSIFIED: {sorted(missing)}")
    if phantom:
        raise SystemExit(f"{len(phantom)} classified column(s) are NOT in the extract (typo?): {sorted(phantom)}")

    rows = []
    for c in sorted(cols):
        tier = tiers[c]
        allowed = (tier in ALLOWED_TIERS) and (c not in WITHHELD_FROM_FEATURES)
        rows.append({
            "column": c,
            "availability": tier,
            "allowed_post_crash_triage": allowed,
            "high_coupling": c in HIGH_COUPLING,
            "rationale": notes[c],
        })
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "feature_availability_ledger.csv", index=False)

    summary = out.groupby("availability").agg(
        n=("column", "size"), n_allowed=("allowed_post_crash_triage", "sum")
    )
    print(f"classified {len(out)} columns (all {len(cols)} extract columns covered)\n")
    print(summary.to_string())
    print(f"\nallowed features (post_crash_triage): {int(out['allowed_post_crash_triage'].sum())}")
    print(f"prohibited (post_outcome): {int((out['availability']=='post_outcome').sum())}")
    print(f"high-coupling (ablatable): {int(out['high_coupling'].sum())}")


if __name__ == "__main__":
    main()
