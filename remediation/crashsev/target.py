"""
crashsev.target — fail-closed, audited target mapping (DATA-002, DATA-005).

The original project has two unsafe target paths:

* ``crashdata/importer.py::_coerce_severity`` returns ``"O"`` (property-damage-only) for
  blank, NaN, and any unrecognised value -> silent coercion of label errors into the
  majority/least-severe class (DATA-005).
* ``peyton_original/severity_mapping_utils.py::map_numeric_severity`` maps the *smallest*
  numeric code to 0 and the *largest* to 2 purely by sort order, collapsing all middle
  codes to 1 -> label inversion whenever the source codes severity in descending order,
  and an undocumented estimand (DATA-002).

This module replaces both with a single explicit, documented, fail-closed contract:

* Recognised KABCO codes map through ``schema.KABCO_TO_ORDINAL``.
* Everything else (blank, NaN, unknown token, unmapped code) maps to ``pd.NA`` and is
  *quarantined and counted*, never silently assigned a class.
* The raw label is preserved alongside the encoded label.
* A machine-readable mapping-audit table reconciles raw and encoded counts exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import schema


# Textual synonyms that unambiguously identify a KABCO code. Used only when the raw values
# are descriptive text rather than K/A/B/C/O letters. Kept intentionally small and
# unambiguous; anything not listed fails closed (returns None -> quarantine).
TEXT_TO_KABCO: Dict[str, str] = {
    # Fatal (K)
    "k": "K", "fatal": "K", "fatal injury": "K", "fatal injury (killed)": "K",
    "killed": "K", "death": "K", "deceased": "K",
    # Suspected serious / incapacitating (A)
    "a": "A", "serious": "A", "suspected serious injury": "A",
    "incapacitating": "A", "incapacitating injury": "A", "major injury": "A",
    # Suspected minor / non-incapacitating (B)
    "b": "B", "minor": "B", "suspected minor injury": "B",
    "non-incapacitating": "B", "non-incapacitating injury": "B", "visible injury": "B",
    # Possible (C)
    "c": "C", "possible": "C", "possible injury": "C", "complaint of pain": "C",
    # None / PDO (O)
    "o": "O", "none": "O", "no injury": "O", "no apparent injury": "O",
    "property damage only": "O", "pdo": "O", "property damage": "O",
}


@dataclass
class MappingAudit:
    """Machine-readable audit of a target mapping run (reconciles raw -> encoded)."""

    raw_value_counts: Dict[str, int]
    mapped_counts: Dict[int, int]
    quarantined_value_counts: Dict[str, int]
    n_total: int
    n_mapped: int
    n_quarantined: int
    kabco_to_ordinal: Dict[str, int] = field(default_factory=lambda: dict(schema.KABCO_TO_ORDINAL))
    blank_maps_to: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_mapped": self.n_mapped,
            "n_quarantined": self.n_quarantined,
            "blank_maps_to": self.blank_maps_to,
            "raw_value_counts": self.raw_value_counts,
            "mapped_counts": {int(k): int(v) for k, v in self.mapped_counts.items()},
            "quarantined_value_counts": self.quarantined_value_counts,
            "kabco_to_ordinal": self.kabco_to_ordinal,
            # Reconciliation invariant: mapped + quarantined == total, exactly.
            "reconciles": (self.n_mapped + self.n_quarantined) == self.n_total,
        }


def _normalise_raw_to_kabco(
    value,
    *,
    text_map: Dict[str, str],
    quarantine: set,
    blank_maps_to: Optional[str],
    numeric_code_map: Dict[str, str],
) -> Optional[str]:
    """Map one raw severity value to a KABCO letter, or None if it cannot be mapped.

    Precedence (fail-closed): blank -> ``blank_maps_to`` (or None); explicit quarantine token
    -> None; direct KABCO letter; codebook text synonym; codebook numeric code; else None.
    A blank value is treated as a class ONLY when the codebook explicitly says so
    (``blank_maps_to``) with recorded evidence; otherwise it is quarantined, never defaulted.
    """
    # 1. blank / NaN
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return blank_maps_to
    s = str(value).strip()
    if s == "":
        return blank_maps_to

    low = s.lower()
    # 2. explicit quarantine token (takes precedence over any mapping)
    if low in quarantine:
        return None

    # 3. direct KABCO letter (case-insensitive)
    up = s.upper()
    if up in schema.KABCO_SEVERITY_RANK:
        return up

    # 4. codebook text synonym
    if low in text_map:
        return text_map[low]

    # 5. codebook numeric code (EXACT string match only; never inferred by sort order)
    if s in numeric_code_map:
        return numeric_code_map[s]

    # 6. unmapped -> fail closed
    return None


def schema_unknown_tokens() -> set:
    """Unknown/placeholder tokens (mirrors DataCleaning/config.py UNKNOWN_STRINGS)."""
    return {
        "unknown", "missing", "unspecified", "not specified", "not applicable",
        "n/a", "na", "null", "blank", "tbd", "tba", "to be determined", "refused",
        "prefer not to say", "no data", "no value", "nan",
    }


def map_severity(
    raw: pd.Series,
    *,
    text_to_kabco: Optional[Dict[str, str]] = None,
    numeric_code_map: Optional[Dict[str, str]] = None,
    quarantine_labels: Optional[List[str]] = None,
    blank_maps_to: Optional[str] = None,
    kabco_to_ordinal: Optional[Dict[str, int]] = None,
) -> tuple[pd.Series, pd.Series, MappingAudit]:
    """
    Map a raw ``Crash Severity`` series to an ordinal target, fail-closed and codebook-driven.

    All mapping parameters come from the verified codebook (``data/target_mapping.yml``, loaded
    by :mod:`crashsev.contracts`); the defaults are convenience fallbacks for tests only.

    Parameters
    ----------
    text_to_kabco:
        raw text label (any case) -> KABCO letter. Defaults to the module ``TEXT_TO_KABCO``.
    numeric_code_map:
        verified string-of-numeric-code -> KABCO letter. Numeric codes are NEVER guessed by
        sort order (the DATA-002 defect); absent an entry here they fail closed.
    quarantine_labels:
        labels that are explicitly unusable (e.g. "Unknown", "Not Reported", "Null value") ->
        quarantined. Defaults to :func:`schema_unknown_tokens`.
    blank_maps_to:
        KABCO letter that a *blank/NaN* severity denotes for THIS source (e.g. "O" for the
        2009-2012 extract, verified), or ``None`` to fail closed and quarantine blanks. Must
        be evidenced in the codebook; enforced by the contract loader, not here.
    kabco_to_ordinal:
        KABCO letter -> ordinal class. Defaults to ``schema.KABCO_TO_ORDINAL``.

    Returns
    -------
    (kabco, y, audit):
        ``kabco`` : raw value normalised to a KABCO letter (or <NA>), preserved.
        ``y``     : the ordinal target as a nullable integer Series (<NA> where unmapped;
                    such rows must be quarantined by the caller, never assigned a class).
        ``audit`` : a :class:`MappingAudit` reconciling raw -> encoded counts.
    """
    text_map = {str(k).strip().lower(): str(v).strip().upper()
                for k, v in (text_to_kabco if text_to_kabco is not None else TEXT_TO_KABCO).items()}
    numeric_map = {str(k): str(v).strip().upper() for k, v in (numeric_code_map or {}).items()}
    quarantine = ({str(q).strip().lower() for q in quarantine_labels}
                  if quarantine_labels is not None else schema_unknown_tokens())
    blank = str(blank_maps_to).strip().upper() if blank_maps_to else None
    ordinal_map = kabco_to_ordinal if kabco_to_ordinal is not None else schema.KABCO_TO_ORDINAL

    kabco_vals: List[Optional[str]] = [
        _normalise_raw_to_kabco(
            v, text_map=text_map, quarantine=quarantine,
            blank_maps_to=blank, numeric_code_map=numeric_map,
        )
        for v in raw.tolist()
    ]

    kabco = pd.Series(kabco_vals, index=raw.index, dtype="object")
    y = kabco.map(ordinal_map).astype("Int64")

    # Build the audit.
    raw_counts = (
        raw.astype("object").where(raw.notna(), other="<NA>")
        .astype(str).value_counts().to_dict()
    )
    mapped_counts = y.dropna().astype(int).value_counts().sort_index().to_dict()
    quarantined_mask = y.isna()
    quarantined_counts = (
        raw[quarantined_mask].astype("object").where(raw[quarantined_mask].notna(), other="<NA>")
        .astype(str).value_counts().to_dict()
    )
    audit = MappingAudit(
        raw_value_counts={str(k): int(v) for k, v in raw_counts.items()},
        mapped_counts={int(k): int(v) for k, v in mapped_counts.items()},
        quarantined_value_counts={str(k): int(v) for k, v in quarantined_counts.items()},
        n_total=int(len(raw)),
        n_mapped=int(y.notna().sum()),
        n_quarantined=int(quarantined_mask.sum()),
        kabco_to_ordinal=dict(ordinal_map),
    )
    audit.blank_maps_to = blank
    return kabco, y, audit


def map_severity_from_mapping(raw: pd.Series, mapping) -> tuple[pd.Series, pd.Series, MappingAudit]:
    """Convenience: run :func:`map_severity` from a codebook mapping object.

    ``mapping`` is any object exposing the codebook fields (e.g.
    :class:`crashsev.contracts.TargetMapping`); this keeps :mod:`crashsev.target` free of an
    import dependency on :mod:`crashsev.contracts`.
    """
    return map_severity(
        raw,
        text_to_kabco=mapping.text_to_kabco,
        numeric_code_map=mapping.numeric_code_map,
        quarantine_labels=mapping.quarantine_labels,
        blank_maps_to=mapping.blank_maps_to,
        kabco_to_ordinal=mapping.kabco_to_ordinal,
    )
