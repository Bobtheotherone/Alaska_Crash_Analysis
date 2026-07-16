"""
crashsev.synth — SYNTHETIC STRUCTURAL FIXTURE (TEST-ONLY).

>>> THIS IS NOT REAL DATA AND MUST NEVER BE USED FOR AN EMPIRICAL CLAIM. <<<

Per the remediation blocking rules (Section 9), synthetic data is permitted ONLY to
exercise and test the harness. This generator produces a dataframe with the *structure* of
the Alaska crash extracts (column names from Column Analysis.txt, KABCO "Crash Severity",
Year 2009-2017, a crash-id group column) so the corrected pipeline, its tests, and the
leakage-optimism demonstration can run end-to-end without the real dataset.

It builds in three deliberate properties:
  1. Realistic class imbalance (~68% O, ~29% B/C, ~3.5% A/K), mirroring the real test-set
     prevalence, so ordinal metrics and imbalance behaviour are exercised.
  2. A modest genuine signal from *legitimate* at-event/pre-event features, so a good model
     can beat trivial baselines by a believable margin (not perfectly).
  3. A temporal drift across years, so a random split is optimistic relative to a
     chronological split (this is what the optimism-gap demo measures), and one
     outcome-derived leakage column ("Number of Fatalities") for the leakage sentinel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema

_WEATHER = ["Clear", "Cloudy", "Rain", "Snow", "Sleet", "Fog", "Unknown"]
_ROAD_SURFACE = ["Dry", "Wet", "Ice/Frost", "Snow", "Slush", "Unknown"]
_LIGHTING = ["Daylight", "Dark - Lighted", "Dark - Not Lighted", "Dusk", "Dawn", "Unknown"]
_CRASH_TYPE = ["Rear End", "Head-On", "Sideswipe", "Angle", "Fixed Object", "Motorcycle",
               "Pedestrian", "Rollover", "Animal"]
_MANNER = ["Rear-End", "Head-On", "Angle", "Sideswipe-Same", "Not a Collision", "Front-To-Rear"]
_FUNC_CLASS = ["Interstate", "Principal Arterial", "Minor Arterial", "Collector", "Local"]
_REGION = ["Central", "Northern", "Southeast", "Southcoast"]
_ALCOHOL = ["No", "Yes", "Unknown"]


def make_synthetic_crash_df(n: int = 12000, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic Alaska-shaped crash dataframe. TEST-ONLY."""
    rng = np.random.default_rng(seed)

    years = rng.choice(np.arange(2009, 2018), size=n, p=_year_probs())
    crash_number = np.array([f"AK{2009}{i:07d}" for i in range(n)])  # one row per crash

    weather = rng.choice(_WEATHER, size=n, p=_p(_WEATHER, {"Clear": 0.5, "Snow": 0.12, "Rain": 0.1}))
    road = rng.choice(_ROAD_SURFACE, size=n, p=_p(_ROAD_SURFACE, {"Dry": 0.55, "Ice/Frost": 0.15}))
    lighting = rng.choice(_LIGHTING, size=n, p=_p(_LIGHTING, {"Daylight": 0.55, "Dark - Not Lighted": 0.18}))
    crash_type = rng.choice(_CRASH_TYPE, size=n, p=_p(_CRASH_TYPE, {"Rear End": 0.3, "Angle": 0.2}))
    manner = rng.choice(_MANNER, size=n)
    func_class = rng.choice(_FUNC_CLASS, size=n, p=_p(_FUNC_CLASS, {"Local": 0.35, "Collector": 0.2}))
    region = rng.choice(_REGION, size=n, p=_p(_REGION, {"Central": 0.5}))
    alcohol = rng.choice(_ALCOHOL, size=n, p=_p(_ALCOHOL, {"No": 0.8, "Yes": 0.12}))
    aadt = np.clip(rng.lognormal(mean=8.5, sigma=1.0, size=n), 50, 120000).round().astype(int)
    posted_speed = rng.choice([15, 25, 35, 45, 55, 65], size=n, p=[0.05, 0.2, 0.25, 0.2, 0.2, 0.1])
    n_units = rng.choice([1, 2, 3], size=n, p=[0.25, 0.65, 0.10])

    # ---- Latent severity score from LEGITIMATE features (modest signal) ----
    z = np.zeros(n, dtype="float64")
    z += np.where(crash_type == "Head-On", 1.4, 0.0)
    z += np.where(crash_type == "Motorcycle", 1.3, 0.0)
    z += np.where(crash_type == "Pedestrian", 1.5, 0.0)
    z += np.where(crash_type == "Rollover", 0.8, 0.0)
    z += np.where(alcohol == "Yes", 1.0, 0.0)
    z += np.where(np.isin(lighting, ["Dark - Not Lighted"]), 0.5, 0.0)
    z += np.where(np.isin(road, ["Ice/Frost", "Snow"]), 0.25, 0.0)
    z += (posted_speed - 40) / 30.0
    z += np.where(n_units == 1, 0.3, 0.0)  # single-vehicle often more severe
    # temporal drift: baseline severity slowly declines over years (reporting/vehicle safety)
    z += (2013 - years) * 0.06
    # noise
    z += rng.normal(0, 1.1, size=n)

    # ---- Map latent to KABCO with realistic imbalance via quantile cutpoints ----
    # target marginal ~ O:0.676, B/C:0.288, A/K:0.036
    q = np.quantile(z, [0.676, 0.676 + 0.288 * 0.6, 0.964, 0.964 + 0.036 * 0.45])
    kabco = np.empty(n, dtype=object)
    kabco[z <= q[0]] = "O"
    kabco[(z > q[0]) & (z <= q[1])] = "C"
    kabco[(z > q[1]) & (z <= q[2])] = "B"
    kabco[(z > q[2]) & (z <= q[3])] = "A"
    kabco[z > q[3]] = "K"

    # ---- Outcome-derived leakage column ----
    n_fatalities = np.where(kabco == "K", rng.integers(1, 3, size=n), 0)
    n_serious = np.where(kabco == "A", rng.integers(1, 3, size=n), 0)

    # inject a few unmapped/blank severity values to exercise fail-closed target mapping
    sev_display = kabco.copy()
    blank_idx = rng.choice(n, size=max(1, n // 500), replace=False)
    sev_display[blank_idx] = rng.choice(["", "UNKNOWN", "N/A"], size=len(blank_idx))

    df = pd.DataFrame({
        "Crash Number": crash_number,
        "Year": years,
        "Month": rng.integers(1, 13, size=n),
        "Day of the Week": rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], size=n),
        "Time of Day": rng.integers(0, 24, size=n),
        "Weather": weather,
        "Road Surface": road,
        "Lighting": lighting,
        "Crash Type": crash_type,
        "Manner of Collision": manner,
        "Functional Class": func_class,
        "Region": region,
        "Alcohol Suspected": alcohol,
        "AADT": aadt,
        "Posted Speed": posted_speed,
        "Number of Motorized Units": n_units,
        # outcome-derived (must be excluded by the leakage denylist):
        "Number of Fatalities": n_fatalities,
        "Number of Serious Injuries": n_serious,
        # target:
        "Crash Severity": sev_display,
    })
    return df


def _year_probs():
    # slightly more crashes in later years
    w = np.array([0.9, 0.92, 0.95, 0.98, 1.0, 1.03, 1.06, 1.1, 1.14])
    return w / w.sum()


def _p(levels, overrides):
    """Build a probability vector over ``levels`` with given overrides, remainder uniform."""
    p = {lvl: None for lvl in levels}
    for k, v in overrides.items():
        p[k] = v
    fixed = sum(v for v in p.values() if v is not None)
    n_free = sum(1 for v in p.values() if v is None)
    rem = max(0.0, 1.0 - fixed) / max(1, n_free)
    vec = np.array([p[lvl] if p[lvl] is not None else rem for lvl in levels], dtype="float64")
    return vec / vec.sum()
