"""crashsev.cluster_bootstrap_sensitivity — proxy-cluster dependence sensitivity (DEP-001, v4.1).

POST HOC, retrospective, deterministic. NOT part of the frozen one-shot benchmark; modifies no
frozen artifact. The published headline interval is a crash-level case bootstrap under a
cross-crash independence approximation (no corridor/agency/reporting-batch cluster id survives
in the de-identified modeling table). This addendum resamples PROXY clusters that DO exist in
the modeling table — calendar day, calendar week, and geographic groupings — to probe how much
residual dependence along those axes could widen the primary paired interval. No proxy is a
validated dependence model. NOTE (corrected 2026-07-15): the licensed raw extract DID contain
agency-related fields (Officer Agency, Reporting Agency, Detachment; data/column_inventory_09_12.csv)
and route/milepoint fields; all were removed as identifier-tier columns during de-identification
(build_modeling_table.DROP_IDENTIFIERS), so they are absent from the released evidence and the
retained local table, not absent from the source (see agency_leave_one_out.json).

Usage (from remediation/):
    python -m crashsev.cluster_bootstrap_sensitivity --data _local_data/modeling_table_09_12.csv \
        --runs runs/final_8af9d5bc23d8 --out experiment
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
MIN_CLUSTERS_FOR_BOOTSTRAP = 30


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PKG_ROOT, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _cluster_ci(err_a, err_b, labels, seed, reps):
    """Percentile CI for mean(err_a) - mean(err_b) resampling whole clusters with replacement."""
    labels = np.asarray(labels)
    uniq, inv = np.unique(labels, return_inverse=True)
    rows_by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for r in range(reps):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([rows_by[i] for i in pick])
        diffs[r] = err_a[rows].mean() - err_b[rows].mean()
    sizes = np.array([len(r) for r in rows_by])
    return {
        "n_clusters": int(len(uniq)),
        "cluster_size_min_median_max": [int(sizes.min()), float(np.median(sizes)), int(sizes.max())],
        "point": float(err_a.mean() - err_b.mean()),
        "ci95_percentile": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "excludes_zero": bool(np.percentile(diffs, 97.5) < 0 or np.percentile(diffs, 2.5) > 0),
        "replicates": int(reps),
    }


def _leave_one_out(err_a, err_b, labels) -> list:
    labels = pd.Series(np.asarray(labels))
    out = []
    for val, cnt in labels.value_counts().items():
        keep = (labels != val).to_numpy()
        out.append({"left_out": str(val), "rows_left_out": int(cnt),
                    "delta_omae_without_group": float(err_a[keep].mean() - err_b[keep].mean())})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Proxy-cluster dependence sensitivity (post hoc).")
    ap.add_argument("--data", default=str(PKG_ROOT / "_local_data" / "modeling_table_09_12.csv"))
    ap.add_argument("--runs", default=str(PKG_ROOT / "runs" / "final_8af9d5bc23d8"))
    ap.add_argument("--out", default=str(PKG_ROOT / "experiment"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    orf = pd.read_csv(Path(args.runs) / "predictions_ordinal_random_forest.csv",
                      float_precision="round_trip")
    y = orf["y_true"].to_numpy(int)
    err_a = np.abs(y - orf["y_pred"].to_numpy(int)).astype(float)
    err_b = np.abs(y - 0).astype(float)          # majority predicts class 0 for every crash
    ids = orf["group_id"].astype(str).to_numpy()

    mt = pd.read_csv(args.data, low_memory=False,
                     usecols=["Crash Number", "DateTime", "Week of the Year", "Year",
                              "Region", "Borough", "Maintenance Responsibility"])
    mt["Crash Number"] = mt["Crash Number"].astype(str)
    mt = mt.set_index("Crash Number").loc[ids]
    assert len(mt) == len(ids)

    def key(series, prefix):
        s = series.astype("object").where(series.notna(), other=f"{prefix}__missing__").astype(str)
        return s.str.strip().to_numpy(), int((s.astype(str).str.contains("__missing__")).sum())

    day_raw = pd.to_datetime(mt["DateTime"], errors="coerce")
    day = np.where(day_raw.notna(), day_raw.dt.strftime("%Y-%m-%d"), "day__missing__")
    week = np.where(mt["Week of the Year"].notna(),
                    "2012-W" + pd.to_numeric(mt["Week of the Year"], errors="coerce")
                    .fillna(-1).astype(int).astype(str).str.zfill(2), "week__missing__")
    region, region_miss = key(mt["Region"], "region")
    borough, borough_miss = key(mt["Borough"], "borough")
    maint, maint_miss = key(mt["Maintenance Responsibility"], "maint")

    res = {
        "analysis": "cluster_bootstrap_sensitivity",
        "status": "POST HOC (v4.1 revision addendum): proxy-cluster resampling of the primary "
                  "paired contrast (ordinal_random_forest minus majority, 2012). NOT prespecified; "
                  "no proxy is a validated dependence model; the frozen headline interval is "
                  "unchanged and remains the published estimate.",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "source_run": "final_8af9d5bc23d8",
        "headline_case_bootstrap_ci_for_reference": [-0.0231, -0.0063],
        "seed": args.seed, "replicates": args.resamples,
        "missing_key_rule": "rows with a missing cluster key form one pooled '<axis>__missing__' "
                            "cluster; counts reported per axis",
        "clusters": {},
        "limitations": [
            "calendar clusters capture shared-day/week shocks (weather, events) only",
            "geographic clusters are coarse and few; they proxy region-level practice, not agency",
            "no route/roadway grouping exists in the licensed extract (route/milepoint are "
            "excluded identifiers)",
            "cluster axes are proxies chosen post hoc; none is a validated dependence structure",
        ],
    }

    for name, labels, miss in (("calendar_day", day, int((pd.Series(day) == "day__missing__").sum())),
                               ("calendar_week", week, int((pd.Series(week) == "week__missing__").sum())),
                               ("region", region, region_miss),
                               ("borough", borough, borough_miss),
                               ("maintenance_responsibility", maint, maint_miss)):
        n_uniq = len(np.unique(labels))
        entry = {"missing_key_rows": miss}
        if n_uniq >= MIN_CLUSTERS_FOR_BOOTSTRAP:
            entry.update(_cluster_ci(err_a, err_b, labels, args.seed, args.resamples))
            entry["method"] = "cluster bootstrap (resample clusters with replacement)"
        else:
            entry["method"] = (f"leave-one-group-out point deltas (only {n_uniq} groups; too few "
                               f"for a stable cluster bootstrap)")
            entry["n_clusters"] = n_uniq
            entry["point"] = float(err_a.mean() - err_b.mean())
            entry["leave_one_out"] = _leave_one_out(err_a, err_b, labels)
        res["clusters"][name] = entry

    jpath = out / "cluster_bootstrap_sensitivity.json"
    jpath.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[cluster-boot] wrote {jpath}")
    for k, v in res["clusters"].items():
        if "ci95_percentile" in v:
            print(f"   {k}: n={v['n_clusters']} point {v['point']:+.4f} "
                  f"CI [{v['ci95_percentile'][0]:+.4f}, {v['ci95_percentile'][1]:+.4f}]")
        else:
            rng_ = [d["delta_omae_without_group"] for d in v["leave_one_out"]]
            print(f"   {k}: n={v['n_clusters']} LOO delta range [{min(rng_):+.4f}, {max(rng_):+.4f}]")

    # agency LOO artifact: agency fields exist in the raw source but were removed as
    # identifier-tier columns at de-identification; document + provide the nearest proxies
    agency = {
        "analysis": "agency_leave_one_out",
        "status": "DEFERRED_EXTERNAL_EVIDENCE — the licensed raw extract CONTAINS agency-related "
                  "fields (Officer Agency, Reporting Agency, Detachment; see "
                  "data/column_inventory_09_12.csv), but they were removed as identifier-tier "
                  "columns during de-identification (crashsev/build_modeling_table.py "
                  "DROP_IDENTIFIERS) before the retained modeling table was built, and the raw "
                  "file is absent from this analysis environment by data-handling policy. No "
                  "row-level agency label can therefore be joined to the frozen evaluation rows; "
                  "a true leave-one-agency-out / agency-cluster analysis requires licensed "
                  "re-supply of the agency fields. The nearest available administrative/geographic "
                  "proxies are reported below (post hoc; descriptive only). CORRECTION 2026-07-15: "
                  "an earlier status line said the extract contained no such field; that statement "
                  "conflated the de-identified modeling table with the licensed source extract.",
        "generated_utc": res["generated_utc"],
        "git_commit": res["git_commit"],
        "source_run": "final_8af9d5bc23d8",
        "proxies": {
            "region_leave_one_out": _leave_one_out(err_a, err_b, region),
            "maintenance_responsibility_leave_one_out": _leave_one_out(err_a, err_b, maint),
        },
        "reading": "Every leave-one-group-out delta stays negative iff the primary contrast does "
                   "not depend on a single region/maintenance group; see values.",
    }
    apath = out / "agency_leave_one_out.json"
    apath.write_text(json.dumps(agency, indent=2), encoding="utf-8")
    print(f"[cluster-boot] wrote {apath}")


if __name__ == "__main__":
    main()
