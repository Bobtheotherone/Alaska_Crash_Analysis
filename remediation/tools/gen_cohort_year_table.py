"""Generate the year-by-year cohort table (r3 pass, L-10 / Phase 5D).

Computes, from the licensed local modeling table (the canonical-hash-pinned analysis
input), the per-year cohort accounting: source rows, mapped rows, quarantined rows,
working-class counts, and class-2 prevalence for 2009-2012 — and RECONCILES every
aggregate against the frozen artifacts before writing anything:

  * totals vs experiment/final_results.json target_audit / split (50,543 / 46,844 /
    3,699; 35,214 development; 11,630 final; class totals 32,046 / 13,047 / 1,751);
  * development-side totals vs experiment/development_report.json (38,003 / 35,214 /
    2,789);
  * per-year source rows and blank/quarantine shares vs the post-hoc reporting audit
    (experiment/target_reporting_process_audit.json, q6).

Emits ``experiment/cohort_year_table.json`` (committed) and
``paper/latex/cohort_year_table.tex`` (generated LaTeX, ``\\input`` by the manuscript).

``--check`` is data-free: re-renders the LaTeX from the committed JSON, byte-compares,
and re-runs every frozen-artifact reconciliation; when the local table is present it
additionally recomputes the counts.

Usage: python tools/gen_cohort_year_table.py [--check]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REM))

CELLS = REM / "experiment" / "cohort_year_table.json"
TEX = REM / "paper" / "latex" / "cohort_year_table.tex"
LOCAL = REM / "_local_data" / "modeling_table_09_12.csv"

MAP = {"<NA>": 0, "Possible": 1, "Non-Incapacitating": 1, "Incapacitating": 2, "Fatal": 2}


def build() -> dict:
    import pandas as pd
    df = pd.read_csv(LOCAL, usecols=["Year", "Crash Severity"],
                     dtype={"Year": "Int64", "Crash Severity": "string"})
    sev = df["Crash Severity"].fillna("<NA>").map(MAP)
    years = sorted(int(y) for y in df["Year"].dropna().unique())
    rows = {}
    for y in years:
        my = df["Year"] == y
        cls = {str(c): int(((sev == c) & my).sum()) for c in (0, 1, 2)}
        mapped = sum(cls.values())
        rows[str(y)] = {
            "source_rows": int(my.sum()),
            "mapped_rows": mapped,
            "quarantined_rows": int(my.sum()) - mapped,
            "class_counts": cls,
            "class2_share_of_mapped": cls["2"] / mapped,
            "role": "Final retrospective evaluation" if y == 2012 else "Development",
        }
    art = {
        "description": (
            "Year-by-year cohort accounting of the licensed 2009-2012 extract "
            "(tab:cohort-year): source rows, mapped rows (researcher-defined working "
            "classes 0/1/2), quarantined rows (Unknown / Not Reported / Null value), and "
            "class composition, computed from the canonical modeling table and reconciled "
            "against the frozen target audits before writing."
        ),
        "sources": ["licensed local modeling table (canonical content hash pinned in the "
                    "evidence chain)",
                    "experiment/final_results.json (target_audit, split)",
                    "experiment/development_report.json (target_audit_dev_only)",
                    "experiment/target_reporting_process_audit.json (q6)"],
        "years": rows,
    }
    problems = reconcile(art)
    assert not problems, problems
    return art


def reconcile(art: dict) -> list:
    problems = []
    fr = json.loads((REM / "experiment" / "final_results.json").read_text("utf-8"))
    dr = json.loads((REM / "experiment" / "development_report.json").read_text("utf-8"))
    tr = json.loads((REM / "experiment" / "target_reporting_process_audit.json").read_text("utf-8"))
    rows = art["years"]
    if sorted(rows) != ["2009", "2010", "2011", "2012"]:
        return ["unexpected year set"]

    def tot(f):
        return sum(rows[y][f] for y in rows)

    ta = fr["target_audit"]
    checks = [
        ("source total", tot("source_rows"), ta["n_total"]),
        ("mapped total", tot("mapped_rows"), ta["n_mapped"]),
        ("quarantined total", tot("quarantined_rows"), ta["n_quarantined"]),
        ("development mapped", sum(rows[y]["mapped_rows"] for y in ("2009", "2010", "2011")),
         fr["split"]["n_development"]),
        ("final mapped", rows["2012"]["mapped_rows"], fr["split"]["n_final_test"]),
        ("dev-year source rows", sum(rows[y]["source_rows"] for y in ("2009", "2010", "2011")),
         dr["target_audit_dev_only"]["n_total"]),
        ("dev quarantined", sum(rows[y]["quarantined_rows"] for y in ("2009", "2010", "2011")),
         dr["target_audit_dev_only"]["n_quarantined"]),
    ]
    for c in ("0", "1", "2"):
        checks.append((f"class-{c} total",
                       sum(rows[y]["class_counts"][c] for y in rows), ta["mapped_counts"][c]))
    for name, got, want in checks:
        if got != want:
            problems.append(f"{name}: {got} != frozen {want}")
    q6 = tr["q6_blank_share_by_year"]
    for y in rows:
        c0 = rows[y]["class_counts"]["0"]
        if rows[y]["source_rows"] != q6[y]["n"]:
            problems.append(f"{y} source rows {rows[y]['source_rows']} != audit q6 {q6[y]['n']}")
        if round(c0 / rows[y]["source_rows"], 4) != q6[y]["blank_share"]:
            problems.append(f"{y} blank share mismatch vs audit q6")
        if round(rows[y]["quarantined_rows"] / rows[y]["source_rows"], 4) != q6[y]["quarantined_share"]:
            problems.append(f"{y} quarantined share mismatch vs audit q6")
    return problems


def render_tex(art: dict) -> str:
    rows = art["years"]

    def cc(y, c):
        d = rows[y]["class_counts"]
        return d[str(c)] if str(c) in d else d[c]

    lines = [
        "% GENERATED FILE — do not edit by hand.",
        "% Source: tools/gen_cohort_year_table.py (canonical modeling table, reconciled",
        "% against the frozen target audits).",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Year-by-year cohort accounting (mapped working classes and quarantine).}",
        "\\label{tab:cohort-year}",
        "\\footnotesize",
        "\\begingroup\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabularx}{\\textwidth}{@{}l R{0.0815\\textwidth} R{0.0815\\textwidth} "
        "R{0.107\\textwidth} R{0.085\\textwidth} R{0.085\\textwidth} R{0.075\\textwidth} "
        "R{0.09\\textwidth} Y@{}}",
        "\\toprule",
        "\\textbf{Year} & \\textbf{Source rows} & \\textbf{Mapped} & \\textbf{Quarantined} & "
        "\\textbf{Class 0} & \\textbf{Class 1} & \\textbf{Class 2} & \\textbf{Class-2 share} & "
        "\\textbf{Role} \\\\",
        "\\midrule",
    ]
    short_role = {"Final retrospective evaluation": "Final retrospective",
                  "Development": "Development"}
    for y in ("2009", "2010", "2011", "2012"):
        r = rows[y]
        lines.append(
            f"{y} & {r['source_rows']:,} & {r['mapped_rows']:,} & {r['quarantined_rows']:,} & "
            f"{cc(y,0):,} & {cc(y,1):,} & {cc(y,2):,} & {r['class2_share_of_mapped']:.3f} & "
            f"{short_role[r['role']]} \\\\")
        if y == "2011":
            dev = [sum(rows[x][f] for x in ("2009", "2010", "2011"))
                   for f in ("source_rows", "mapped_rows", "quarantined_rows")]
            devc = [sum(cc(x, c) for x in ("2009", "2010", "2011")) for c in (0, 1, 2)]
            lines.append("\\midrule")
            lines.append(
                f"2009--2011 & {dev[0]:,} & {dev[1]:,} & {dev[2]:,} & {devc[0]:,} & "
                f"{devc[1]:,} & {devc[2]:,} & {devc[2]/dev[1]:.3f} & Development total \\\\")
            lines.append("\\midrule")
    tot = [sum(rows[y][f] for y in rows) for f in ("source_rows", "mapped_rows", "quarantined_rows")]
    totc = [sum(cc(y, c) for y in rows) for c in (0, 1, 2)]
    lines += [
        "\\midrule",
        f"Total & {tot[0]:,} & {tot[1]:,} & {tot[2]:,} & {totc[0]:,} & {totc[1]:,} & "
        f"{totc[2]:,} & {totc[2]/tot[1]:.3f} & --- \\\\",
        "\\bottomrule",
        "\\end{tabularx}",
        "\\endgroup",
        "\\tabnote{Counts are computed from the canonical modeling table and reconcile exactly "
        "with the frozen target audits (\\artifact{experiment/final_results.json}, "
        "\\artifact{experiment/development_report.json}) and the per-year audit shares "
        "(\\artifact{experiment/target_reporting_process_audit.json}, q6). Quarantined = "
        "\\texttt{Unknown}, \\texttt{Not Reported}, or \\texttt{Null value} severity; never "
        "coerced into a class. Class-2 share is of mapped rows. Development rows feed the "
        "rolling-origin folds of \\cref{sec:selection}; 2012 is the historically exposed final "
        "retrospective cohort.}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    check = "--check" in sys.argv[1:]
    if check:
        if not CELLS.exists() or not TEX.exists():
            print("[FAIL] cohort-year artifacts missing; regenerate")
            return 1
        art = json.loads(CELLS.read_text("utf-8"))
        problems = reconcile(art)
        if TEX.read_text("utf-8") != render_tex(art):
            problems.append("cohort_year_table.tex stale w.r.t. committed cells JSON")
        if LOCAL.exists():
            regen = build()
            if json.dumps(regen, indent=2) + "\n" != CELLS.read_text("utf-8"):
                problems.append("cohort_year_table.json stale w.r.t. regeneration from local data")
        else:
            print("[SKIP] licensed local table absent: count recomputation skipped "
                  "(reconciliation + render checks only)")
        for p in problems:
            print(f"[FAIL] {p}")
        if not problems:
            print("[PASS] cohort-year table is current")
        return 0 if not problems else 1
    art = build()
    CELLS.write_text(json.dumps(art, indent=2) + "\n", encoding="utf-8")
    TEX.write_text(render_tex(art), encoding="utf-8")
    print(f"wrote {CELLS}\nwrote {TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
