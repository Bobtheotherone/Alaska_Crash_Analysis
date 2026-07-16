"""Generate the feature-governance appendix tables (r3 pass, L-08 / Phase 5A-B).

Makes the restricted feature tier inspectable inside the manuscript:

  * retained-49 table   — every strict-tier input field with model representation,
    per-development-year missingness (2009/2010/2011), reporting-thoroughness risk,
    and the contract-ledger rationale;
  * prohibited-14 table — every OUTCOME_DERIVED field with its exclusion category,
    plausible availability stage, and target-relationship / leakage rationale;
  * exclusion summary   — every remaining category (identifiers, constants,
    timing-uncertain broad-only, outcome-proximal broad-only, split axis, target)
    with counts and disposition.

Sources: the authoritative contract ledger (``data/feature_availability_ledger.csv``),
the frozen run's retained-column list (``experiment/final_results.json``), the status
taxonomy of ``crashsev/feature_timing_evidence.py`` (imported, not duplicated), and the
licensed local modeling table for missingness. Emits

  * ``experiment/feature_governance_cells.json``   (committed, deterministic)
  * ``paper/latex/feature_governance_tables.tex``  (generated LaTeX, ``\\input`` by the
    manuscript, never edited by hand)

``--check`` is data-free: it re-renders the LaTeX from the COMMITTED cells JSON and
byte-compares, and structurally reconciles the JSON against the frozen retained list
(the licensed table is absent from clean-room checkouts by policy). When the local
table IS present, ``--check`` additionally recomputes the missingness numbers.

Usage: python tools/gen_feature_governance_tables.py [--check]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REM))

CELLS = REM / "experiment" / "feature_governance_cells.json"
TEX = REM / "paper" / "latex" / "feature_governance_tables.tex"
LOCAL = REM / "_local_data" / "modeling_table_09_12.csv"
DEV_YEARS = (2009, 2010, 2011)

LATEX_ESCAPES = {"&": "\\&", "%": "\\%", "#": "\\#", "_": "\\_", "$": "\\$"}


def esc(s: str) -> str:
    out = str(s)
    for k, v in LATEX_ESCAPES.items():
        out = out.replace(k, v)
    return out


def build() -> dict:
    import pandas as pd
    from crashsev import cli as CLI
    from crashsev import feature_timing_evidence as FTE
    from crashsev.missingness_summary import _missing_mask

    cfg = CLI.load_config(str(REM / "configs" / "route_r_09_12.yml"))
    frozen = json.loads((REM / "experiment" / "final_results.json").read_text("utf-8"))
    retained = set(frozen["allowed_feature_columns"])
    dropped_invariant = set(frozen["dropped_invariant_features"])
    led = pd.read_csv(REM / "data" / "feature_availability_ledger.csv")
    strict = set(led.loc[led["strict_scene_tier"] == True, "column"])  # noqa: E712
    broad = set(led.loc[led["allowed_post_crash_triage"] == True, "column"])  # noqa: E712

    prep = CLI.prepare_development(str(LOCAL), cfg)
    Xd, yd = prep.X_dev, prep.y_dev.to_numpy()
    years = pd.Series(prep.years_dev).astype(int).to_numpy()
    tokens = prep.schema.string_missing_tokens

    miss_year, contrast = {}, {}
    for c in Xd.columns:
        m = _missing_mask(Xd[c], tokens).to_numpy()
        miss_year[c] = {str(y): float(m[years == y].mean()) for y in DEV_YEARS}
        contrast[c] = abs(float(m[yd == 0].mean()) - float(m[yd == 2].mean()))

    retained_rows, prohibited_rows, other = [], [], {}
    for _, r in led.sort_values("column").iterrows():
        col = r["column"]
        status, stage = FTE._status(r, retained, strict, dropped_invariant, cfg["year_col"])
        rationale = str(r.get("rationale", "")).strip()
        if col in retained:
            assert status == "PLAUSIBLY_EARLY", (col, status)
            retained_rows.append({
                "field": col,
                "kind": "numeric" if col in frozen["numeric_features"] else "categorical",
                "missing": miss_year[col],
                "thoroughness": ("elevated" if contrast[col] >= FTE.CONTRAST_FLAG else "low"),
                "rationale": rationale,
            })
        elif status == "OUTCOME_DERIVED":
            prohibited_rows.append({
                "field": col,
                "category": "Outcome-derived",
                "stage": "recorded with or after the severity outcome (post-outcome "
                         "response, count, damage, or enforcement)",
                "rationale": rationale,
            })
        else:
            entry = other.setdefault(status, {"count": 0, "fields": []})
            entry["count"] += 1
            entry["fields"].append(col)

    assert len(retained_rows) == 49, f"expected 49 retained fields, got {len(retained_rows)}"
    assert len(prohibited_rows) == 14, f"expected 14 prohibited fields, got {len(prohibited_rows)}"
    assert {r["field"] for r in retained_rows} == retained

    dispositions = {
        "TIMING_UNCERTAIN": "excluded from the primary restricted tier; admitted only to "
                            "the broad-tier sensitivity (H4)",
        "OUTCOME_PROXIMAL": "damage/response-adjacent (LEDGER-TENSION-001); broad-tier "
                            "sensitivity only",
        "EXCLUDED_IDENTIFIER": "identifier / privacy exclusion; removed at de-identification, "
                               "never modeled",
        "EXCLUDED_CONSTANT": "invariant in the extract or on development rows; no information",
        "SPLIT_ONLY": "chronological split axis; withheld as a feature",
        "TARGET (excluded)": "the outcome itself",
    }
    summary = [{"status": s, "count": other[s]["count"],
                "disposition": dispositions[s], "fields": sorted(other[s]["fields"])}
               for s in ("TIMING_UNCERTAIN", "OUTCOME_PROXIMAL", "EXCLUDED_IDENTIFIER",
                         "EXCLUDED_CONSTANT", "SPLIT_ONLY", "TARGET (excluded)")]

    return {
        "description": (
            "Structured cells for the feature-governance appendix tables "
            "(tab:retained-fields / tab:prohibited-fields / tab:exclusion-summary). "
            "Retained set = the frozen run's 49 allowed columns (all PLAUSIBLY_EARLY, "
            "author-judged; none custodian-verified). Missingness = per-development-year "
            "share of missing/sentinel values on mapped development rows. Thoroughness "
            "risk = elevated when |missing(class 0) - missing(class 2)| >= 0.05 on "
            "development rows. Prohibited = the 14 OUTCOME_DERIVED contract-ledger fields."
        ),
        "sources": ["data/feature_availability_ledger.csv",
                    "experiment/final_results.json (allowed_feature_columns)",
                    "licensed local modeling table (canonical content hash pinned in the "
                    "evidence chain) for missingness"],
        "dev_years": list(DEV_YEARS),
        "retained": retained_rows,
        "prohibited": prohibited_rows,
        "exclusion_summary": summary,
    }


def render_tex(art: dict) -> str:
    """Two single-page [H] parts for the 49-row retained table (a page-split longtable
    trips a pdfTeX 'infinite glue shrinkage' warning under this preamble), then the
    prohibited-14 and exclusion-summary tables."""
    y = [str(v) for v in art["dev_years"]]
    # breakable spaces: these header cells must WRAP inside their narrow columns
    header = (f"\\textbf{{Source field}} & \\textbf{{Type}} & \\textbf{{Miss.\\ {y[0]}}} & "
              f"\\textbf{{Miss.\\ {y[1]}}} & \\textbf{{Miss.\\ {y[2]}}} & "
              "\\textbf{Thoro.\\ risk} & \\textbf{Contract-ledger rationale} \\\\")
    note = ("\\tabnote{Type: numeric fields are median-imputed; categorical fields are "
            "most-frequent-imputed and one-hot encoded with an infrequent bucket. Miss.\\ = "
            "share of missing or sentinel values on mapped development rows of that year. "
            "Thoro.\\ risk = \\emph{elevated} when development missingness differs by at "
            "least 0.05 between classes 0 and 2, so the field's presence is "
            "outcome-correlated (\\artifact{experiment/target_reporting_process_audit.json}, "
            "q8). Every field is used directly as a model input after imputation and "
            "encoding; the year is only the split axis and no retained field is transformed "
            "beyond imputation, encoding, and (for linear models) standardization.}")
    rows = art["retained"]
    half = (len(rows) + 1) // 2
    lines = [
        "% GENERATED FILE — do not edit by hand.",
        "% Source: tools/gen_feature_governance_tables.py (contract ledger + frozen",
        "% retained-column list + licensed local modeling-table missingness).",
        "% Two single-page [H] parts: a page-split longtable trips a pdfTeX",
        "% 'infinite glue shrinkage' warning under this preamble.",
    ]
    for i, part in enumerate((rows[:half], rows[half:]), 1):
        lines += ["\\begin{table}[H]", "\\centering"]
        if i == 1:
            lines.append(
                "\\caption[The 49 retained restricted-tier source fields]{The 49 retained "
                "restricted-tier source fields, part 1 of 2 (all author-judged "
                "\\emph{plausibly early}; none custodian-verified).}"
                "\\label{tab:retained-fields}")
        else:
            lines.append(
                # \ContinuedFloat (caption pkg): part 2 keeps part 1's table number and,
                # with list=no, adds no List-of-Tables entry, so the LoT stays contiguous.
                "\\ContinuedFloat\n"
                "\\captionsetup{list=no}\\caption{The 49 retained restricted-tier source "
                "fields, part 2 of 2 (continued).}\\label{tab:retained-fields-b}")
        lines += [
            "\\footnotesize",
            "\\begingroup\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{@{}L{0.235\\textwidth}L{0.115\\textwidth}R{0.055\\textwidth}"
            "R{0.055\\textwidth}R{0.055\\textwidth}L{0.08\\textwidth}L{0.295\\textwidth}@{}}",
            "\\toprule", header, "\\midrule",
        ]
        for r in part:
            m = r["missing"]
            lines.append(" & ".join([
                esc(r["field"]), r["kind"],
                f"{m[y[0]]:.3f}", f"{m[y[1]]:.3f}", f"{m[y[2]]:.3f}",
                r["thoroughness"], esc(r["rationale"]),
            ]) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\endgroup"]
        if i == 2:
            lines.append(note)
        lines.append("\\end{table}")
    lines.append("")
    # ---- prohibited 14 (single page, non-splitting) ----
    lines += [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{The 14 prohibited outcome-derived source fields (excluded from every "
        "modeling tier).}",
        "\\label{tab:prohibited-fields}",
        "\\footnotesize",
        "\\begin{tabular}{@{}L{0.30\\textwidth}L{0.13\\textwidth}L{0.50\\textwidth}@{}}",
        "\\toprule",
        "\\textbf{Source field} & \\textbf{Category} & \\textbf{Relationship to the target / "
        "leakage rationale} \\\\",
        "\\midrule",
    ]
    for r in art["prohibited"]:
        lines.append(" & ".join([esc(r["field"]), r["category"], esc(r["rationale"])]) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\tabnote{Every field in this table is recorded with or after the severity outcome "
        "(post-crash response, injury/fatality counting, damage assessment, or enforcement), "
        "so admitting it would let the predictors carry the outcome itself or its direct "
        "consequences.}",
        "\\end{table}",
        "",
    ]
    # ---- exclusion summary ----
    lines += [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Disposition of the remaining source fields (neither retained nor "
        "outcome-derived).}",
        "\\label{tab:exclusion-summary}",
        "\\footnotesize",
        "\\begin{tabularx}{\\textwidth}{@{}L{0.22\\textwidth}R{0.07\\textwidth}Y@{}}",
        "\\toprule",
        "\\textbf{Category} & \\textbf{Fields} & \\textbf{Disposition} \\\\",
        "\\midrule",
    ]
    for s in art["exclusion_summary"]:
        label = {"TIMING_UNCERTAIN": "Timing-uncertain", "OUTCOME_PROXIMAL": "Outcome-proximal",
                 "EXCLUDED_IDENTIFIER": "Identifier", "EXCLUDED_CONSTANT": "Constant/invariant",
                 "SPLIT_ONLY": "Split axis (Year)", "TARGET (excluded)": "Target field"}[s["status"]]
        lines.append(f"{label} & {s['count']} & {esc(s['disposition'])} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\tabnote{The per-field ledger \\artifact{paper/FEATURE_TIMING_EVIDENCE.md} names "
        "every field in each category with its grade and rationale; the machine-readable "
        "source is \\artifact{data/feature_availability_ledger.csv}.}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _structural_check(art: dict) -> list:
    problems = []
    frozen = json.loads((REM / "experiment" / "final_results.json").read_text("utf-8"))
    if [r["field"] for r in art["retained"]] != sorted(frozen["allowed_feature_columns"]):
        problems.append("retained rows != frozen allowed_feature_columns")
    if len(art["prohibited"]) != 14:
        problems.append("prohibited rows != 14")
    kinds = {r["field"]: r["kind"] for r in art["retained"]}
    for c in frozen["numeric_features"]:
        if kinds.get(c) != "numeric":
            problems.append(f"{c} should be numeric")
    for c in frozen["categorical_features"]:
        if kinds.get(c) != "categorical":
            problems.append(f"{c} should be categorical")
    return problems


def main() -> int:
    check = "--check" in sys.argv[1:]
    if check:
        if not CELLS.exists() or not TEX.exists():
            print("[FAIL] feature-governance artifacts missing; regenerate")
            return 1
        art = json.loads(CELLS.read_text("utf-8"))
        problems = _structural_check(art)
        if TEX.read_text("utf-8") != render_tex(art):
            problems.append("feature_governance_tables.tex stale w.r.t. committed cells JSON")
        if LOCAL.exists():
            regen = build()
            if json.dumps(regen, indent=2) + "\n" != CELLS.read_text("utf-8"):
                problems.append("cells JSON stale w.r.t. full regeneration from local data")
        else:
            print("[SKIP] licensed local table absent: missingness recomputation skipped "
                  "(structural + render checks only)")
        for p in problems:
            print(f"[FAIL] {p}")
        if not problems:
            print("[PASS] feature-governance tables are current")
        return 0 if not problems else 1
    art = build()
    CELLS.write_text(json.dumps(art, indent=2) + "\n", encoding="utf-8")
    TEX.write_text(render_tex(art), encoding="utf-8")
    print(f"wrote {CELLS}\nwrote {TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
