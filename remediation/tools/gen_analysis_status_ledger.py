"""Generate the machine-readable status ledger for every analysis involving 2012 (E-022).

Correction-release schema — one row per analysis, with the REQUIRED columns

    analysis_id, description, status, first_run_commit_or_time, run_id,
    uses_2012_labels, changes_governed_output, reported_location

and status drawn ONLY from the four controlled values

    protocol-primary        declared in the frozen protocol; the primary chain
    protocol-secondary      declared in the frozen protocol/amendment BEFORE the
                            governed execution; secondary or diagnostic role
    exploratory-final-only  absent from development; evaluated only in the final run
    post-hoc                specified after the frozen primary evaluation (separately
                            versioned; frozen governed outputs never modified)

Run/commit identities are pulled from the artifacts themselves, never from memory.
Emits deterministically (no timestamps):
  * ``paper/ANALYSIS_STATUS_LEDGER.csv``
  * ``paper/latex/analysis_status_table.tex``  (the compact manuscript appendix table)

Usage: python tools/gen_analysis_status_ledger.py [--check]
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

REM = Path(__file__).resolve().parents[1]
E = REM / "experiment"
OUT_CSV = REM / "paper" / "ANALYSIS_STATUS_LEDGER.csv"
OUT_TEX = REM / "paper" / "latex" / "analysis_status_table.tex"

COLUMNS = ["analysis_id", "description", "status", "first_run_commit_or_time", "run_id",
           "uses_2012_labels", "changes_governed_output", "reported_location"]

ALLOWED_STATUS = {"protocol-primary", "protocol-secondary", "exploratory-final-only",
                  "post-hoc"}


def j(p: Path) -> dict:
    return json.loads(p.read_text("utf-8"))


def build_rows() -> list[dict]:
    fr = j(E / "final_results.json")
    br = j(E / "broad_sensitivity" / "final_results.json")
    lm = j(E / "lowmiss_sensitivity" / "final_results.json")
    lf = j(E / "leakage_factorial.json")
    tr = j(E / "target_reporting_process_audit.json")
    rv = j(E / "raw_vs_calibrated_decisions.json")
    smu = j(E / "severe_metric_uncertainty.json")
    cbs = j(E / "cluster_bootstrap_sensitivity.json")
    ag = j(E / "agency_leave_one_out.json")
    rt = j(E / "prediction_roundtrip_test.json")

    run_commit = fr["git"]["commit"][:12]
    RUN = fr["run_id"]  # final_8af9d5bc23d8

    def row(aid, desc, status, first, run_id, uses, changes, loc):
        assert status in ALLOWED_STATUS, status
        return dict(zip(COLUMNS, [aid, desc, status, first, run_id, uses, changes, loc]))

    rows = [
        row("A01-development-selection",
            "Rolling-origin development model selection and role designation on 2009-2011 "
            "(poison-token sentinel enforces that no 2012 outcome is readable)",
            "protocol-primary",
            f"git {run_commit} (dev report pinned by FROZEN.lock)",
            "none (development phase)", "no",
            "no (produces the frozen roles)", "Sec.7; Sec.8.2"),
        row("A02-primary-benchmark",
            "Primary governed benchmark: restricted tier, posterior-median rule, "
            "protocol-designated weighted candidate vs majority baseline — the one frozen "
            "primary evaluation execution of the 2012 cohort",
            "protocol-primary",
            f"git {run_commit}", RUN, "yes",
            "creates them (the frozen governed outputs)",
            "Sec.8.3; Appendix A"),
        row("A03-secondary-model-comparisons",
            "All 13 secondary paired model-vs-majority contrasts inside the same frozen "
            "execution (declared secondary; only H1 is the headline comparison)",
            "protocol-secondary",
            f"git {run_commit}", RUN, "yes", "no", "Sec.8.3; Appendix A"),
        row("A04-probability-and-calibration",
            "Proper scores (log loss, Brier, unnormalized RPS), top-label ECE, and "
            "development-only calibration applied to 2012 for probability quality only",
            "protocol-secondary",
            f"git {run_commit}", RUN, "yes", "no", "Sec.8.4"),
        row("A05-decision-rule-sensitivity",
            "Posterior-median vs argmax hard decisions; deterministic recomputation from "
            "the frozen run's stored predictions (no refit); argmax declared a sensitivity "
            "in the frozen protocol",
            "protocol-secondary",
            f"git {run_commit} (derived from the frozen bundle)",
            RUN, "yes", "no", "Sec.6.3; Sec.8.6"),
        row("A06-error-analysis",
            "Confusion structure, severe misses, false alarms, confidence bins; "
            "deterministic recomputation from the frozen run's stored predictions",
            "protocol-secondary",
            f"git {run_commit} (derived from the frozen bundle)",
            RUN, "yes", "no", "Sec.9"),
        row("A07-severe-ranking",
            "Severe one-vs-rest AP/AUROC ranking from the frozen probabilities (declared "
            "in the frozen metric suite; no threshold selected)",
            "protocol-secondary",
            f"git {run_commit} (derived from the frozen bundle)",
            RUN, "yes", "no", "Sec.9"),
        row("A08-broad-tier-sensitivity",
            "Broad feature-tier sensitivity (hypothesis H4), prespecified in the protocol "
            "amendment before the governed v4 execution; separately versioned frozen run",
            "protocol-secondary",
            f"git {br['git']['commit'][:12]}", br["run_id"], "yes", "no",
            "Sec.8.7-8.8"),
        row("A09-lowmiss-sensitivity",
            "High-missingness field-removal sensitivity (>50% development-missingness "
            "rule); specified AFTER the primary evaluation (v4.1 round) and executed as a "
            "separately versioned frozen run",
            "post-hoc",
            f"git {lm['git']['commit'][:12]}", lm["run_id"], "yes", "no", "Sec.8.8"),
        row("A10-seed-robustness",
            "Five-seed refit robustness of the four forest variants (v4.1 retrospective "
            "addendum; refits, separately versioned)",
            "post-hoc",
            "v4.1 finalization round (experiment/seed_robustness.md)",
            "seed refits (no governed run id)", "yes", "no", "Sec.8.8"),
        row("A11-blank-excluded-target-sensitivity",
            "Blank-excluded target mapping consequence check (shows the alternative "
            "mapping is not a comparable three-class problem)",
            "post-hoc",
            "v4.1 finalization round (experiment/target_sensitivity.md)",
            "sensitivity refit (no governed run id)", "yes", "no", "Sec.8.8"),
        row("A12-ebm-exploratory",
            "Explainable Boosting Machine, absent from development by design; evaluated "
            "only inside the final frozen execution; never competes for the primary role",
            "exploratory-final-only",
            f"git {run_commit}", RUN, "yes", "no", "Sec.6.2; Sec.8.3; Appendix A"),
        row("A13-leakage-factorial",
            "Matched 2x2x2 protocol-defect diagnostic, development years only "
            "(2009-2011); declared diagnostic; conditional contrasts, not causal effects",
            "protocol-secondary",
            f"artifact generated {lf['generated_utc'][:10]} (v4 finalization)",
            "development-only diagnostic (no 2012 run)", "no", "no",
            "Sec.8.5"),
        row("A14-target-reporting-audit",
            "Target and reporting-process audit: class-0 provenance, count-field zero "
            "signatures, quarantine comparison, person-level co-missingness, and the "
            "missingness-indicator-only probe",
            "post-hoc",
            f"git {tr['git_commit'][:12]}",
            "post-hoc audit (missingness probe: deid parquet pinned)", "yes", "no",
            "Sec.4.2; Sec.8.8; Sec.11(5)"),
        row("A15-calibrated-label-counterfactual",
            "Verification that headline hard labels reconstruct from raw probabilities "
            "(162,820 labels, zero mismatches) plus the unused development-only-calibrated "
            "label counterfactual (1,894 labels change; oMAE 0.3442); the frozen headline "
            "is not replaced",
            "post-hoc",
            f"git {rv['git_commit'][:12]}", rv["source_run"], "yes",
            "no (counterfactual only; frozen labels unchanged)", "Sec.6.3; Sec.8.4"),
        row("A16-severe-metric-uncertainty",
            "Descriptive case-bootstrap intervals for the primary model's severe recall, "
            "precision, AP, AUROC (450 severe positives; crash-level unit)",
            "post-hoc",
            f"git {smu['git_commit'][:12]}", smu["source_run"], "yes", "no", "Sec.9"),
        row("A17-proxy-cluster-dependence",
            "Calendar-day (366), calendar-week (53), region (4), borough (18), and "
            "maintenance-responsibility (9) proxy-cluster resampling and leave-one-out "
            "re-evaluation of the primary contrast",
            "post-hoc",
            f"git {cbs['git_commit'][:12]}", cbs["source_run"], "yes", "no",
            "Sec.8.8; Sec.11(7)"),
        row("A18-agency-cluster-deferred",
            "Agency-cluster dependence analysis: DEFERRED — agency fields exist in the "
            "licensed raw extract but were removed as identifier-tier columns at "
            "de-identification; no row-level agency label is joinable without licensed "
            "re-supply",
            "post-hoc",
            f"git {ag['git_commit'][:12]}",
            "deferred (feasibility record only; no analysis executed)",
            "no (no analysis executed)", "no", "Sec.8.8; Sec.11(7)"),
        row("A19-prediction-roundtrip",
            "Lossless round-trip verification of the de-identified prediction evidence "
            "(packaging verification)",
            "post-hoc",
            f"git {rt['git_commit'][:12]}", rt["source_run"], "yes", "no", "Sec.12.1"),
        row("A20-iteration3-reanalysis",
            "Re-analysis of the four Iteration III published confusion matrices "
            "(historical; the prior random 80/20 split drew from 2009-2012)",
            "post-hoc",
            "transcribed source matrices (N=10,936)",
            "none (recomputation of published matrices)", "yes", "no",
            "Sec.8.1; Appendix B"),
        row("A21-v3-comparison",
            "Comparison against the historical v3 frozen runs (superseded protocol; "
            "preserved un-overwritten); assembled for the protocol-correction narrative",
            "post-hoc",
            "v3 runs preserved at their original commits",
            "final_f27613102c96; final_17f99a4c8f94", "yes", "no", "Sec.8.7"),
        row("A22-feature-timing-ledger",
            "Per-field timing-evidence ledger: author-judged grades; 0 of 100 fields "
            "custodian-verified (documentation)",
            "post-hoc",
            "v4.1 revision addendum (paper/FEATURE_TIMING_EVIDENCE.md)",
            "none (documentation)", "no", "no", "Sec.4.3; Sec.11(4)"),
        row("A23-hyperparameter-ledger",
            "Fixed model configurations extracted from registry code and the frozen "
            "configuration (documentation; no search was run)",
            "post-hoc",
            f"config {fr['config_sha256'][:12]}",
            "none (documentation)", "no", "no", "Appendix D"),
        row("A24-metric-conventions",
            "Machine-readable metric-convention sidecar: unnormalized RPS, log-loss "
            "clipping epsilon, ECE binning, F1 zero-division reporting convention, "
            "paired-bootstrap mechanics, Frank-Hall probability construction "
            "(documentation; extracted from code)",
            "post-hoc",
            "correction release + r3 pass (experiment/metric_conventions.json)",
            "none (documentation)", "no", "no", "Sec.6.2; Sec.6.4; Sec.6.5; Sec.8.4"),
        row("A25-complete-result-tables",
            "Complete Appendix A and development-results tables generated from the frozen "
            "artifacts under the stated count-based F1 reporting convention (display of "
            "stored values; recomputed from stored confusion matrices and asserted against "
            "the stored fields; no refit, no new interrogation)",
            "post-hoc",
            "r3 submission pass (tools/gen_full_results_tables.py; "
            "tools/gen_dev_results_table.py)",
            RUN, "yes", "no", "Sec.6.4; Appendix A; Appendix F"),
        row("A26-severe-reliability-diagram",
            "Severe-class reliability diagram: calibrated points from the frozen run "
            "artifact; raw curve recomputed from the released frozen probabilities under "
            "the same 10-bin rule (presentation-level derivation; no refit)",
            "post-hoc",
            "r3 submission pass (figure_generation/make_figures.py::fig06c_reliability)",
            RUN, "yes", "no", "Sec.8.4"),
    ]
    return rows


def render(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("#", r"\#").replace("_", r"\_").replace("$", r"\$")
            .replace(">", r"$>$").replace("<", r"$<$"))


def render_tex(rows: list[dict]) -> str:
    """Compact manuscript appendix table (generated; never hand-edited). Rendered as two
    single-page [H] parts: 26 rows no longer fit one page, and a page-split longtable
    trips a pdfTeX 'infinite glue shrinkage' warning under this preamble."""
    short_run = {
        "none (development phase)": "---",
        "development-only diagnostic (no 2012 run)": "---",
        "none (recomputation of published matrices)": "---",
        "none (documentation)": "---",
        "seed refits (no governed run id)": "refits",
        "sensitivity refit (no governed run id)": "refit",
        "deferred (feasibility record only; no analysis executed)": "deferred",
        "final_f27613102c96; final_17f99a4c8f94": "v3 frozen runs",
        "post-hoc audit (missingness probe: deid parquet pinned)": "audit",
    }

    def data_line(r):
        name = _tex_escape(r["analysis_id"].split("-", 1)[1].replace("-", " "))
        status = _tex_escape(r["status"])
        uses = _tex_escape(r["uses_2012_labels"])
        run = r["run_id"]
        run = short_run.get(run, run)
        if run.startswith("final_"):
            # \artifact uses \path, which takes raw underscores verbatim
            run = r"\artifact{" + run + "}"
        else:
            run = _tex_escape(run)
        if r["changes_governed_output"].startswith("creates"):
            status = status + r" (creates)"
        loc = _tex_escape(r["reported_location"]).replace("Sec.", r"\S")
        return f"{name} & {status} & {uses} & {run} & {loc} \\\\"

    half = (len(rows) + 1) // 2
    parts = [rows[:half], rows[half:]]
    lines = [
        "% GENERATED by tools/gen_analysis_status_ledger.py -- do not edit by hand.",
        "% Full detail (descriptions, commit/run identities): paper/ANALYSIS_STATUS_LEDGER.csv",
        "% (two non-splitting single-page parts: 26 rows exceed one page, and a page-split",
        "%  longtable trips a pdfTeX 'infinite glue shrinkage' warning under this preamble)",
    ]
    note = (r"\tabnote{Status values: \emph{protocol-primary} and \emph{protocol-secondary} "
            r"were declared in the frozen protocol or its committed pre-execution amendment; "
            r"\emph{exploratory-final-only} was absent from development by design; "
            r"\emph{post-hoc} was specified after the frozen primary evaluation. "
            r"``creates'' marks the one execution that produced the frozen governed outputs; "
            r"every other analysis changed no governed output. Run --- = no separate 2012 "
            r"evaluation execution.}")
    for i, part in enumerate(parts, 1):
        lines += [
            r"\begin{table}[H]",
            r"\centering",
        ]
        if i == 1:
            lines.append(
                r"\caption[Status ledger of the 2012-cohort analyses]{Status ledger of every "
                r"analysis involving the 2012 cohort, part 1 of 2 (generated from the "
                r"machine-readable ledger \texttt{paper/ANALYSIS\_STATUS\_LEDGER.csv}).}"
                r"\label{tab:analysis-status}")
        else:
            lines.append(
                # \ContinuedFloat (caption pkg): part 2 keeps part 1's table number and,
                # with list=no, adds no List-of-Tables entry, so the LoT stays contiguous.
                r"\ContinuedFloat"
                "\n"
                r"\captionsetup{list=no}\caption{Status ledger of the 2012-cohort analyses, "
                r"part 2 of 2 (continued).}\label{tab:analysis-status-b}")
        lines += [
            r"\footnotesize",
            r"\renewcommand{\arraystretch}{1.06}",
            r"\begin{tabular}{@{}L{0.21\textwidth}L{0.17\textwidth}C{0.07\textwidth}L{0.16\textwidth}L{0.22\textwidth}@{}}",
            r"\toprule",
            r"\textbf{Analysis} & \textbf{Status} & \textbf{2012 labels} & "
            r"\textbf{Run} & \textbf{Reported} \\",
            r"\midrule",
        ]
        lines += [data_line(r) for r in part]
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
        ]
        if i == 2:
            lines.append(note)
        lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = build_rows()
    csv_payload = render(rows)
    tex_payload = render_tex(rows)
    if "--check" in sys.argv[1:]:
        ok = True
        if not OUT_CSV.exists() or OUT_CSV.read_text("utf-8") != csv_payload:
            print("[FAIL] ANALYSIS_STATUS_LEDGER.csv stale or missing; regenerate")
            ok = False
        if not OUT_TEX.exists() or OUT_TEX.read_text("utf-8") != tex_payload:
            print("[FAIL] analysis_status_table.tex stale or missing; regenerate")
            ok = False
        if ok:
            print("[PASS] ANALYSIS_STATUS_LEDGER.csv and analysis_status_table.tex are current")
        return 0 if ok else 1
    OUT_CSV.write_text(csv_payload, encoding="utf-8")
    OUT_TEX.write_text(tex_payload, encoding="utf-8")
    print(f"wrote {OUT_CSV} ({len(rows)} rows) and {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
