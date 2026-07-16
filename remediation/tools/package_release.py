"""Assemble the final-submission release: final PDF, source archive, evidence copies,
ledgers, contact sheet, manifests, SHA256SUMS, and a machine test report.

Run AFTER the final repo commit. The packager re-runs every verification gate itself and
refuses to package on any failure:

    python tools/package_release.py <release_dir>

Gates executed (all must pass): pytest suite; verify_manuscript_numbers.py (tex + PDF);
verify_frozen_and_locked.py; audit_paper.py; generator --check freshness for the three
generated ledgers/artifacts.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REM = Path(__file__).resolve().parents[1]
ACA = REM.parent
SRC = REM / "paper" / "latex"
if len(sys.argv) < 2:
    raise SystemExit("usage: python tools/package_release.py <release_dir>")
REL = Path(sys.argv[1])
REL.mkdir(parents=True, exist_ok=True)

PDF_NAME = "Mercado-Barbosa_UAA_Student_Paper_Final_Submission.pdf"
ZIP_NAME = "Mercado-Barbosa_UAA_Student_Paper_Final_Submission_Source.zip"
BASELINE_COMMIT = "42194a619c3a8c1a146c1202ac6afe6a919e392b"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(ACA), *args],
                          capture_output=True, text=True).stdout.strip()


# 0. verification gates --------------------------------------------------------
def run_gate(name: str, cmd: list[str], cwd: Path = REM) -> str:
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).splitlines()[-3:])
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"[package] GATE FAILED: {name}")
    print(f"[gate] {name}: OK :: {tail.splitlines()[-1] if tail else ''}")
    return r.stdout


report_lines = [f"# Machine test report — r4 final-submission release",
                f"generated_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
                f"repo_commit: {git('rev-parse', 'HEAD')}", ""]
py = sys.executable
_basetemp = Path("C:/tmp/aca-pkg") if Path("C:/").exists() else Path.home() / ".aca-pkg-tmp"
_basetemp.mkdir(parents=True, exist_ok=True)
gates = [
    ("pytest", [py, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
                "--basetemp", str(_basetemp)]),
    ("gen_reanalysis_tables --check", [py, "tools/gen_reanalysis_tables.py", "--check"]),
    ("gen_hyperparameter_ledger --check", [py, "tools/gen_hyperparameter_ledger.py", "--check"]),
    ("gen_analysis_status_ledger --check", [py, "tools/gen_analysis_status_ledger.py", "--check"]),
    ("gen_metric_conventions --check", [py, "tools/gen_metric_conventions.py", "--check"]),
    ("gen_full_results_tables --check", [py, "tools/gen_full_results_tables.py", "--check"]),
    ("gen_dev_results_table --check", [py, "tools/gen_dev_results_table.py", "--check"]),
    ("gen_feature_governance_tables --check", [py, "tools/gen_feature_governance_tables.py", "--check"]),
    ("gen_cohort_year_table --check", [py, "tools/gen_cohort_year_table.py", "--check"]),
    ("verify_manuscript_numbers", [py, "tools/verify_manuscript_numbers.py",
                                   str(SRC), str(SRC / "main.pdf")]),
    ("verify_frozen_and_locked", [py, "tools/verify_frozen_and_locked.py"]),
    ("audit_paper", [py, "tools/audit_paper.py", "--render", str(SRC / "page_render")]),
]
gate_outputs = {"generated_utc": report_lines[1].split(": ", 1)[1],
                "repo_commit": git("rev-parse", "HEAD"), "gates": {}}
for name, cmd in gates:
    out = run_gate(name, cmd)
    summary = []
    for ln in out.splitlines():
        if ln.startswith("[forbidden-list] "):
            gate_outputs["forbidden_list"] = json.loads(ln[len("[forbidden-list] "):])
            continue
        if (("passed" in ln and "failed" in ln) or "SUMMARY" in ln
                or ln.strip().endswith("passed") or ln.startswith("[SKIP]")
                or re.match(r"^\d+ passed\b", ln.strip())):
            summary.append(ln.strip())
            if not ln.startswith("[SKIP]"):
                report_lines.append(f"{name}: {ln.strip()}")
    gate_outputs["gates"][name] = {"cmd": " ".join(cmd[1:] if cmd[0] == py else cmd),
                                   "exit": 0, "summary_lines": summary,
                                   "tail": out.splitlines()[-3:]}
report_lines.append("")
(REL / "test_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
(REL / "gate_outputs.json").write_text(json.dumps(gate_outputs, indent=2) + "\n",
                                       encoding="utf-8")

# 1. final PDF ---------------------------------------------------------------
pdf = REL / PDF_NAME
pdf.write_bytes((SRC / "main.pdf").read_bytes())
from pypdf import PdfReader
n_pages = len(PdfReader(str(pdf)).pages)

# 2. source archive (source of the shipped PDF; aux/derived files excluded) ---
zip_path = REL / ZIP_NAME
include = ["main.tex", "analysis_status_table.tex", "full_results_tables.tex",
           "dev_results_table.tex", "feature_governance_tables.tex",
           "cohort_year_table.tex", "references.bib", "main.bbl",
           "build.sh", "README.md",
           "FIGURE_REPLACEMENT_GUIDE.md", "figures/user/README.md",
           "figure_generation/make_figures.py"] + \
          [f"figures/user/{p.name}" for p in sorted((SRC / "figures/user").glob("*.pdf"))]
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in include:
        z.write(SRC / rel, arcname=f"Mercado-Barbosa_UAA_Student_Paper_Source/{rel}")

# 3. de-identified evidence copies -------------------------------------------
ev = REL / "evidence"
ev.mkdir(exist_ok=True)
for name in ("predictions_lossless.parquet", "missingness_only_predictions.parquet"):
    (ev / name).write_bytes((REM / "experiment" / name).read_bytes())

# 3b. ledger + governance-document copies from the committed tree --------------
for rel_src, rel_dst in [
    ("paper/CORRECTION_LEDGER.json", "CORRECTION_LEDGER.json"),
    ("paper/ANALYSIS_STATUS_LEDGER.csv", "ANALYSIS_STATUS_LEDGER.csv"),
    ("paper/audit_resolution_matrix.csv", "audit_resolution_matrix.csv"),
    ("paper/custodian_semantics_request.md", "custodian_semantics_request.md"),
    ("paper/unresolved_external_evidence.md", "unresolved_external_evidence.md"),
    ("paper/CLAIM_EVIDENCE_MATRIX.md", "CLAIM_EVIDENCE_MATRIX.md"),
    ("paper/REVISION_SUMMARY.md", "REVISION_SUMMARY.md"),
    ("experiment/metric_conventions.json", "metric_conventions.json"),
    # r3 submission-readiness deliverables
    ("paper/DISCREPANCY_LEDGER_R3.md", "DISCREPANCY_LEDGER_R3.md"),
    ("paper/REVISION_MEMO_R3.md", "REVISION_MEMO_R3.md"),
    ("paper/AUTHOR_ACTIONS_R3.md", "AUTHOR_ACTIONS_R3.md"),
    ("paper/ADMISSIONS_REVIEW_CHECKLIST_R3.md", "ADMISSIONS_REVIEW_CHECKLIST_R3.md"),
    ("paper/ORAL_DEFENSE_SHEET.md", "ORAL_DEFENSE_SHEET.md"),
    ("paper/NUMERICAL_VERIFICATION_R3.md", "NUMERICAL_VERIFICATION_R3.md"),
    # r4 final-submission deliverables
    ("paper/REVISION_MEMO_R4.md", "REVISION_MEMO_R4.md"),
    ("paper/AUTHOR_ACTIONS_R4.md", "AUTHOR_ACTIONS_R4.md"),
    ("paper/ADMISSIONS_REVIEW_CHECKLIST_R4.md", "ADMISSIONS_REVIEW_CHECKLIST_R4.md"),
    ("paper/NUMERICAL_VERIFICATION_R4.md", "NUMERICAL_VERIFICATION_R4.md"),
    ("paper/PUBLIC_REPOSITORY_AUDIT_R4.md", "PUBLIC_REPOSITORY_AUDIT_R4.md"),
    ("paper/PRIVACY_AND_NDA_ATTESTATION_R4.md", "PRIVACY_AND_NDA_ATTESTATION_R4.md"),
    ("paper/GITHUB_PUBLICATION_R4.md", "GITHUB_PUBLICATION_R4.md"),
    ("paper/RELEASE_WORKLOG_R4.md", "RELEASE_WORKLOG_R4.md"),
    ("paper/AI_USE_AND_PRIVACY.md", "AI_USE_AND_PRIVACY.md"),
    ("paper/FEATURE_TIMING_EVIDENCE.md", "FEATURE_TIMING_EVIDENCE.md"),
    ("experiment/full_results_table_cells.json", "full_results_table_cells.json"),
    ("experiment/dev_results_table_cells.json", "dev_results_table_cells.json"),
    ("experiment/feature_governance_cells.json", "feature_governance_cells.json"),
    ("experiment/cohort_year_table.json", "cohort_year_table.json"),
]:
    (REL / rel_dst).write_bytes((REM / rel_src).read_bytes())

# 3c. tracked-change (redline) PDFs: r3->r4 (this release) + r2->r3 (historical) ----
_redline = SRC / "redline_r3_to_r4.pdf"
if _redline.exists():
    (REL / "redline_r3_to_r4.pdf").write_bytes(_redline.read_bytes())
else:
    print("[package] WARNING: redline_r3_to_r4.pdf absent; regenerate with latexdiff")
_redline_prev = SRC / "redline_r2_to_r3.pdf"
if _redline_prev.exists():
    (REL / "redline_r2_to_r3.pdf").write_bytes(_redline_prev.read_bytes())

# 4. page renders + contact sheet ----------------------------------------------
rend = REL / "page_render"
rend.mkdir(exist_ok=True)
for p in sorted((SRC / "page_render").glob("page-*.png")):
    (rend / p.name).write_bytes(p.read_bytes())
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
pages = sorted(rend.glob("page-*.png"))
cols, rows_ = 8, (len(pages) + 7) // 8
fig, axes = plt.subplots(rows_, cols, figsize=(cols * 2.0, rows_ * 2.6))
for ax in axes.ravel():
    ax.axis("off")
for i, p in enumerate(pages):
    ax = axes.ravel()[i]
    ax.imshow(mpimg.imread(str(p)))
    ax.set_title(p.stem.replace("page-", "p."), fontsize=7)
fig.suptitle(f"{PDF_NAME} — {len(pages)} pages", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(rend / "contact_sheet.png", dpi=130)
plt.close(fig)

# 5. build manifest ------------------------------------------------------------
env = {}
for mod in ("numpy", "pandas", "sklearn", "scipy", "pyarrow", "pypdf", "matplotlib"):
    try:
        env[mod] = __import__(mod).__version__
    except Exception:
        env[mod] = "absent"
latex_ver = subprocess.run(["pdflatex", "--version"], capture_output=True,
                           text=True).stdout.splitlines()[0]
build_manifest = {
    "release": "portfolio_final_r4",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "baseline": {
        "governed_baseline_commit": BASELINE_COMMIT,
        "prior_release": "r3 submission-readiness release (preserved unchanged; tag portfolio-final-r3)",
        "prior_pdf_sha256": "ad417e6794319ece723d96d24e009bfc67db510360087130b1422e6a32b6c283",
        "prior_pdf_bytes": 1024944, "prior_pdf_pages": 56,
        "prior_repo_commit": "111f6f9",
    },
    "public_release": {
        "repository": "https://github.com/Bobtheotherone/Alaska_Crash_Analysis",
        "tag": "portfolio-final-r4",
        "release_url": "https://github.com/Bobtheotherone/Alaska_Crash_Analysis/releases/tag/portfolio-final-r4",
        "publication_date": "2026-07-20",
        "doi": None,
        "doi_note": "No archival DOI had been issued at the time this submission package "
                    "was finalized; the tagged public release is dated July 20, 2026.",
    },
    "revision": {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "repo_clean_after_commit": git("status", "--porcelain") == "",
    },
    "final_pdf": {"file": PDF_NAME, "sha256": sha256(pdf),
                  "bytes": pdf.stat().st_size, "pages": n_pages},
    "source_archive": {"file": ZIP_NAME, "sha256": sha256(zip_path),
                       "bytes": zip_path.stat().st_size, "entries": len(include)},
    "environment": {"os": platform.platform(), "python": platform.python_version(),
                    "latex": latex_ver, **env},
    "build_commands": [
        "bash build.sh   (pdflatex x4 + bibtex, from remediation/paper/latex)",
        "python tools/gen_reanalysis_tables.py && python tools/gen_hyperparameter_ledger.py && python tools/gen_analysis_status_ledger.py && python tools/gen_metric_conventions.py",
        "python tools/verify_manuscript_numbers.py paper/latex paper/latex/main.pdf",
        "python tools/verify_frozen_and_locked.py [kit internal_checks.py]",
        "python -m pytest tests -q --basetemp C:\\tmp\\aca-tests",
        "python tools/audit_paper.py --render paper/latex/page_render",
        "python tools/package_release.py <release_dir>",
        "python tools/rebuild_identity_check.py --release-pdf <release_dir>/<pdf> --out <release_dir>/rebuild_identity.json",
        "python tools/pdf_token_diff.py <prior_pdf> <release_dir>/<pdf> --out <release_dir>/pdf_token_diff.json",
        "python tools/build_handoff.py <release_dir> <stage_parent> --tier1   (writes cleanroom_results.json)",
        "python tools/package_release.py <release_dir>   (2nd run folds captured clean-room/rebuild/token-diff results into FINAL_VERIFICATION_REPORT.md)",
    ],
    "locked_value_policy": "round-half-even (Python round) at each table's printed precision; "
                           "verified positionally per (table, row, column) against the frozen artifacts",
}
(REL / "build_manifest.json").write_text(json.dumps(build_manifest, indent=2), encoding="utf-8")

# 6. artifact manifest (repo artifacts this release relies on + release evidence) --
def entry(p: Path, status: str, note: str) -> dict:
    return {"path": str(p.relative_to(ACA)).replace("\\", "/"), "sha256": sha256(p),
            "bytes": p.stat().st_size, "status": status, "note": note}
am = {"generated_utc": build_manifest["generated_utc"],
      "repo_commit": build_manifest["revision"]["commit"],
      "artifacts": []}
for rel, note in [
    ("experiment/target_reporting_process_audit.json", "post hoc; target/reporting audit (agency note corrected 2026-07-15)"),
    ("experiment/target_reporting_process_audit.md", "post hoc; summary"),
    ("experiment/raw_vs_calibrated_decisions.json", "post hoc; raw-label proof + exact calibration rederivation"),
    ("experiment/severe_metric_uncertainty.json", "post hoc; severe-metric CIs"),
    ("experiment/severe_metric_uncertainty.md", "post hoc; summary"),
    ("experiment/cluster_bootstrap_sensitivity.json", "post hoc; proxy-cluster CIs"),
    ("experiment/agency_leave_one_out.json",
     "post hoc; DEFERRED (agency fields removed at de-identification; raw absent by policy) + proxies; status corrected 2026-07-15"),
    ("experiment/prediction_roundtrip_test.json", "post hoc; lossless round-trip verification"),
    ("experiment/predictions_lossless.parquet", "post hoc; deid lossless predictions (gitignored; shipped here)"),
    ("experiment/missingness_only_predictions.parquet", "post hoc; deid probe predictions (gitignored; shipped here)"),
    ("experiment/hyperparameter_ledger.json", "generated from registry code + frozen config (2026-07-15)"),
    ("paper/FEATURE_TIMING_EVIDENCE.md", "post hoc; per-field timing ledger"),
    ("paper/CLAIM_EVIDENCE_MATRIX.md", "updated; R5 agency correction"),
    ("paper/ANALYSIS_STATUS_LEDGER.csv", "generated status ledger (26 analyses incl. r3 rows A25/A26)"),
    ("paper/CORRECTION_LEDGER.json", "correction ledger: completion pass + locked correction release + r3 submission-readiness pass"),
    ("reanalysis/reanalysis_table_cells.json", "generated structured cells for the prior-iteration tables"),
    ("crashsev/manuscript_tables.py", "new module; positional table verification"),
    ("crashsev/target_reporting_audit.py", "module; agency wording corrected"),
    ("crashsev/cluster_bootstrap_sensitivity.py", "module; agency wording corrected"),
    ("tools/verify_manuscript_numbers.py", "hardened verifier (positional cells; git gate; correction-release language gate)"),
    ("tools/verify_frozen_and_locked.py", "repaired verifier (root-relative; explicit skips; no external-checkout fallback)"),
    ("tools/gen_reanalysis_tables.py", "generator"),
    ("tools/gen_hyperparameter_ledger.py", "generator"),
    ("tools/gen_analysis_status_ledger.py", "generator (correction-release schema; also emits the appendix table)"),
    ("tools/gen_metric_conventions.py", "generator (RPS/log-loss/ECE convention sidecar)"),
    ("tools/gen_final_verification_report.py", "report generator (captured gate output only)"),
    ("tools/pdf_token_diff.py", "numeric-token no-drift gate"),
    ("tools/rebuild_identity_check.py", "source-rebuild identity check"),
    ("experiment/metric_conventions.json", "generated metric-convention sidecar (frozen artifacts untouched)"),
    ("paper/latex/analysis_status_table.tex", "generated appendix table (analysis-status ledger)"),
    ("tests/test_manuscript_tables.py", "regression tests for the checker false-pass defect"),
    ("tests/test_verifier_paths.py", "R-002 regression tests (no drive-letter literals; decoy-checkout ignored)"),
    ("tests/test_metric_conventions.py", "convention-switch guards (RPS unnormalized; eps; ECE bins)"),
    ("tests/test_prediction_evidence.py", "parquet-authoritative + CSV default-parser regression + released-probability validity (r3)"),
    ("paper/REVISION_SUMMARY.md", "committed revision summaries (r3 submission-readiness pass + correction pass)"),
    ("paper/latex/main.tex", "manuscript source (r3 submission-readiness edits)"),
    # r3 submission-readiness artifacts
    ("tools/gen_full_results_tables.py", "generator (complete Appendix A tables from the frozen artifact; F1-CONV-001)"),
    ("tools/gen_dev_results_table.py", "generator (complete development record; selection semantics asserted)"),
    ("tools/gen_feature_governance_tables.py", "generator (retained-49 / prohibited-14 / disposition tables)"),
    ("tools/gen_cohort_year_table.py", "generator (year-by-year cohort, reconciled to frozen audits)"),
    ("experiment/full_results_table_cells.json", "generated structured cells (Appendix A)"),
    ("experiment/dev_results_table_cells.json", "generated structured cells (Appendix F)"),
    ("experiment/feature_governance_cells.json", "generated structured cells (Appendix G)"),
    ("experiment/cohort_year_table.json", "generated year-by-year cohort accounting"),
    ("paper/latex/full_results_tables.tex", "generated appendix tables (complete 2012 results)"),
    ("paper/latex/dev_results_table.tex", "generated appendix table (development record)"),
    ("paper/latex/feature_governance_tables.tex", "generated appendix tables (feature governance)"),
    ("paper/latex/cohort_year_table.tex", "generated table (year-by-year cohort)"),
    ("paper/latex/figures/user/fig06c_reliability.pdf", "severe-class reliability diagram (frozen calibrated points + recomputed raw curve)"),
    ("paper/AI_USE_AND_PRIVACY.md", "AI-use and privacy record (author attestation completed r4: local-only processing of sensitive records)"),
    ("paper/DISCREPANCY_LEDGER_R3.md", "Phase-1 verification ledger for the r3 review directive"),
    ("paper/REVISION_MEMO_R3.md", "r3 revision memorandum (location/original/revised/reason/evidence/impact)"),
    ("paper/AUTHOR_ACTIONS_R3.md", "unresolved author actions (markers 1-5 + standing items)"),
    ("paper/ADMISSIONS_REVIEW_CHECKLIST_R3.md", "final admissions checklist: PASS WITH AUTHOR ACTION"),
    ("paper/ORAL_DEFENSE_SHEET.md", "oral-defense question/answer sheet"),
    ("paper/NUMERICAL_VERIFICATION_R3.md", "per-table numerical verification report (artifact hashes + gates)"),
    ("paper/latex/redline_r2_to_r3.tex", "latexdiff --flatten tracked-change source vs r2"),
    # r4 final-submission artifacts
    ("paper/REVISION_MEMO_R4.md", "r4 revision memorandum (location/r3/r4/reason/evidence/impact)"),
    ("paper/AUTHOR_ACTIONS_R4.md", "r4 author actions (markers resolved; remaining external publication actions only)"),
    ("paper/ADMISSIONS_REVIEW_CHECKLIST_R4.md", "final r4 admissions checklist"),
    ("paper/NUMERICAL_VERIFICATION_R4.md", "r4 per-table numerical verification report"),
    ("paper/PUBLIC_REPOSITORY_AUDIT_R4.md", "full-history public-repository privacy audit (F-R4-01 containment)"),
    ("paper/PRIVACY_AND_NDA_ATTESTATION_R4.md", "author privacy/NDA attestation of record"),
    ("paper/GITHUB_PUBLICATION_R4.md", "public-repository publication record (URL/tag/release/CI/DOI status)"),
    ("paper/RELEASE_WORKLOG_R4.md", "r4 release worklog (every command and remote action)"),
    ("paper/latex/redline_r3_to_r4.tex", "latexdiff --flatten tracked-change source vs r3"),
]:
    am["artifacts"].append(entry(REM / rel, "committed" if not rel.endswith(".parquet")
                                 else "local+release", note))
(REL / "artifact_manifest.json").write_text(json.dumps(am, indent=2), encoding="utf-8")

# 6b. FINAL_VERIFICATION_REPORT.md — generated ONLY from captured outputs (D-001) ----
import importlib.util as _ilu
_fvr_spec = _ilu.spec_from_file_location(
    "gen_final_verification_report",
    Path(__file__).resolve().parent / "gen_final_verification_report.py")
_fvr = _ilu.module_from_spec(_fvr_spec)
_fvr_spec.loader.exec_module(_fvr)
(REL / "FINAL_VERIFICATION_REPORT.md").write_text(_fvr.build(REL), encoding="utf-8")
print("[package] FINAL_VERIFICATION_REPORT.md generated from captured gate output")

# 6c. fail-closed forbidden-content scan of the assembled release (r4, F-R4-01) -----
# Privacy scan: prohibited source/credential file types may never be staged, no
# generic secret pattern may appear, and the licensed raw workbook's identity (read
# from the LOCAL-ONLY restricted reproduction log, so this script never embeds it)
# may not appear in any staged byte. Editorial-marker absence is enforced separately
# by verify_manuscript_numbers.py (gate above); historical r3 documents legitimately
# QUOTE the old marker string and are not re-flagged here.
FORBIDDEN_SUFFIXES = (".xlsx", ".xls", ".xlsm", ".env", ".pem", ".key", ".bundle",
                      ".sqlite", ".sqlite3", ".db")
_bad_paths = [p for p in REL.rglob("*") if p.is_file() and
              (p.name.lower().endswith(FORBIDDEN_SUFFIXES)
               or "credential" in p.name.lower() or "secret" in p.name.lower()
               or ".private-manifest" in p.name.lower())]
if _bad_paths:
    raise SystemExit(f"[package] FORBIDDEN FILE STAGED: {[str(p) for p in _bad_paths[:5]]}")
_secret_res = [re.compile(rb"AKIA[0-9A-Z]{16}"),
               re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
               re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
               re.compile(rb"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
               re.compile(rb"sk-ant-[A-Za-z0-9\-_]{20,}"),
               re.compile(rb"xox[baprs]-[0-9A-Za-z\-]{10,}")]
_priv_tokens: list[bytes] = []
_restricted_log = ACA / "private_reproduction_log" / "RAW_WORKBOOK_IDENTITY.md"
if _restricted_log.exists():
    _m = re.search(r"SHA-256:\s*([0-9a-fA-F]{64})",
                   _restricted_log.read_text(encoding="utf-8"))
    if _m:
        _h = _m.group(1).lower().encode()
        _priv_tokens += [_h, _h[:8]]
else:
    print("[package] NOTE: restricted reproduction log absent; raw-hash byte scan "
          "skipped (generic secret patterns still enforced)")
_scan_hits = []
for p in sorted(REL.rglob("*")):
    if not p.is_file() or p.name == "SHA256SUMS.txt":
        continue
    if p.suffix.lower() == ".png":
        continue  # page renders / contact sheet
    data = p.read_bytes()
    low = data.lower()
    for t in _priv_tokens:
        if t in low:
            _scan_hits.append(f"{p.relative_to(REL)}: restricted identifier")
    for rx in _secret_res:
        if rx.search(data):
            _scan_hits.append(f"{p.relative_to(REL)}: secret pattern {rx.pattern[:24]!r}")
if _scan_hits:
    raise SystemExit("[package] FORBIDDEN CONTENT: " + " | ".join(_scan_hits[:8]))
print(f"[package] forbidden-content scan: clean "
      f"({sum(1 for p in REL.rglob('*') if p.is_file())} staged files)")

# 7. SHA256SUMS over every release file (excluding page renders; listed separately) --
lines = []
for p in sorted(REL.rglob("*")):
    if p.is_dir() or p.name == "SHA256SUMS.txt":
        continue
    rel = p.relative_to(REL).as_posix()
    if rel.startswith("page_render/page-"):
        continue  # renders summarized by the contact sheet + build manifest
    lines.append(f"{sha256(p)}  {rel}")
(REL / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"[package] release assembled at {REL}")
print(f"[package] PDF {sha256(pdf)} ({n_pages} pp, {pdf.stat().st_size} B)")
print(f"[package] source zip {sha256(zip_path)}")
print(f"[package] SHA256SUMS entries: {len(lines)}")
