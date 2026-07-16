"""Build the self-contained final-submission handoff: stage, manifest, deterministic zip,
and clean-room extraction test.

    python tools/build_handoff.py <release_dir> <stage_parent> [--prior-zip <path>]

* <release_dir>  — output of tools/package_release.py (final PDF, source zip, ledgers…)
* <stage_parent> — directory in which the stage folder and the zip are created
* --prior-zip    — prior handoff zip whose de-identified evidence/ tiers are carried
                   forward byte-identically (default: dist/Alaska_Crash_Analysis_Portfolio_Handoff_42194a6.zip)

Steps: require a clean tree; git-archive HEAD into canonical/; copy release deliverables,
ledgers, README_REPRODUCE, and the stand-alone verifier; extract the prior deid evidence
tiers; add the two lossless parquets (hash-checked against their committed pins); write
provenance (git state, log, full bundle); write MANIFEST.json + SHA256SUMS.txt;
build a deterministic zip; then extract the zip into a fresh temp directory and run
VERIFY_HANDOFF.py there (hard gate).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REM = Path(__file__).resolve().parents[1]
ACA = REM.parent

if len(sys.argv) < 3:
    raise SystemExit(__doc__)
REL = Path(sys.argv[1])
STAGE_PARENT = Path(sys.argv[2])
PRIOR_ZIP = Path(sys.argv[sys.argv.index("--prior-zip") + 1]) if "--prior-zip" in sys.argv \
    else ACA / "dist" / "Alaska_Crash_Analysis_Portfolio_Handoff_42194a6.zip"

PDF_NAME = "Mercado-Barbosa_UAA_Student_Paper_Final_Submission.pdf"
SRC_ZIP_NAME = "Mercado-Barbosa_UAA_Student_Paper_Final_Submission_Source.zip"


def git(*args, check=True) -> str:
    r = subprocess.run(["git", "-C", str(ACA), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr[:300]}")
    return r.stdout.strip()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


assert git("status", "--porcelain") == "", "tree must be clean before packaging"
HEAD = git("rev-parse", "HEAD")
SHORT = git("rev-parse", "--short", "HEAD")
BRANCH = git("rev-parse", "--abbrev-ref", "HEAD")
TOP = f"Alaska_Crash_Analysis_Portfolio_Final_{SHORT}"
STAGE = STAGE_PARENT / TOP
if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)

# ---- 1. canonical/ = git archive at HEAD (exact blob bytes) --------------------
zbytes = subprocess.run(
    ["git", "-C", str(ACA), "-c", "core.autocrlf=false", "archive", "--format=zip", HEAD],
    capture_output=True)
assert zbytes.returncode == 0, zbytes.stderr[:300]
tmp_zip = STAGE_PARENT / f"_canonical_{SHORT}.zip"
tmp_zip.write_bytes(zbytes.stdout)
with zipfile.ZipFile(tmp_zip) as z:
    z.extractall(STAGE / "canonical")
tmp_zip.unlink()
n_canon = sum(1 for _ in (STAGE / "canonical").rglob("*") if _.is_file())
print(f"[handoff] canonical/: {n_canon} files @ {SHORT}")

# ---- 2. paper/ deliverables ------------------------------------------------------
paper = STAGE / "paper"
paper.mkdir()
for name in (PDF_NAME, SRC_ZIP_NAME, "build_manifest.json", "artifact_manifest.json",
             "test_report.txt", "audit_resolution_matrix.csv",
             "custodian_semantics_request.md"):
    shutil.copyfile(REL / name, paper / name)
# captured-output evidence backing FINAL_VERIFICATION_REPORT.md (present after the
# corresponding capture step has run; copied when available)
for name in ("gate_outputs.json", "cleanroom_results.json", "rebuild_identity.json",
             "pdf_token_diff.json", "metric_conventions.json"):
    if (REL / name).exists():
        shutil.copyfile(REL / name, paper / name)
shutil.copyfile(REL / "page_render" / "contact_sheet.png", paper / "contact_sheet.png")
audits = paper / "audit_reports"
audits.mkdir()
WS = REL.parent.parent  # workspace root holding the AXIOM reports
for name in ("AXIOM_LOCAL_Audit_Report_d7ea4b7d.md", "AXIOM_LOCAL_Audit_Report_f8aab323.md",
             "AXIOM_LOCAL_Audit_Report_c94ba2b6.md"):
    src = WS / name
    if src.exists():
        shutil.copyfile(src, audits / name)
    else:
        print(f"[handoff] WARNING: audit report not found: {src}")

# ---- 3. top-level ledgers, summaries, verifier -----------------------------------
for src, dst in [
    (REL / "CORRECTION_LEDGER.json", "CORRECTION_LEDGER.json"),
    (REL / "ANALYSIS_STATUS_LEDGER.csv", "ANALYSIS_STATUS_LEDGER.csv"),
    (REL / "unresolved_external_evidence.md", "unresolved_external_evidence.md"),
    (REL / "REVISION_SUMMARY.md", "REVISION_SUMMARY.md"),
    (REL / "FINAL_VERIFICATION_REPORT.md", "FINAL_VERIFICATION_REPORT.md"),
    # r4 final-submission reports at package root for reviewer visibility
    (REL / "REVISION_MEMO_R4.md", "REVISION_MEMO_R4.md"),
    (REL / "AUTHOR_ACTIONS_R4.md", "AUTHOR_ACTIONS_R4.md"),
    (REL / "PUBLIC_REPOSITORY_AUDIT_R4.md", "PUBLIC_REPOSITORY_AUDIT_R4.md"),
    (REL / "PRIVACY_AND_NDA_ATTESTATION_R4.md", "PRIVACY_AND_NDA_ATTESTATION_R4.md"),
    (REM / "paper" / "README_REPRODUCE.md", "README_REPRODUCE.md"),
    (REM / "tools" / "verify_handoff_template.py", "VERIFY_HANDOFF.py"),
]:
    shutil.copyfile(src, STAGE / dst)

# ---- 4. evidence/: prior deid tiers (byte-identical) + lossless parquets ---------
with zipfile.ZipFile(PRIOR_ZIP) as z:
    members = [m for m in z.namelist() if "/evidence/" in m and not m.endswith("/")]
    for m in members:
        rel = m.split("/", 1)[1]  # strip the prior top dir
        dest = STAGE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(z.read(m))
print(f"[handoff] evidence/: {len(members)} prior deid files carried forward")
pins = {
    "predictions_lossless.parquet": json.loads(
        (REM / "experiment" / "prediction_roundtrip_test.json").read_text("utf-8")
    )["lossless_parquet"]["sha256"],
    "missingness_only_predictions.parquet": json.loads(
        (REM / "experiment" / "target_reporting_process_audit.json").read_text("utf-8")
    )["q9_missingness_indicator_only_probe"]["predictions_parquet"]["sha256"],
}
for name, pin in pins.items():
    src = REM / "experiment" / name
    got = sha256(src)
    assert got == pin, f"{name}: hash {got[:12]} != committed pin {pin[:12]}"
    shutil.copyfile(src, STAGE / "evidence" / name)
print("[handoff] lossless parquets added (pins verified)")

# ---- 5. provenance ----------------------------------------------------------------
# r4: this package is a PUBLIC release asset, so it carries no git bundle. A bundle
# embeds complete history (the r3 controlled-delivery package shipped one with --all,
# including every local branch); the r4 privacy audit (PUBLIC_REPOSITORY_AUDIT_R4.md,
# F-R4-01) withholds parts of the local lineage from public distribution pending the
# author's publication-route decision. History provenance for the public package is
# the public repository itself at the release tag; canonical/ already carries the
# exact source tree bytes, and COMMIT_LOG.txt records subjects only.
prov = STAGE / "provenance"
prov.mkdir()
(prov / "COMMIT_LOG.txt").write_text(git("log", "--oneline") + "\n", encoding="utf-8")
tags = git("tag", "-l", "portfolio-*")
(prov / "REPO_GIT_STATE.txt").write_text(
    "Alaska Crash Analysis - repository state at final-submission packaging\n"
    f"commit: {HEAD}\nshort:  {SHORT}\nbranch: {BRANCH}\n"
    f"tree:   clean\n\n== portfolio tags ==\n{tags}\n\n"
    "history: public repository "
    "https://github.com/Bobtheotherone/Alaska_Crash_Analysis at release tag "
    "portfolio-final-r4 (exact commit above); this package intentionally carries "
    "no git bundle (see paper/PUBLIC_REPOSITORY_AUDIT_R4.md)\n"
    "source tree: byte-exact copy under canonical/ (git archive of the release "
    "commit)\n", encoding="utf-8")

# ---- 6. MANIFEST.json + SHA256SUMS.txt ---------------------------------------------
entries = []
for p in sorted(STAGE.rglob("*")):
    if p.is_file():
        entries.append({"path": p.relative_to(STAGE).as_posix(),
                        "sha256": sha256(p), "bytes": p.stat().st_size})
manifest = {
    "package": TOP,
    "release_commit": HEAD,
    "branch": BRANCH,
    "final_pdf": {"path": f"paper/{PDF_NAME}",
                  "sha256": next(e["sha256"] for e in entries if e["path"] == f"paper/{PDF_NAME}")},
    "files": len(entries) + 2,  # + MANIFEST.json + SHA256SUMS.txt
    "entries": entries,
}
(STAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
sums = [f"{sha256(STAGE / 'MANIFEST.json')}  MANIFEST.json"] + \
       [f"{e['sha256']}  {e['path']}" for e in entries]
(STAGE / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

# ---- 6b. fail-closed forbidden-content scan of the staged package (r4) --------------
# Same policy as package_release.py §6c: no prohibited file types, no generic secret
# patterns, and the licensed raw workbook's identity (read from the LOCAL-ONLY
# restricted reproduction log so this script never embeds it) in no staged byte.
import re as _re
_FORBIDDEN_SUFFIXES = (".xlsx", ".xls", ".xlsm", ".env", ".pem", ".key", ".bundle",
                       ".sqlite", ".sqlite3", ".db")
_bad = [p for p in STAGE.rglob("*") if p.is_file() and
        (p.name.lower().endswith(_FORBIDDEN_SUFFIXES)
         or "credential" in p.name.lower() or ".private-manifest" in p.name.lower())]
assert not _bad, f"[handoff] FORBIDDEN FILE STAGED: {[str(p) for p in _bad[:5]]}"
_secret_res = [_re.compile(rb"AKIA[0-9A-Z]{16}"),
               _re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
               _re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
               _re.compile(rb"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
               _re.compile(rb"sk-ant-[A-Za-z0-9\-_]{20,}"),
               _re.compile(rb"xox[baprs]-[0-9A-Za-z\-]{10,}")]
_priv_tokens = []
_rlog = ACA / "private_reproduction_log" / "RAW_WORKBOOK_IDENTITY.md"
if _rlog.exists():
    _m = _re.search(r"SHA-256:\s*([0-9a-fA-F]{64})", _rlog.read_text(encoding="utf-8"))
    if _m:
        _h = _m.group(1).lower().encode()
        _priv_tokens += [_h, _h[:8]]
else:
    print("[handoff] NOTE: restricted reproduction log absent; raw-hash byte scan "
          "skipped (generic secret patterns still enforced)")
_hits = []
for p in sorted(STAGE.rglob("*")):
    if not p.is_file() or p.suffix.lower() in (".png", ".parquet"):
        continue  # renders; parquets are pin-verified against committed hashes above
    data = p.read_bytes()
    low = data.lower()
    for t in _priv_tokens:
        if t in low:
            _hits.append(f"{p.relative_to(STAGE)}: restricted identifier")
    for rx in _secret_res:
        if rx.search(data):
            _hits.append(f"{p.relative_to(STAGE)}: secret pattern")
assert not _hits, "[handoff] FORBIDDEN CONTENT: " + " | ".join(_hits[:8])
print(f"[handoff] forbidden-content scan: clean "
      f"({sum(1 for p in STAGE.rglob('*') if p.is_file())} staged files)")

# ---- 7. deterministic zip -----------------------------------------------------------
OUT = STAGE_PARENT / f"{TOP}.zip"
files = sorted(p for p in STAGE.rglob("*") if p.is_file())
with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for full in files:
        rel = full.relative_to(STAGE).as_posix()
        assert "\\" not in rel and not rel.startswith("/") and ":" not in rel, rel
        zi = zipfile.ZipInfo(f"{TOP}/{rel}", date_time=(2026, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o644 << 16
        z.writestr(zi, full.read_bytes())
zip_sha = sha256(OUT)
print(f"[handoff] zip: {OUT}")
print(f"[handoff] entries: {len(files)}  size: {OUT.stat().st_size} B "
      f"({OUT.stat().st_size/1048576:.2f} MiB)")
print(f"[handoff] sha256: {zip_sha}")

# ---- 8. clean-room extraction test ---------------------------------------------------
# extract to a SHORT path: Windows MAX_PATH/ACL failures at long %TEMP% paths are
# environment failures, not scientific ones (README_REPRODUCE documents the distinction)
_short_parent = Path("C:/tmp") if Path("C:/").exists() else None
if _short_parent is not None:
    _short_parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="aca_clean_",
                                 dir=str(_short_parent) if _short_parent else None) as td:
    with zipfile.ZipFile(OUT) as z:
        z.extractall(td)
    root = Path(td) / TOP
    r = subprocess.run([sys.executable, str(root / "VERIFY_HANDOFF.py")],
                       capture_output=True, text=True, cwd=str(root))
    tail = "\n".join(r.stdout.splitlines()[-4:])
    print(f"[cleanroom] VERIFY_HANDOFF.py exit={r.returncode}\n{tail}")
    if r.returncode != 0:
        print(r.stdout[-4000:])
        raise SystemExit("[handoff] CLEAN-ROOM VERIFICATION FAILED")

    if "--tier1" in sys.argv:
        # Tier-1 in the clean room: full no-license verification from the extraction.
        # Results are CAPTURED to the release dir; FINAL_VERIFICATION_REPORT.md reports
        # them as the clean-extraction counts (R-005), never merged with live counts.
        rem = root / "canonical" / "remediation"
        bt = Path(td) / "bt"
        checks = {}

        def cap(name: str, cmd: list[str], cwd: Path, hard: bool = True):
            rr = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))

            def keep(line: str) -> bool:
                s = line.strip()
                import re as _re
                return (("passed" in s and "failed" in s) or s.endswith("passed")
                        or bool(_re.match(r"^\d+ passed\b", s))
                        or s.startswith("[SKIP]")
                        or (s.startswith("[PASS]") and name.startswith("gen_")))

            summary = [ln.strip() for ln in rr.stdout.splitlines() if keep(ln)]
            checks[name] = {"cmd": " ".join(cmd[1:]), "exit": rr.returncode,
                            "summary_lines": summary[:12],
                            "tail": rr.stdout.splitlines()[-3:]}
            status = "OK" if rr.returncode == 0 else "FAIL"
            print(f"[cleanroom-tier1] {name}: {status} :: "
                  f"{summary[-1] if summary else ''}")
            if hard and rr.returncode != 0:
                print(rr.stdout[-3000:])
                print(rr.stderr[-1500:])
                raise SystemExit(f"[handoff] CLEAN-ROOM TIER-1 FAILED: {name}")

        cap("pytest", [sys.executable, "-m", "pytest", "tests", "-q",
                       "-p", "no:cacheprovider", "--basetemp", str(bt)], rem)
        for g in ("gen_reanalysis_tables", "gen_hyperparameter_ledger",
                  "gen_analysis_status_ledger", "gen_metric_conventions",
                  # r3 generated tables — feature-governance and cohort-year run their
                  # data-free structural/render modes inside an extraction by design
                  "gen_full_results_tables", "gen_dev_results_table",
                  "gen_feature_governance_tables", "gen_cohort_year_table"):
            cap(f"{g} --check", [sys.executable, f"tools/{g}.py", "--check"], rem)
        cap("verify_manuscript_numbers",
            [sys.executable, "tools/verify_manuscript_numbers.py", "paper/latex",
             str(root / "paper" / PDF_NAME)], rem)
        cap("verify_frozen_and_locked (clean extraction)",
            [sys.executable, "tools/verify_frozen_and_locked.py"], rem)
        (REL / "cleanroom_results.json").write_text(json.dumps({
            "extract_root": "clean-room temporary extraction (short path)",
            "commit": HEAD, "zip_sha256": zip_sha,
            "checks": checks}, indent=2) + "\n", encoding="utf-8")
        print(f"[cleanroom-tier1] captured -> {REL / 'cleanroom_results.json'}")
print("[handoff] DONE")
print(json.dumps({"zip": str(OUT), "sha256": zip_sha, "entries": len(files),
                  "commit": HEAD}, indent=2))
