"""Generate research_audit/RELEASE_STATE.md — the single machine-generated release record
(DOC-002). Hand-maintained status prose drifts (v3 and v4 both shipped stale self-certifications);
this file is regenerated at packaging time and is the only authoritative statement of:

  commit / branch / tags · canonical config · exact test count (from a live pytest run) ·
  governed runs and their generator commits · key artifact hashes · audit-gate results.

Run from ``remediation/``:  python research_audit/make_release_state.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent


def sh(args, cwd=REPO):
    return subprocess.run(args, capture_output=True, text=True, cwd=str(cwd)).stdout.strip()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    commit = sh(["git", "rev-parse", "HEAD"])
    short = sh(["git", "rev-parse", "--short", "HEAD"])
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    tags = sh(["git", "tag", "-l", "portfolio-*"]).splitlines()
    tag_lines = [f"| tag `{t}` | `{sh(['git', 'rev-list', '-n1', t])[:12]}` |" for t in tags]

    # live test count — with an explicitly created basetemp (the machine's default pytest temp
    # root can be ACL-broken, silently turning tmp_path tests into errors and understating the
    # count), and failing the record if ANY test errored or failed.
    import tempfile
    bt = Path(tempfile.mkdtemp(prefix="crashsev_release_state_")) / "bt"
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
                        f"--basetemp={bt}"],
                       capture_output=True, text=True, cwd=str(PKG))
    out = r.stdout + r.stderr
    m = re.search(r"(\d+) passed", out)
    bad = re.search(r"(\d+) (?:failed|error)", out)
    tests = (m.group(1) + " passed" if m else "UNKNOWN (pytest output unparsed)")
    tests_ok = bool(m) and not bad and r.returncode == 0
    if bad:
        tests += f" — GATE FAIL: {bad.group(0)}"

    # audit gates
    scan = subprocess.run([sys.executable, "research_audit/claim_scan.py"],
                          capture_output=True, text=True, cwd=str(PKG))
    layout = subprocess.run([sys.executable, "research_audit/pdf_layout_audit.py"],
                            capture_output=True, text=True, cwd=str(PKG))

    runs = []
    for d in sorted((PKG / "evidence_release").iterdir()):
        if d.is_dir() and (d / "manifest.json").exists():
            man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            runs.append((man.get("run_id", d.name), man.get("protocol_version", "?"),
                         man.get("feature_tier", "?"), man.get("decision_rule", "?"),
                         man.get("git", {}).get("commit", "?")[:9]))

    arts = {p: sha(PKG / p) for p in (
        "experiment/final_results.json", "experiment/development_report.json",
        "configs/route_r_09_12.yml", "data/feature_availability_ledger.csv",
        "data/target_mapping.yml", "requirements-lock.txt") if (PKG / p).exists()}

    lines = [
        "# RELEASE STATE (machine-generated — do not hand-edit; regenerate at packaging)",
        "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by "
        f"`research_audit/make_release_state.py`.",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| commit | `{commit}` (`{short}`) |",
        f"| branch | `{branch}` |",
        *tag_lines,
        f"| canonical config | `configs/route_r_09_12.yml` (v4 protocol; strict tier, posterior-median) |",
        f"| test suite | **{tests}** (live run at generation time) |",
        f"| claim scan | {'PASS' if scan.returncode == 0 else 'FAIL'} |",
        f"| PDF layout audit | {'PASS' if layout.returncode == 0 else 'FAIL'} |",
        "",
        "## Governed runs (committed skeletons under `evidence_release/`)",
        "",
        "| run_id | protocol | tier | rule | generator commit |",
        "|---|---|---|---|---|",
        *[f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | `{r[4]}` |" for r in runs],
        "",
        "## Key artifact hashes (SHA-256)",
        "",
        "| artifact | sha256 |",
        "|---|---|",
        *[f"| `{p}` | `{h}` |" for p, h in arts.items()],
        "",
        "Historical/planning documents (`FINAL_BENCHMARK_PROTOCOL.md` body, Route-A-era configs,",
        "prior release notes) are retained as records of what was specified when; where any prose",
        "conflicts with this generated file, **this file is authoritative**.",
    ]
    out = PKG / "research_audit" / "RELEASE_STATE.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"[release-state] wrote {out}  (tests: {tests}; scan "
          f"{'PASS' if scan.returncode == 0 else 'FAIL'}; layout "
          f"{'PASS' if layout.returncode == 0 else 'FAIL'})")
    return 0 if (tests_ok and scan.returncode == 0 and layout.returncode == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
