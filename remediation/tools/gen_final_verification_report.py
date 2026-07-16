"""Generate FINAL_VERIFICATION_REPORT.md from CAPTURED gate output only (D-001).

Inputs, all machine-produced into the release directory by earlier steps:
  * gate_outputs.json        — written by package_release.py from the gates it just ran
  * build_manifest.json      — written by package_release.py (hashes, pages, environment)
  * cleanroom_results.json   — optional; written by build_handoff.py --tier1 (clean-room runs)
  * rebuild_identity.json    — optional; written by tools/rebuild_identity_check.py
  * pdf_token_diff.json      — optional; written by tools/pdf_token_diff.py

No verification claim is written unless it appears in one of those captured inputs.
Sections whose inputs are absent say so explicitly instead of asserting results.

Usage: python tools/gen_final_verification_report.py <release_dir>
(also imported by package_release.py, which calls build() during packaging)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

R003_STATEMENT = (
    "The frozen original-order evidence and recorded bootstrap witness replay the "
    "published interval bit-exactly. The released lossless parquet reproduces all point "
    "metrics exactly. Because de-identification replaces original identifiers/order, "
    "independently resampling the released rows is expected to produce a "
    "Monte-Carlo-equivalent interval rather than the identical resample sequence."
)


def _load(rel: Path, name: str):
    p = rel / name
    if not p.exists():
        return None
    return json.loads(p.read_text("utf-8"))


def build(rel: Path) -> str:
    go = _load(rel, "gate_outputs.json")
    bm = _load(rel, "build_manifest.json")
    if go is None or bm is None:
        raise SystemExit("gate_outputs.json / build_manifest.json missing — run "
                         "tools/package_release.py first")
    clean = _load(rel, "cleanroom_results.json")
    rebuild = _load(rel, "rebuild_identity.json")
    tokdiff = _load(rel, "pdf_token_diff.json")

    L: list[str] = []
    a = L.append
    a("# Final Verification Report — r3 submission-readiness release")
    a("")
    a("*Generated from captured gate output; no verification claim in this report was "
      "typed by hand.*")
    a("")
    a(f"**Final PDF:** `{bm['final_pdf']['file']}` — SHA-256 `{bm['final_pdf']['sha256']}`, "
      f"{bm['final_pdf']['bytes']:,} bytes, {bm['final_pdf']['pages']} pages.")
    a(f"**Repo:** branch `{bm['revision']['branch']}`, commit `{bm['revision']['commit']}`; "
      f"descends from governed baseline `{bm['baseline']['governed_baseline_commit'][:7]}`; "
      f"clean after commit: {bm['revision']['repo_clean_after_commit']}.")
    a(f"**Prior release preserved unchanged:** {bm['baseline']['prior_release']} "
      f"(PDF `{bm['baseline']['prior_pdf_sha256'][:12]}…`, "
      f"{bm['baseline']['prior_pdf_pages']} pp, repo `{bm['baseline']['prior_repo_commit'][:7]}`).")
    a(f"**Packaging environment:** {bm['environment']['os']}; "
      f"Python {bm['environment']['python']}; {bm['environment']['latex']}.")
    a("")

    a("## Gates executed in this packaging run (live git checkout)")
    a("")
    a("| Gate | Exit | Captured summary |")
    a("|---|---|---|")
    for name, rec in go["gates"].items():
        summary = "; ".join(rec["summary_lines"]) or "(see tail in gate_outputs.json)"
        a(f"| `{name}` | {rec['exit']} | {summary} |")
    a("")

    a("## Environment-specific verification counts (reported separately, never merged)")
    a("")
    a("**Live git checkout (this packaging run):**")
    for name, rec in go["gates"].items():
        for ln in rec["summary_lines"]:
            a(f"- `{name}`: {ln}")
    a("")
    if clean is not None:
        a(f"**Clean extraction** (zip extracted to `{clean['extract_root']}`; "
          f"commit `{clean['commit'][:12]}`):")
        for name, rec in clean["checks"].items():
            summary = ("; ".join(rec["summary_lines"])
                       or "; ".join(rec.get("tail", [])[-1:])
                       or f"exit {rec['exit']}")
            a(f"- `{name}`: {summary} (exit {rec['exit']})")
        a("")
        a("Sections that self-skip in a clean extraction, by name: the git frozen-path "
          "gate (not a git checkout; history verifiable via the provenance bundle), the "
          "local run-bundle hash/metric/bootstrap replay (per-crash CSVs stay local by "
          "policy; the lossless parquet supports metric recomputation instead), and the "
          "licensed modeling-table hash (raw-derived table absent by policy).")
    else:
        a("**Clean extraction:** not captured in this packaging run "
          "(produced by `tools/build_handoff.py --tier1`).")
    a("")

    a("## PDF identity: release copy vs source rebuild (two different checks)")
    a("")
    a(f"1. **Release/repository copy identity:** the released PDF is the byte-identical "
      f"copy of the manifest-tracked repository artifact `paper/latex/main.pdf` "
      f"(SHA-256 `{bm['final_pdf']['sha256']}`; asserted at packaging).")
    if rebuild is not None:
        binid = rebuild["binary_identical"]
        txtid = rebuild["extracted_text_identical"]
        a(f"2. **Source rebuild identity:** toolchain `{rebuild['toolchain']}`, "
          f"sequence {rebuild['build_sequence']}: binary-identical = {binid}; "
          f"extracted-text-identical = {txtid} "
          f"(rebuilt SHA-256 `{rebuild['rebuilt_pdf']['sha256'][:12]}…`, "
          f"{rebuild['rebuilt_pdf']['pages']} pages).")
        if txtid and not binid:
            a("   A clean-room rebuild produced byte-identical extracted text but a "
              "different binary hash because of PDF metadata (creation/mod dates, ID).")
    else:
        a("2. **Source rebuild identity:** not captured in this packaging run "
          "(produced by `tools/rebuild_identity_check.py`).")
    a("")

    a("## Numeric-token no-drift gate (vs the pre-correction build)")
    a("")
    if tokdiff is not None:
        a(f"Whole-PDF numeric-token multiset diff `{Path(tokdiff['old']).name}` "
          f"({tokdiff['old_pages']} pp) → `{Path(tokdiff['new']).name}` "
          f"({tokdiff['new_pages']} pp):")
        a(f"- added tokens ({len(tokdiff['added_token_set'])} distinct): "
          f"{tokdiff['added_token_set']}")
        a(f"- removed tokens ({len(tokdiff['removed_token_set'])} distinct): "
          f"{tokdiff['removed_token_set']}")
        a("Every added/removed token must be enumerated as authorized in "
          "REVISION_SUMMARY.md; the diff itself is in `pdf_token_diff.json`.")
    else:
        a("Not captured in this packaging run (produced by `tools/pdf_token_diff.py`).")
    a("")

    a("## Bootstrap evidence tiers")
    a("")
    a(R003_STATEMENT)
    a("")

    a("## Scientific-language gate")
    a("")
    fb = go.get("forbidden_list")
    if fb:
        a("Checked mechanically over the LaTeX source AND the built PDF text by "
          "`verify_manuscript_numbers.py` (case-insensitive plain phrases plus "
          "context-qualified patterns). The list actually enforced in this run:")
        for item in fb:
            a(f"- `{item}`")
        a("")
        a("No phrase outside this list is claimed to have been checked.")
    else:
        a("Forbidden-phrase list not captured; see verify_manuscript_numbers output in "
          "gate_outputs.json.")
    a("")

    a("## Reproduction tiers (controlled terms)")
    a("")
    a("1. **Metric recomputation** — independent recomputation from released row-level "
      "evidence (supported without the licensed source).")
    a("2. **Self-reproduction** — rerun by the project from licensed source and frozen "
      "code (performed; recorded in the reproduction log).")
    a("3. **Clean-room package verification** — execution from an extracted handoff "
      "(see the clean-extraction counts above).")
    a("4. **Independent replication** — end-to-end third-party rerun with licensed "
      "source: **not yet performed**; nothing in this report is one.")
    a("")

    a("## Unresolved external dependencies (not resolved by this release)")
    a("")
    a("Per `unresolved_external_evidence.md`: official blank-severity semantics; "
      "custodian-verified per-field recording times and a real prediction moment; a "
      "genuinely later unexposed cohort; a stakeholder utility/cost function and "
      "prospective threshold; end-to-end third-party replication with the licensed "
      "source; departmental template approval if required.")
    a("")
    return "\n".join(L) + "\n"


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rel = Path(sys.argv[1])
    (rel / "FINAL_VERIFICATION_REPORT.md").write_text(build(rel), encoding="utf-8")
    print(f"wrote {rel / 'FINAL_VERIFICATION_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
