"""R-002 regression tests: no hard-coded machine paths; the acceptance verifier resolves
its root explicitly, reports explicit SKIPs for absent inputs, never falls back to
another checkout, and ignores a decoy `aca`-style checkout sitting elsewhere on disk.

Historical defect: tools/verify_frozen_and_locked.py pinned ``C:\\aca`` (and a
user-home auditor-kit path), so a clean-room extraction silently verified the author's
live checkout instead of the extracted package.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path

REM = Path(__file__).resolve().parents[1]

# single-letter drive followed by :\ or :/ starting a real path (escape sequences like
# "\n" in string literals are excluded by the negative lookahead)
DRIVE_LITERAL = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:(?:\\(?![ntr\\\"'])|/)")

# The ONLY tolerated drive-letter use outside verifiers: the deliberate short SCRATCH
# location C:\tmp (R-011 — long %TEMP% paths hit Windows MAX_PATH/ACL failures) and its
# existence guard Path("C:/"). These are write-target defaults for build tooling, never
# a source of verified truth, and they are forbidden entirely in verify_* files.
SCRATCH_OK = re.compile(r"[Cc]:[\\/](?:tmp\b|\"\)| ?$)")


def _py_sources():
    for sub in ("tools", "crashsev", "paper/latex/figure_generation"):
        for p in sorted((REM / sub).glob("*.py")):
            yield p


def test_no_drive_letter_path_literals_in_shipped_scripts():
    offenders = []
    for p in _py_sources():
        strict = p.name.startswith("verify_")
        text = p.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if DRIVE_LITERAL.search(line):
                if not strict and SCRATCH_OK.search(line):
                    continue
                offenders.append(f"{p.relative_to(REM)}:{i}: {line.strip()[:100]}")
    assert not offenders, "drive-letter path literals present:\n" + "\n".join(offenders)


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_frozen_and_locked", REM / "tools" / "verify_frozen_and_locked.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assignment_hash(pairs) -> str:
    import hashlib
    canon = sorted((str(i), str(p)) for i, p in pairs)
    return hashlib.sha256(json.dumps(canon, separators=(",", ":")).encode()).hexdigest()


def _make_min_package(root: Path, with_lock: bool = True) -> Path:
    """A minimal extracted-package layout, fully SYNTHETIC (hermetic: runs identically in
    the live checkout and in a clean-room extraction where the local-only FROZEN.lock and
    split_assignment.csv are absent by policy). No run bundles, no licensed table, no git.
    With the lock present, the lock-pin checks must PASS from these files alone."""
    import hashlib
    rem = root / "canonical" / "remediation"
    (rem / "experiment").mkdir(parents=True)
    dev = rem / "experiment" / "development_report.json"
    dev.write_text(json.dumps({"synthetic": True, "dev_cv": {}}), encoding="utf-8")
    split = rem / "experiment" / "split_assignment.csv"
    rows = [("1", "development"), ("2", "development"), ("3", "final_test")]
    split.write_text("row_id,partition\n" + "\n".join(f"{i},{p}" for i, p in rows) + "\n",
                     encoding="utf-8")
    if with_lock:
        (rem / "experiment" / "FROZEN.lock").write_text(json.dumps({
            "development_report_sha256": hashlib.sha256(dev.read_bytes()).hexdigest(),
            "split_sha256": _assignment_hash(rows),
            "data_sha256": "0" * 64,
        }), encoding="utf-8")
    (rem / "crashsev").mkdir()
    (rem / "crashsev" / "metrics.py").write_text("# marker for root detection\n",
                                                 encoding="utf-8")
    return rem


def _run_verifier(mod, rem: Path, capsys, extra=()):
    rc = mod.main(["--root", str(rem), *extra])
    out = capsys.readouterr().out
    return rc, out


def test_clean_package_passes_with_explicit_skips_and_ignores_decoy(tmp_path, capsys):
    mod = _load_verifier()
    rem = _make_min_package(tmp_path / "pkg")

    rc, out = _run_verifier(mod, rem, capsys)
    assert rc == 0, out
    # the two committed lock pins verify; everything unavailable is an explicit SKIP
    assert "[PASS] FROZEN.lock pins development_report.json byte hash" in out
    assert "[PASS] dev-side split canonical hash" in out
    assert out.count("[FAIL]") == 0
    assert "[SKIP] run bundle hashes / metric replay / bootstrap replay" in out
    assert "[SKIP] modeling-table canonical content hash" in out
    git_lines = [ln for ln in out.splitlines() if "git frozen-path gate" in ln]
    assert git_lines and all(ln.startswith("[SKIP]") for ln in git_lines), git_lines
    # the verifier states its confinement root and it is the tmp package, nothing else
    assert f"package top = {rem.parent}" in out

    # decoy: an unrelated aca-style checkout with corrupted frozen pins elsewhere on disk
    decoy = tmp_path / "decoy" / "aca" / "remediation"
    (decoy / "experiment").mkdir(parents=True)
    (decoy / "experiment" / "FROZEN.lock").write_text(
        json.dumps({"development_report_sha256": "0" * 64, "split_sha256": "0" * 64,
                    "data_sha256": "0" * 64}), encoding="utf-8")
    (decoy / "crashsev").mkdir()
    (decoy / "crashsev" / "metrics.py").write_text("# decoy\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path / "decoy" / "aca")],
                   capture_output=True)

    rc2, out2 = _run_verifier(mod, rem, capsys)
    assert rc2 == 0, out2
    assert str(decoy) not in out2 and "decoy" not in out2
    # identical verdict structure with and without the decoy present
    def counts(o):
        return (o.count("[PASS]"), o.count("[FAIL]"), o.count("[SKIP]"))
    assert counts(out2) == counts(out)


def test_require_full_fails_when_sections_skipped(tmp_path, capsys):
    mod = _load_verifier()
    rem = _make_min_package(tmp_path / "pkg")
    rc, out = _run_verifier(mod, rem, capsys, extra=("--require-full",))
    assert rc == 1
    assert "--require-full" in out


def test_extracted_archive_without_local_only_lock_skips_explicitly(tmp_path, capsys):
    """The real extracted handoff canonical/ tree has NO FROZEN.lock and NO
    split_assignment.csv (local-only by policy: the split file carries real crash
    identifiers). The verifier must report explicit SKIPs, never crash or fall back."""
    mod = _load_verifier()
    rem = _make_min_package(tmp_path / "pkg", with_lock=False)
    (rem / "experiment" / "split_assignment.csv").unlink()
    rc, out = _run_verifier(mod, rem, capsys)
    assert rc == 0, out
    assert out.count("[FAIL]") == 0
    assert "[SKIP] freeze-lock pins" in out
    assert "local-only" in out


def test_corrupted_pin_fails_in_clean_package(tmp_path, capsys):
    mod = _load_verifier()
    rem = _make_min_package(tmp_path / "pkg")
    lock = json.loads((rem / "experiment" / "FROZEN.lock").read_text("utf-8"))
    lock["development_report_sha256"] = "0" * 64
    (rem / "experiment" / "FROZEN.lock").write_text(json.dumps(lock), encoding="utf-8")
    rc, out = _run_verifier(mod, rem, capsys)
    assert rc == 1
    assert "[FAIL] FROZEN.lock pins development_report.json byte hash" in out
