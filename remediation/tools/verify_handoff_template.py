#!/usr/bin/env python3
"""
VERIFY_HANDOFF.py - standalone integrity/reproducibility verifier for the Alaska Crash
Analysis final-submission handoff. Run it from the package root (the directory containing
SHA256SUMS.txt):

    python VERIFY_HANDOFF.py

It verifies PACKAGE INTEGRITY and declared REPRODUCIBLE RELATIONSHIPS. It does NOT certify
that the paper's scientific conclusions are correct - see README_REPRODUCE.md for scope and
for the deeper verification tiers (pytest suite, manuscript-number verification, bootstrap
replay). Exit code is 0 only if every hard check passes. Standard library only.
"""
import csv
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(ROOT, "SHA256SUMS.txt")
SELF = os.path.basename(os.path.abspath(__file__))

problems = []
notes = []


def fail(msg):
    problems.append(msg)


def note(msg):
    notes.append(msg)


def pr(s):
    try:
        print(s)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        sys.stdout.write(s.encode(enc, "replace").decode(enc) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 1. manifest hashes
def parse_manifest():
    entries = []
    if not os.path.exists(MANIFEST):
        fail("SHA256SUMS.txt not found - run from the package root.")
        return entries
    with open(MANIFEST, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([0-9a-f]{64})  (.+)$", line)
            if m:
                entries.append((m.group(1), m.group(2)))
    return entries


def check_hashes(entries):
    listed = set()
    for want, rel in entries:
        listed.add(rel.replace("\\", "/"))
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            fail(f"missing file listed in manifest: {rel}")
            continue
        if sha256(p) != want:
            fail(f"HASH MISMATCH: {rel}")
    return listed


# ---------------------------------------------------------------- 2. unexpected files
ALLOWED_UNLISTED = {"SHA256SUMS.txt"}


def check_unexpected(listed):
    on_disk = set()
    for dp, dn, fn in os.walk(ROOT):
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), ROOT).replace("\\", "/")
            on_disk.add(rel)
    for rel in sorted(on_disk - listed - ALLOWED_UNLISTED):
        fail(f"unexpected file not in manifest: {rel}")
    return on_disk


# ---------------------------------------------------------------- 3. structural hygiene
FORBIDDEN_PATTERNS = [
    (re.compile(r"(^|/)\.venv/|(^|/)venv/|site-packages/"), "vendored virtualenv"),
    (re.compile(r"__pycache__/|\.pyc$"),                    "bytecode/cache"),
    (re.compile(r"\.pytest_cache/"),                        "test cache"),
    (re.compile(r"\.sqlite3?$|db\.sqlite"),                 "database state"),
    (re.compile(r"\.env$"),                                 "env file"),
    (re.compile(r"\.xlsx$"),                                "raw spreadsheet (restricted data)"),
    (re.compile(r"Handoff.*\.zip$|handoff.*\.zip$"),        "nested handoff archive"),
]


def check_structure(on_disk):
    for rel in sorted(on_disk):
        for pat, label in FORBIDDEN_PATTERNS:
            if pat.search(rel):
                fail(f"forbidden content present ({label}): {rel}")
        if os.path.isabs(rel) or re.match(r"^[A-Za-z]:", rel):
            fail(f"absolute path in tree: {rel}")


# ---------------------------------------------------------------- 4. identifiers & secrets
REAL_ID = re.compile(r"\b20\d{7}\b")
SURR_OK = re.compile(r"^T\d{5}$")
SECRET = re.compile(
    r"BEGIN [A-Z ]*PRIVATE KEY|aws_secret_access_key|ghp_[A-Za-z0-9]{20}|xox[baprs]-[A-Za-z0-9-]{10}",
    re.I,
)
TEXT_EXT = {".md", ".txt", ".py", ".csv", ".json", ".yml", ".yaml", ".cfg", ".ini", ".toml"}


def check_identifiers_and_secrets(on_disk):
    for rel in sorted(on_disk):
        if os.path.splitext(rel)[1].lower() not in TEXT_EXT:
            continue
        if rel == SELF or rel.endswith("tools/verify_handoff_template.py"):
            continue  # this verifier (and its committed template copy in canonical/)
            # contains secret-SHAPED regex literals by design; bytes are manifest-pinned
        p = os.path.join(ROOT, rel)
        try:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                data = fh.read()
        except Exception:
            continue
        if SECRET.search(data):
            fail(f"possible secret pattern in {rel} (value not printed)")
        if "/evidence/" in ("/" + rel) and rel.endswith(".csv") and "prediction" in rel.lower():
            for row in data.splitlines()[1:]:
                first = row.split(",", 1)[0].strip()
                if REAL_ID.match(first) and not SURR_OK.match(first):
                    fail(f"real crash-id in shipped predictions {rel} (id not printed)")
                    break


# ---------------------------------------------------------------- 5. recompute headline metrics
EXPECTED_OMAE = {
    "v4_corrected_2012_evidence": {
        "ordinal_random_forest": 0.347,
        "random_forest": 0.376,
        "random_forest_unweighted": 0.340,
        "majority": 0.361,
    },
    "broad_tier_sensitivity_evidence": {
        "ordinal_random_forest": 0.333,
        "majority": 0.361,
    },
    "lowmiss_sensitivity_evidence": {
        "ordinal_random_forest": 0.352,
        "random_forest_unweighted": 0.339,
        "majority": 0.361,
    },
    "v3_historical_2012_evidence": {
        "ordinal_random_forest": 0.331,
        "random_forest": 0.344,
        "majority": 0.361,
    },
}
TOL = 0.002


def recompute_metrics():
    preds = []
    for dp, dn, fn in os.walk(os.path.join(ROOT, "evidence")):
        for f in fn:
            if re.match(r"predictions_.*\.csv$", f):
                preds.append(os.path.join(dp, f))
    if not preds:
        fail("no de-identified prediction CSVs found under evidence/")
        return
    checked = 0
    for f in preds:
        model = re.sub(r"^predictions_|\.csv$", "", os.path.basename(f))
        ys, yps = [], []
        with open(f, newline="", encoding="utf-8") as fh:
            for d in csv.DictReader(fh):
                ys.append(int(d["y_true"]))
                yps.append(int(d["y_pred"]))
        omae = sum(abs(a - b) for a, b in zip(yps, ys)) / len(ys)
        norm = f.replace("\\", "/")
        tier = next((t for t in EXPECTED_OMAE if "/" + t + "/" in norm), None)
        if tier and model in EXPECTED_OMAE[tier]:
            checked += 1
            if abs(omae - EXPECTED_OMAE[tier][model]) > TOL:
                fail(f"metric mismatch [{tier}]: {model} ordinal MAE recomputed {omae:.4f}, "
                     f"expected ~{EXPECTED_OMAE[tier][model]:.3f}")
            else:
                note(f"metric OK [{tier}]: {model} ordinal MAE {omae:.4f} (n={len(ys)})")
    if checked < 12:
        fail(f"only {checked} tier/model headline metrics were checkable (expected 12)")


# ------------------------------------------------- 6. lossless parquet hashes vs committed pins
def check_parquet_pins():
    pins = [
        ("evidence/predictions_lossless.parquet",
         "canonical/remediation/experiment/prediction_roundtrip_test.json",
         ("lossless_parquet", "sha256")),
        ("evidence/missingness_only_predictions.parquet",
         "canonical/remediation/experiment/target_reporting_process_audit.json",
         ("q9_missingness_indicator_only_probe", "predictions_parquet", "sha256")),
    ]
    for rel, pin_file, keys in pins:
        p = os.path.join(ROOT, rel)
        pf = os.path.join(ROOT, pin_file)
        if not os.path.exists(p):
            fail(f"missing de-identified evidence file: {rel}")
            continue
        if not os.path.exists(pf):
            fail(f"missing pin artifact: {pin_file}")
            continue
        with open(pf, encoding="utf-8") as fh:
            obj = json.load(fh)
        for k in keys:
            obj = obj[k]
        if sha256(p) != obj:
            fail(f"parquet hash does not match the committed pin: {rel}")
        else:
            note(f"parquet pin OK: {rel}")


# ---------------------------------------------------------------- 7. claim-matrix references
def check_claim_matrix():
    cm = os.path.join(ROOT, "canonical", "remediation", "paper", "CLAIM_EVIDENCE_MATRIX.md")
    if not os.path.exists(cm):
        fail("canonical/remediation/paper/CLAIM_EVIDENCE_MATRIX.md missing")
        return
    with open(cm, encoding="utf-8") as fh:
        text = fh.read()
    refs = set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:py|md|json|csv|yml|yaml))`", text))
    missing = []
    for ref in sorted(refs):
        cands = [
            os.path.join(ROOT, "canonical", "remediation", ref),
            os.path.join(ROOT, "canonical", ref),
            os.path.join(ROOT, ref),
        ]
        if any(os.path.exists(c) for c in cands) or ("/" not in ref):
            continue
        if ref.endswith(".parquet") or "/runs/" in ref or ref.startswith("_local_data"):
            continue  # local-only by policy; documented in README_REPRODUCE.md
        missing.append(ref)
    if missing:
        note(f"claim-matrix references not resolved locally ({len(missing)}) - "
             f"some may be legacy prior-source or local-only paths: {', '.join(missing[:6])}"
             + (" ..." if len(missing) > 6 else ""))
    else:
        note("all claim-matrix file references resolve within the package.")


# ---------------------------------------------------------------- main
def main():
    entries = parse_manifest()
    listed = check_hashes(entries)
    on_disk = check_unexpected(listed)
    check_structure(on_disk)
    check_identifiers_and_secrets(on_disk)
    recompute_metrics()
    check_parquet_pins()
    check_claim_matrix()

    pr("=" * 72)
    pr("Alaska Crash Analysis - final-submission handoff verification")
    pr("=" * 72)
    pr(f"files hashed & verified : {len(entries)}")
    pr(f"files on disk           : {len(on_disk)}")
    for m in notes:
        pr(f"  note: {m}")
    if problems:
        pr(f"\nRESULT: FAIL ({len(problems)} hard problem(s))")
        for p in problems:
            pr(f"  FAIL: {p}")
        return 1
    pr("\nRESULT: PASS - integrity, hygiene, privacy, headline-metric, and pin checks all passed.")
    pr("(This verifies the package, not the scientific conclusions; see README_REPRODUCE.md.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
