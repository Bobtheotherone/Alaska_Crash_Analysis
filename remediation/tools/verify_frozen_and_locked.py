"""Acceptance verifier: frozen evidence unchanged + all governed numbers reproduce.

Checks (each PASSes, FAILs, or — when its required inputs are absent from the target
package — reports an explicit SKIP):
  1. every artifact hash in runs/final_8af9d5bc23d8/manifest.json re-verifies byte-exactly
     (SKIP when the local-only run bundle is not present, e.g. in an extracted handoff);
  2. FROZEN.lock still pins the development report (byte hash) and the canonical dev-side
     and run-side split-assignment hashes recompute to the frozen values;
  3. the modeling table's canonical content hash equals the governed data_sha256
     (SKIP when the licensed local table is absent by data-handling policy);
  4. all 14 models' oMAE/accuracy recompute exactly from the frozen predictions;
  5. all 13 paired bootstrap intervals reproduce bit-exactly under the declared procedure
     (seed 42, 2,000 replicates, crash-level paired percentile);
  6. git reports NO modification to any frozen path relative to the recorded baseline —
     run ONLY against the target package's own checkout; when the target is not a git
     checkout this is an explicit SKIP, never a fallback to some other directory;
  7. the auditor kit's internal arithmetic checks pass (optional positional argument).

Path policy (R-002): the remediation root is resolved from --root, then the
CRASHSEV_REMEDIATION environment variable, then by walking up from this file. Every
inspected path must live inside the resolved package top (the root's parent); the
verifier refuses to read anything outside it and never consults another checkout.

Usage:
    python verify_frozen_and_locked.py [kit_internal_checks.py] [--root <remediation_dir>]
                                       [--require-full]

Exit 0 iff no check FAILs (and, with --require-full, nothing was skipped).
Counts are environment-specific: a live git checkout with the local run bundles and the
licensed modeling table runs everything; a clean package extraction reports the
unavailable sections as SKIP by name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_COMMIT = "42194a619c3a8c1a146c1202ac6afe6a919e392b"


def _resolve_root(cli_root: str | None) -> Path:
    if cli_root:
        return Path(cli_root).resolve()
    env = os.environ.get("CRASHSEV_REMEDIATION")
    if env:
        return Path(env).resolve()
    for anc in Path(__file__).resolve().parents:
        if (anc / "crashsev" / "metrics.py").exists() and (anc / "experiment").is_dir():
            return anc
    raise SystemExit("cannot locate the remediation root; pass --root or set CRASHSEV_REMEDIATION")


PASS = FAIL = SKIP = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name} :: {detail}")


def skip(name, reason):
    global SKIP
    SKIP += 1
    print(f"[SKIP] {name} :: {reason}")


def main(argv: list[str] | None = None) -> int:
    global PASS, FAIL, SKIP
    PASS = FAIL = SKIP = 0
    ap = argparse.ArgumentParser(description=__doc__, add_help=False)
    ap.add_argument("kit", nargs="?", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--require-full", action="store_true")
    ap.add_argument("-h", "--help", action="help")
    args = ap.parse_args(argv)

    aca = _resolve_root(args.root)          # <remediation>
    top = aca.parent                        # package top (repo root or extracted canonical/)
    run = aca / "runs" / "final_8af9d5bc23d8"
    exp = aca / "experiment"
    print(f"[root] remediation = {aca}")
    print(f"[root] package top = {top} (no path outside this directory is inspected)")

    def confined(p: Path) -> Path:
        rp = Path(p).resolve()
        if not rp.is_relative_to(top):
            raise SystemExit(f"refusing to inspect a path outside the target package: {rp}")
        return rp

    def sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(confined(p), "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()

    # 1./4./5. run-bundle artifact hashes + exact metric/bootstrap replay ------------
    have_run = (run / "manifest.json").exists()
    if not have_run:
        skip("run bundle hashes / metric replay / bootstrap replay (sections 1, 4, 5)",
             "local-only run bundle runs/final_8af9d5bc23d8 is not present in this package "
             "(per-crash predictions stay local by policy; the de-identified lossless "
             "parquet supports metric recomputation instead)")
        mj = None
    else:
        mj = json.loads((run / "manifest.json").read_text("utf-8"))
        bad = [rel for rel, want in mj["artifact_sha256"].items()
               if not (run / rel).exists() or sha256(run / rel) != want]
        check(f"run bundle: {len(mj['artifact_sha256'])} artifact hashes byte-exact", not bad, str(bad))

    # 2. freeze lock pins -----------------------------------------------------------
    def assignment_hash(df):
        pairs = sorted((str(i), str(p)) for i, p in zip(df["row_id"], df["partition"]))
        return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()

    lock_path = exp / "FROZEN.lock"
    lock = None
    if not lock_path.exists():
        skip("freeze-lock pins (development report hash; dev-side split hash)",
             "experiment/FROZEN.lock is local-only (kept out of git with the "
             "identifier-bearing split_assignment.csv); present in the working checkout, "
             "absent by policy from an extracted archive")
    else:
        lock = json.loads(confined(lock_path).read_text("utf-8"))
        check("FROZEN.lock pins development_report.json byte hash",
              sha256(exp / "development_report.json") == lock["development_report_sha256"])
        split_path = exp / "split_assignment.csv"
        if not split_path.exists():
            skip("dev-side split canonical hash",
                 "experiment/split_assignment.csv is local-only (contains real crash "
                 "identifiers; never packaged)")
        else:
            check("dev-side split canonical hash == frozen dev hash (b7fd83f5…)",
                  assignment_hash(pd.read_csv(confined(split_path))) == lock["split_sha256"])
    if have_run:
        check("run split canonical hash == manifest assignment hash (119085e9…)",
              assignment_hash(pd.read_csv(confined(run / "split_assignment.csv")))
              == mj["split"]["assignment_sha256"])
    else:
        skip("run split canonical hash", "run bundle absent (see section 1)")

    # 3. modeling-table content hash --------------------------------------------------
    mt_path = aca / "_local_data" / "modeling_table_09_12.csv"
    if not mt_path.exists():
        skip("modeling-table canonical content hash",
             "licensed modeling table absent from this package by data-handling policy "
             "(rebuildable from the licensed raw extract; see README_REPRODUCE Tier 2)")
    elif lock is None:
        skip("modeling-table canonical content hash",
             "governed data_sha256 pin unavailable (FROZEN.lock absent; see above)")
    else:
        mt = pd.read_csv(confined(mt_path), low_memory=False)
        h = hashlib.sha256(pd.util.hash_pandas_object(mt, index=False).values.tobytes()).hexdigest()
        check("modeling-table canonical content hash == governed data_sha256",
              h == lock["data_sha256"])

    # 4./5. metrics + bootstrap CIs from frozen predictions (exact float parse) -------
    if have_run:
        maj = pd.read_csv(confined(run / "predictions_majority.csv"), float_precision="round_trip")
        y = maj["y_true"].to_numpy(int)
        groups = maj["group_id"].astype(str).to_numpy()
        uniq = np.unique(groups)
        order = np.argsort(groups, kind="stable")
        sg = groups[order]
        ym = maj["y_pred"].to_numpy(int)
        for name, rec in mj["results"].items():
            df = pd.read_csv(confined(run / f"predictions_{name}.csv"),
                             float_precision="round_trip")
            yp = df["y_pred"].to_numpy(int)
            omae = float(np.mean(np.abs(y - yp)))
            acc = float(np.mean(y == yp))
            check(f"{name}: oMAE/accuracy exact vs manifest",
                  omae == rec["ordinal_mae"] and acc == rec["accuracy"],
                  f"{omae} vs {rec['ordinal_mae']}")
            if name != "majority":
                rng = np.random.default_rng(mj["config"]["seed"])
                ea, eb = np.abs(y - yp).astype(float), np.abs(y - ym).astype(float)
                diffs = np.empty(mj["config"]["bootstrap_resamples"])
                for i in range(len(diffs)):
                    s = rng.choice(uniq, size=len(uniq), replace=True)
                    rows = order[np.searchsorted(sg, s)]
                    diffs[i] = ea[rows].mean() - eb[rows].mean()
                lo, hi = float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))
                ci = mj["paired_difference_vs_baseline"][name]
                check(f"{name}: paired bootstrap CI bit-exact",
                      lo == ci["ci_low"] and hi == ci["ci_high"], f"[{lo},{hi}] vs recorded")

    # 6. git: no frozen path modified — target package's own checkout ONLY ------------
    frozen_prefixes = ("remediation/runs/", "remediation/evidence_release/",
                       "remediation/configs/", "remediation/data/",
                       "remediation/experiment/final_results.json",
                       "remediation/experiment/development_report.json",
                       "remediation/experiment/leakage_factorial",
                       "remediation/experiment/broad_sensitivity/",
                       "remediation/experiment/lowmiss_sensitivity/",
                       "remediation/experiment/decision_rule_sensitivity.md",
                       "remediation/experiment/error_analysis.md",
                       "remediation/experiment/severe_ranking.md",
                       "remediation/experiment/seed_robustness.md",
                       "remediation/experiment/target_sensitivity.md",
                       "remediation/experiment/missingness_summary.md",
                       "remediation/experiment/split_assignment.csv",
                       "remediation/experiment/FROZEN.lock",
                       "remediation/experiment/FINAL.done")
    in_repo = subprocess.run(["git", "-C", str(top), "rev-parse", "--is-inside-work-tree"],
                             capture_output=True, text=True)
    if in_repo.returncode == 0 and in_repo.stdout.strip() == "true":
        # guard: the checkout consulted must BE the target package, not an enclosing repo
        top_level = subprocess.run(["git", "-C", str(top), "rev-parse", "--show-toplevel"],
                                   capture_output=True, text=True).stdout.strip()
        if Path(top_level).resolve() != top.resolve():
            skip("git frozen-path gate",
                 f"the target package is not itself a git checkout (nearest checkout is "
                 f"{top_level}); refusing to verify against an enclosing/other repository")
        else:
            st = subprocess.run(["git", "-C", str(top), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.splitlines()
            touched = []
            for ln in st:
                if not ln.strip():
                    continue
                path = ln.strip().split(None, 1)[-1].split(" -> ")[-1].strip('"')
                if any(path.startswith(p) or p.rstrip("/") == path for p in frozen_prefixes):
                    # new (untracked) additions inside experiment/ are allowed; MODIFICATIONS are not
                    if not ln.strip().startswith("??"):
                        touched.append(ln.strip())
            check("git: no frozen path modified (untracked additions allowed)", not touched,
                  str(touched))
            mb = subprocess.run(["git", "-C", str(top), "merge-base", "HEAD", BASELINE_COMMIT],
                                capture_output=True, text=True)
            if mb.returncode != 0:
                skip("git: descends from baseline 42194a6",
                     "baseline commit not present in this checkout's history")
            else:
                check("git: revision branch descends from baseline 42194a6",
                      mb.stdout.strip() == BASELINE_COMMIT)
    else:
        skip("git frozen-path gate",
             "the target package is not a git checkout; history is verifiable via the "
             "public repository at the release tag (the gate runs and must pass "
             "at packaging time; results captured in gate_outputs.json)")

    # 7. auditor-kit internal arithmetic ------------------------------------------------
    if args.kit:
        # the kit path is an explicit caller authorization and may live outside the
        # package (it is the external auditor's own tool); it is not confined
        kit_script = Path(args.kit).resolve()
        if not kit_script.exists():
            skip("auditor kit internal checks", f"kit script not found: {kit_script}")
        else:
            r = subprocess.run([sys.executable, str(kit_script)], capture_output=True, text=True)
            try:
                summ = json.loads(r.stdout)["summary"]
                ok = r.returncode == 0 and summ["passed"] == summ["total"] and summ["total"] >= 12
                label = f"auditor kit internal_checks.py: {summ['passed']}/{summ['total']}"
            except Exception as exc:
                ok, label = False, f"auditor kit internal_checks.py (unparseable: {exc})"
            check(label, ok, r.stdout[-200:])

    print(f"\n===== frozen/locked verification: {PASS} passed, {FAIL} failed, "
          f"{SKIP} skipped =====")
    if args.require_full and SKIP:
        print("[FAIL-POLICY] --require-full set and sections were skipped")
        return 1
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
