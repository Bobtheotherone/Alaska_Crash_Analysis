# Reproducing the study

Two levels of reproduction: (A) the code, tests, and re-analysis with **no raw data**, and
(B) the full empirical result with a **licensed copy of the raw extract**.

## 0. Environment

```bash
cd remediation
python -m pip install -r requirements-lock.txt   # minimal DIRECT pins - not a resolved transitive lock (Py 3.12-3.13; the xgboost==3.3.0 pin requires >=3.12)
python -m pip install -e .
```

CI workflows are defined (`.github/workflows/crashsev-ci.yml` — test matrix on
Windows/Linux across Python 3.12/3.13 — and `.github/workflows/verify-portfolio.yml`
— the no-license verification tier); their hosted execution status is visible on the
public repository.

## A. Reproduce without raw data (anyone)

```bash
python -m pytest tests/ -q                        # failure-mode + governance test suite (66 at
                                                  # v4, incl. the structural-isolation poison
                                                  # sentinel; the authoritative count is the
                                                  # packaged tests_and_ci/pytest_output.txt,
                                                  # regenerated at packaging time)
python -m crashsev.reanalysis                     # re-analysis of the paper's 4 matrices
python -m paper.make_final_figures                # 4 final-result figures from experiment/final_results.json
python -m paper.render_paper                      # rebuild final_paper.pdf (all 8 figures)
```

* `tests/` proves the contracts reject impostors/malformed mappings, the split reconciles and
  quarantines missing years, the metric contracts hold, the ordered-logit gradient is correct,
  seeds propagate, bundles are immutable, and sentinels are cleaned.
* `python -m crashsev.reanalysis` regenerates `reanalysis/reanalysis_table.md` and the three
  re-analysis figures from `data/confusion_matrices_from_paper.json` (no raw data needed).
* Every committed number in `data/*.md`, `data/*.json`, and `experiment/*.json` is an aggregate;
  none contains a per-crash row.

> Note (local Windows): if `tmp_path`-based tests error with a temp-permission or long-path
> problem, pass a short writable base, e.g. `python -m pytest tests/ -q --basetemp=C:/ptmp`.
> CI runners are unaffected.

## B. Reproduce the empirical result (with the licensed raw extract)

You need a lawful copy of `Crash Level 09-12 (1).xlsx` (see `research/DATA_LICENSE_NOTE.md`).
Point the environment variable / `--data` at it. The raw file and everything derived from it
per-crash stay **local** (git-ignored); only aggregates are committed.

```bash
# 1. Gate-0 contract check (rejects an impostor; accepts the real extract)
python -m crashsev.cli validate-data --data "<path>/Crash Level 09-12 (1).xlsx"

# 2. Build the de-identified local modelling table (drops coordinates/ids; git-ignored)
python -m crashsev.build_modeling_table --data "<path>/Crash Level 09-12 (1).xlsx"

# 3. Development phase: dev-CV model selection. v4 (configs/route_r_09_12.yml) runs under
#    STRUCTURAL outcome isolation: final-year outcome values are never validated, mapped,
#    audited, or serialized in this phase (GOV-001 v4). The v3 protocol config
#    (configs/route_a_09_12.yml) is retained for the historical chain.
python -m crashsev.cli develop --data _local_data/modeling_table_09_12.csv \
    --config configs/route_r_09_12.yml --out experiment

# 4. Commit any SOURCE changes (governance needs a clean source tree; the experiment/ and
#    runs/ outputs are exempt from the dirty check)
git add -A && git commit -m "reproduction run"

# 5. Freeze, then the ONE final evaluation on the held-out, out-of-time 2012 test
#    (the v4 freeze pins the outcome-free DEVELOPMENT-side assignment hash)
python -m crashsev.cli freeze-experiment --out experiment
python -m crashsev.cli evaluate-final --data _local_data/modeling_table_09_12.csv \
    --config configs/route_r_09_12.yml --out experiment --out-runs runs

# 6. (optional) the controlled leakage factorial (development data only)
python -m crashsev.leakage_factorial --data _local_data/modeling_table_09_12.csv \
    --config configs/route_a_09_12.yml --out experiment
```

Outputs:
* `experiment/final_results.json` — committable aggregate (metrics, CIs, calibration, model cards).
* `runs/final_<hash>/` — the immutable, hashed run bundle (per-crash predictions kept **local**).
* `experiment/FINAL.done` — one-shot marker; a second `evaluate-final` on the same freeze refuses.

### Determinism / governance notes

* Same code + same data + same config ⇒ identical split hash and identical results.
* `evaluate-final` refuses a dirty **source** tree, an unfrozen experiment, an input whose hash
  differs from `FROZEN.lock`, or a second run against the same freeze.
* v4 structural isolation (GOV-001): `develop` partitions rows by YEAR before any target
  interpretation - final-year outcomes are never validated, mapped, audited, or serialized in
  that phase, and its artifacts cannot yield them by derivation. Precision note: the shared
  table IS loaded and content-hashed whole (opaque integrity step) before partitioning, so the
  guarantee is semantic/analytic, not byte-level non-access. The final-test feature/label
  matrices are materialised only inside `evaluate-final`.
* The governed `data_sha256` hashes dataframe *content* (`pd.util.hash_pandas_object`), not CSV file
  bytes, so it is stable across environments/serializations; `evaluate-final` keys the `FROZEN.lock`
  check on it, so a freshly rebuilt modelling table passes even if its file bytes differ.

## Self-reproduction (verified)

This procedure was executed as a **self-reproduction** (same project; not an independent third
party) from a fresh `git archive` extract and confirmed
**bit-exact** *(v3-era record; the suite has since grown — see the packaged
`tests_and_ci/pytest_output.txt` and `research_audit/RELEASE_STATE.md` for the current count)*:
54/54 tests pass at that revision; every figure and re-analysis table regenerates byte-identically;
Gate-0 accepts the real extract (exit 0) and rejects an impostor (exit 2); and a full
`develop → freeze → evaluate-final` re-run reproduces the held-out 2012-test metrics **and** the seeded
2000-resample bootstrap CIs to a maximum absolute difference of **0.000e+00**. Full evidence table:
`research/REPRODUCTION_LOG.md`.
