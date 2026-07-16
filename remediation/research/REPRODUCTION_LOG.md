# Self-reproduction log (same project; not an independent third party)

A record of a **self-reproduction** (same project; not an independent third party) performed from a **fresh extract of the committed tree**
(`git archive HEAD` → clean directory, no `.git`, no untracked files, no ignored `_local_data/`,
no `__pycache__`). It substantiates the reproducibility claims in `REPRODUCE.md` and closes the
"re-run" item previously listed in `ASSESSMENT.md` — the re-run is a self-reproduction by the same
project, not an independent third-party replication.

Environment: Windows 11, Python 3.13; package versions per `PROVENANCE.md` / `requirements-lock.txt`.

## 1. Package integrity (no raw data)

| Check | Result |
|---|---|
| `git fsck --full` | clean (no errors/corruption) |
| Tracked bytecode / caches / local data | none (`git ls-files` shows no `__pycache__`, `*.pyc`, `_local_data/`, `*.xlsx`) |
| Coordinate / PII scan of tracked text | no `Latitude`/`Longitude` decimal pairs, no lat,lon patterns |
| `python -m pytest tests/` (from clean extract) | **54 passed** (incl. the second-round GOV/WGT/METRIC/SPLIT/AUTH guard tests) |

## 2. Aggregate-derived artifacts are byte-identical

Regenerated in the clean extract from committed aggregates only (no raw data) and compared by
SHA-256 against the committed files:

| Regenerated from | Artifacts | Result |
|---|---|---|
| `experiment/final_results.json` (`paper.make_final_figures`) | 4 final-result figures | **4/4 byte-identical** |
| `data/confusion_matrices_from_paper.json` (`crashsev.reanalysis`) | 3 re-analysis figures + `reanalysis_metrics.csv` + `reanalysis_table.md` | **5/5 byte-identical** |
| `paper.render_paper` | `final_paper.pdf` | valid PDF, `err=0`, **7/7 figures embedded**, 15 pages (content-identical; the byline date is now pinned to the canonical version date, so the render is deterministic) |

## 3. Data-dependent Gate 0 (with the licensed extract)

| Check (run from the clean extract) | Result |
|---|---|
| `crashsev.cli validate-data` on the real `Crash Level 09-12 (1).xlsx` | `GATE 0: PASS`, exit **0** |
| `crashsev.cli validate-data` on a 3-column impostor CSV | `GATE 0: FAIL`, exit **2** (nonzero — automation-safe) |
| Rebuild de-identified modelling table from raw → `data/modeling_table_audit.md` | **byte-identical** to committed (deterministic de-identification) |
| Per-crash modelling table | stays in git-ignored `_local_data/` — confirmed `git check-ignore` |

> **Scope note (data-restricted).** §3–§4 require the licensed extract, which is **not** included in
> the handoff (per `DATA_AUTHORITY_AND_ACCESS.md`); these rows document reproduction performed in the
> project workspace while the extract was available. The raw `.xlsx` is no longer retained in the
> workspace at finalization, so the raw-file Gate-0 rows are a prior-round record and are not
> re-runnable from the shipped package. An external reviewer reproduces the *packaged* relationships
> instead via `09_VERIFY_HANDOFF.py` (integrity + headline ordinal-MAE recomputed from the shipped
> de-identified 2012 predictions).

## 4. Full governed pipeline reproduces the held-out 2012-test result

Re-ran `develop → freeze-experiment → evaluate-final` from the committed code (HEAD) on the licensed
extract, into scratch output dirs (committed artifacts untouched):

| Governed quantity | Committed run | Re-run | Match |
|---|---|---|---|
| `data_sha256` (canonical content hash, `pd.util.hash_pandas_object`) | `059559cd…` | `059559cd…` | ✅ |
| `split_sha256` | `119085e9…` | `119085e9…` | ✅ |
| config / schema / target / ledger hashes | (4 hashes) | identical | ✅ |
| split counts (dev / test / quarantined) | 35,214 / 11,630 / 3,699 | 35,214 / 11,630 / 3,699 | ✅ |
| selected primary comparator baseline | majority | majority | ✅ |
| Headline final-test metrics | see below | see below | ✅ (see §5) |

> **Note on hashes.** The governed `data_sha256` hashes *dataframe content* (via
> `pd.util.hash_pandas_object`), not CSV file bytes, so it is stable across pandas serialization
> differences. The raw modelling-table **file** hash is therefore *not* used for governance and may
> differ between environments without affecting reproduction; `evaluate-final` correctly keys the
> `FROZEN.lock` check on the content hash, so a reviewer's freshly built table passes.

## 5. Final-test metric reproduction — **exact**

Comparing the committed `experiment/final_results.json` against the re-run's `final_results.json`
across all **14 models** (the 12 primary models plus the two matched unweighted ablation controls,
WGT-001):

| Comparison | Max absolute difference |
|---|---|
| Every point metric (ordinal MAE, severe P/R, accuracy, QWK, log loss, Brier, RPS, balanced acc, within-one, ECE) | **0.000e+00** |
| Every paired difference-vs-majority **and** its 95% crash-level bootstrap CI (2000 resamples) | **0.000e+00** |
| Calibration ECE (raw and development-fit) | **0.000e+00** |

Representative values (committed == re-run): ordinal RF ordinal MAE **0.3310**, severe recall 0.2778,
log loss 0.6802; random forest 0.3436; majority 0.3614 / log loss 11.1457. Paired Δ vs majority:
ordinal RF **−0.03035** CI [−0.03689, −0.02356]; random forest −0.01780 CI [−0.02674, −0.00843] —
both identical to the committed values and both excluding 0.

**Conclusion.** The governed pipeline is **bit-exactly reproducible** under self-reproduction (same
project; not an independent third-party replication): from a fresh state on the
licensed extract, `develop → freeze-experiment → evaluate-final` reproduces the identical content
hash, split, per-model metrics, and seeded bootstrap CIs. The seeded bootstrap makes even the
uncertainty intervals deterministic. Nothing in the committed result depends on un-pinned state.

*Second-round self-reproduction performed at HEAD `e17c263` on 2026-07-10 (re-run bundle
`final_593077e6279f`, into scratch dirs; committed artifacts untouched). The committed authoritative
result was generated at commit `e55d6f1` (an ancestor of the release commit `e17c263`, which carries
these exact artifacts). The two runs are **bit-identical across all 14 models**: the maximum absolute
difference is `0.000e+00` for every point metric, every paired difference-vs-majority and its 95%
crash-level bootstrap CI, and every calibration ECE; `data_sha256` (`059559cd…`) and `split_sha256`
(`119085e9…`) match exactly. This supersedes the first-round reproduction (re-run `bb5a471` vs
committed `bd8d042`), which predated the calibration-selection fix and the two matched ablation
controls; the headline was unchanged across both.*

---

*Third-round (v3.1) addendum, 2026-07-11.* Three corrections/extensions to the record above:
(1) **Raw workbook located and byte-hashed.** The scope note above said the raw `.xlsx` was "no
longer retained in the workspace"; the finalization-time scan had looked only at the backup root —
the file sits in the `2025\` subdirectory. Byte identity (value withheld from public release — restricted reproduction log; 25,391,525 bytes) is now
recorded, and rebuilding the modelling table from that byte-hashed file reproduces the committed
`_local_data` CSV **byte-identically** and the governed `data_sha256 = 059559cd…` **exactly**
(`DATA_AUTHORITY_AND_ACCESS.md` §2). (2) **Re-analysis regenerated under the METRIC-001
convention.** The committed `reanalysis/` tables above predated the NaN-for-undefined metric
semantics (majority macro-F1 / severe precision were zero-filled); they are regenerated at v3.1
(majority row now `nan`/"—"), so §2's byte-identical claim holds again against the *current*
committed artifacts. Model rows and all three figures are unchanged. (3) **Factorial report
regenerated with matched simple effects + interactions** (FAC-001): the rerun reproduced all
8 × 3 cell means **bit-identically** (max abs diff 0.000e+00 vs the committed cells); only the
report structure changed. The witness for the de-identification transform of the final run bundle
is committed at `evidence_release/final_f27613102c96/DEID_TRANSFORM.md` (BUNDLE-CRLF-001 disclosed
there).

---

## v4 corrected-protocol self-reproduction (2026-07-12) — bit-exact

From a fresh `git archive` extract of the release tree (no `.git`, no caches; the licensed local
modelling table supplied per Route B), the complete v4 governed pipeline was re-executed:
`pytest` (**66 passed**) → `develop` (structural outcome isolation) → `freeze-experiment`
(dev-side assignment hash) → `evaluate-final`. Comparing the re-run `final_results.json` against
the committed v4 aggregate across **every** results field, paired difference + 95% bootstrap CI,
calibration block, split block, target audit, and the development-assignment hash (excluding only
timestamps/git/environment blocks): **0 differing leaves; maximum absolute numeric difference
0.000e+00.** The v4 result (ordinal RF 0.3469 vs majority 0.3614; Δ −0.0144 [−0.0231, −0.0063])
is therefore bit-exactly self-reproducible, including its seeded 2000-resample intervals. This
remains **self-reproduction** (same project, same machine) — not independent reproduction.
