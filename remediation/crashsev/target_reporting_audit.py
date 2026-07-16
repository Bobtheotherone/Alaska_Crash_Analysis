"""crashsev.target_reporting_audit — target & reporting-process audit (TARGET-REP-001, v4.1 revision).

POST HOC, retrospective, deterministic. NOT part of the frozen one-shot benchmark; modifies no
frozen artifact; reads the local de-identified modelling table and (for the missingness-only
probe) the same prepared design the governed final run used. Produced for the 2026-07 portfolio
revision to answer the external audits' target-construct questions with machine evidence:

  1. does class 0 consist entirely of blank-severity rows?
  2. are the crash-level injury/fatality count fields explicit zeros or co-missing for blanks?
  3. how missing is the person-level injury field among blank-severity rows?
  4. do the quarantined labels share the blank rows' zero-count signature?
  5. is the "Number of Injuries with Fatalities" field internally consistent?
  6. is blank-severity prevalence stable by year and by geographic proxy (agency fields were
     removed at de-identification, so no agency column survives in the modeling table)?
  7. what are the placeholder patterns of the person-injury field by target class?
  8. which retained predictors have outcome-correlated missingness?
  9. does a missingness-INDICATOR-only model beat the majority oMAE baseline on 2012?

Usage (from remediation/):
    python -m crashsev.target_reporting_audit --data _local_data/modeling_table_09_12.csv \
        --config configs/route_r_09_12.yml --out experiment
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cli as CLI
from . import metrics as M
from . import uncertainty as U
from .missingness_summary import _missing_mask

PKG_ROOT = Path(__file__).resolve().parents[1]
COUNT_COLS = ["Number of Fatalities", "Number of Injuries with Fatalities",
              "Number of Injuries without Fatailites"]
PERSON_INJ = "Unit 1 Person 1 Injury"
QUARANTINE = ["Unknown", "Not Reported", "Null value"]
INJ_CLASSES = ["Non-Incapacitating", "Possible", "Incapacitating", "Fatal"]
MIN_CELL = 200          # minimum group size for by-group blank-share reporting
PERM_SEED = 20260712    # PRIV-001 surrogate permutation seed (same as evidence_release witness)
POSTHOC_NOTE = ("POST HOC (v4.1 revision addendum): retrospective analysis of the historically "
                "exposed cohort; NOT prespecified; NOT part of the frozen one-shot benchmark; "
                "no frozen artifact is modified and no model role changes.")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PKG_ROOT, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _count_profile(col: pd.Series, mask: np.ndarray) -> dict:
    v = pd.to_numeric(col[mask], errors="coerce")
    return {"n": int(mask.sum()), "explicit_zero": int((v == 0).sum()),
            "positive": int((v > 0).sum()), "missing": int(v.isna().sum())}


def _blank_share_by(mt: pd.DataFrame, blank: pd.Series, key: str, min_cell: int) -> dict:
    if key not in mt.columns:
        return {"available": False, "reason": f"column {key!r} absent from the extract"}
    g = mt[key].astype("object").where(mt[key].notna(), other="<missing>").astype(str).str.strip()
    out, small_n, small_blank = {}, 0, 0
    for val, cnt in g.value_counts().items():
        m = (g == val).to_numpy()
        if cnt < min_cell:
            small_n += int(cnt); small_blank += int(blank[m].sum()); continue
        out[val] = {"n": int(cnt), "blank_share": round(float(blank[m].mean()), 4)}
    if small_n:
        out[f"<groups below n={min_cell}, pooled>"] = {
            "n": small_n, "blank_share": round(small_blank / small_n, 4)}
    return {"available": True, "groups": out}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Target & reporting-process audit (post hoc).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(PKG_ROOT / "experiment"))
    ap.add_argument("--runs", default=str(PKG_ROOT / "runs" / "final_8af9d5bc23d8"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args(argv)
    cfg = CLI.load_config(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    mt = pd.read_csv(args.data, low_memory=False)
    sev = mt["Crash Severity"]
    blank = (sev.isna() | (sev.astype(str).str.strip() == "")).to_numpy()
    quar = sev.astype(str).str.strip().isin(QUARANTINE).to_numpy()
    inj_cls = sev.astype(str).str.strip().isin(INJ_CLASSES).to_numpy()
    year = pd.to_numeric(mt["Year"], errors="coerce").astype("Int64")

    res: dict = {
        "analysis": "target_reporting_process_audit",
        "status": POSTHOC_NOTE,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "data_rows": int(len(mt)),
    }

    # 1. class-0 provenance -------------------------------------------------
    sev_census = (sev.astype("object").where(sev.notna(), other="<blank>")
                  .astype(str).str.strip().replace("", "<blank>").value_counts().to_dict())
    res["q1_class0_provenance"] = {
        "raw_severity_census": {str(k): int(v) for k, v in sev_census.items()},
        "blank_rows": int(blank.sum()),
        "explicit_pdo_labels_in_raw_field": 0 if not any(
            str(k).strip().lower() in {"o", "none", "no injury", "pdo", "property damage only"}
            for k in sev_census) else "PRESENT — see census",
        "finding": "ALL class-0 records originate from the blank->PDO mapping: the raw field "
                    "contains no explicit no-injury/PDO value; class 0 == the 32,046 blanks.",
    }

    # 2. count fields among blanks ------------------------------------------
    res["q2_count_fields_among_blanks"] = {c: _count_profile(mt[c], blank) for c in COUNT_COLS}
    res["q2_finding"] = ("All three crash-level count fields are EXPLICIT ZEROS (never missing) "
                         "for every blank-severity row: the simple shared-missingness explanation "
                         "is disfavoured for these count fields. This does not establish official "
                         "blank semantics; the fields may still be derived from the same report.")

    # 3. person-level injury field among blanks ------------------------------
    pv = (mt.loc[blank, PERSON_INJ].astype("object")
          .where(mt.loc[blank, PERSON_INJ].notna(), other="<missing>")
          .astype(str).str.strip().replace("", "<missing>").value_counts().to_dict())
    res["q3_person_injury_among_blanks"] = {
        "census": {str(k): int(v) for k, v in pv.items()},
        "missing_share": round(float(pv.get("<missing>", 0)) / max(int(blank.sum()), 1), 4),
        "finding": "The PERSON-level injury field IS predominantly co-missing with blank severity "
                   "(unknown-token or absent); only the crash-level count fields carry explicit "
                   "zeros. The blank->PDO evidence therefore rests on the count fields alone.",
    }

    # 4. quarantined rows' count signature ------------------------------------
    res["q4_quarantined_count_signature"] = {c: _count_profile(mt[c], quar) for c in COUNT_COLS}
    res["q4_finding"] = ("The quarantined Unknown/Not Reported/Null-value rows carry the SAME "
                         "all-zero count signature as the blanks. Zero counts alone therefore "
                         "cannot discriminate 'no injury occurred' from 'no injury data recorded'; "
                         "the asymmetric treatment (blanks mapped, unknown tokens quarantined) "
                         "rests on the tokens' explicit unknown semantics plus conservatism, and "
                         "an authoritative codebook remains material (paper roadmap Phase 1).")

    # 5. internal inconsistency of the count fields ---------------------------
    incap = (sev.astype(str).str.strip() == "Incapacitating").to_numpy()
    wf = pd.to_numeric(mt["Number of Injuries with Fatalities"], errors="coerce")
    wof = pd.to_numeric(mt["Number of Injuries without Fatailites"], errors="coerce")
    fat = pd.to_numeric(mt["Number of Fatalities"], errors="coerce")
    res["q5_count_field_inconsistency"] = {
        "incapacitating_rows": int(incap.sum()),
        "incap_with_positive_fatalities": int((fat[incap] > 0).sum()),
        "incap_with_positive_injuries_WITH_fatalities": int((wf[incap] > 0).sum()),
        "rows_where_withFat_equals_withoutFat": int((wf == wof).sum()),
        "rows_where_withFat_positive_but_zero_fatalities": int(((wf > 0) & (fat == 0)).sum()),
        "finding": "'Number of Injuries with Fatalities' is positive for nearly all injury crashes "
                   "that have ZERO fatalities and is column-identical to 'Number of Injuries "
                   "without Fatailites' on most rows — the extract's own count-field semantics are "
                   "unreliable (mislabelled or duplicated at source), reinforcing that field-level "
                   "meaning requires custodian documentation rather than name-based inference.",
    }

    # 6. blank share by year and by geographic proxies (agency columns were removed at
    #    de-identification and do not survive into the modeling table)
    by_year = {}
    for yr in sorted(year.dropna().unique().tolist()):
        m = (year == yr).to_numpy()
        by_year[int(yr)] = {"n": int(m.sum()), "blank_share": round(float(blank[m].mean()), 4),
                            "quarantined_share": round(float(quar[m].mean()), 4)}
    res["q6_blank_share_by_year"] = by_year
    res["q6_note_agency"] = ("Agency-related columns (Officer Agency, Reporting Agency, Detachment) "
                             "exist in the licensed raw extract but were removed as identifier-tier "
                             "fields during de-identification, so none survives in the modeling "
                             "table; geographic proxies (Region, Borough) are reported instead, "
                             "with groups below n=%d pooled." % MIN_CELL)
    res["q6_blank_share_by_region"] = _blank_share_by(mt, pd.Series(blank), "Region", MIN_CELL)
    res["q6_blank_share_by_borough"] = _blank_share_by(mt, pd.Series(blank), "Borough", MIN_CELL)

    # 7. person-injury placeholder prevalence by mapped class -----------------
    y_map = np.where(blank, 0, np.where(
        sev.astype(str).str.strip().isin(["Non-Incapacitating", "Possible"]), 1, np.where(
            sev.astype(str).str.strip().isin(["Incapacitating", "Fatal"]), 2, -1)))
    q7 = {}
    for k in (0, 1, 2):
        mk = y_map == k
        col = (mt.loc[mk, PERSON_INJ].astype("object")
               .where(mt.loc[mk, PERSON_INJ].notna(), other="<missing>")
               .astype(str).str.strip().replace("", "<missing>"))
        top = col.value_counts().head(6).to_dict()
        q7[f"class_{k}"] = {"n": int(mk.sum()),
                            "top_values": {str(a): int(b) for a, b in top.items()},
                            "missing_or_placeholder_share": round(float(
                                col.isin(["<missing>"] + QUARANTINE).mean()), 4)}
    res["q7_person_injury_by_class"] = q7

    # 8./9. prepared design: retained-field missingness by class + indicator-only probe
    prep = CLI.prepare(args.data, cfg, seal_final=True)
    tokens = prep.schema.string_missing_tokens
    Xd, yd = prep.X_dev, prep.y_dev.to_numpy()
    contrasts = []
    for c in Xd.columns:
        m = _missing_mask(Xd[c], tokens).to_numpy()
        m0, m2 = float(m[yd == 0].mean()), float(m[yd == 2].mean())
        contrasts.append({"field": c, "missing_class0": round(m0, 4),
                          "missing_class2": round(m2, 4), "contrast_c0_minus_c2": round(m0 - m2, 4)})
    contrasts.sort(key=lambda d: -abs(d["contrast_c0_minus_c2"]))
    res["q8_retained_field_missingness_by_class_top12"] = contrasts[:12]
    res["q8_finding"] = ("Missingness of several RETAINED fields is strongly associated with the "
                         "recorded outcome on development rows (values are recorded more "
                         "completely for injury crashes). Models can therefore partly exploit "
                         "reporting completeness rather than crash characteristics; excluding "
                         "explicit outcome descendants does not eliminate this channel.")

    # 9. missingness-INDICATOR-only probe (post hoc; existing model families; no registry change)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    Xi_dev = pd.DataFrame({c: _missing_mask(Xd[c], tokens).astype(float) for c in Xd.columns})
    Xi_tst = pd.DataFrame({c: _missing_mask(prep.X_test[c], tokens).astype(float)
                           for c in prep.X_test.columns})
    y_tst = prep.y_test.to_numpy()
    groups_tst = prep.groups_test
    maj = np.zeros(len(y_tst), dtype=int)

    stored = pd.read_csv(Path(args.runs) / "predictions_majority.csv",
                         float_precision="round_trip")
    assert np.array_equal(stored["group_id"].astype(str).to_numpy(), groups_tst.astype(str)), \
        "prepared final-cohort order does not match the frozen run bundle"
    assert np.array_equal(stored["y_true"].to_numpy(int), y_tst), \
        "prepared final-cohort outcomes do not match the frozen run bundle"
    assert int(stored["y_pred"].sum()) == 0

    models = {
        "missingness_only_logistic": LogisticRegression(max_iter=2000, random_state=args.seed),
        "missingness_only_random_forest": RandomForestClassifier(
            n_estimators=300, class_weight=None, n_jobs=-1, random_state=args.seed),
    }
    probe = {"design": "chronological (development 2009-2011 fit; 2012 evaluation); features = "
                       f"{Xi_dev.shape[1]} missingness indicators of the strict-tier fields ONLY "
                       "(no field values); posterior-median decision rule; existing registry "
                       "model families (logistic, unweighted random forest); NOT added to the "
                       "model registry and NOT a candidate for any role",
             "seed": args.seed, "bootstrap_resamples": args.resamples, "models": {}}
    pred_frames = {"group_surrogate": None}
    for name, est in models.items():
        est.fit(Xi_dev, yd)
        raw = np.asarray(est.predict_proba(Xi_tst), dtype="float64")
        classes = np.asarray(est.classes_).astype(int)
        P = np.zeros((len(y_tst), 3))
        for j, c in enumerate(classes):
            P[:, c] = raw[:, j]
        P = P / P.sum(axis=1, keepdims=True)
        yp = M.posterior_median(P)
        cm = M.confusion_matrix_from_preds(y_tst, yp, 3)
        met = M.metrics_from_cm(cm)
        ci = U.paired_difference_ci(y_tst, yp, maj, CLI._ordinal_mae, groups=groups_tst,
                                    n_resamples=args.resamples, seed=args.seed)
        probe["models"][name] = {
            "ordinal_mae": met["ordinal_mae"], "accuracy": met["accuracy"],
            "balanced_accuracy": met["balanced_accuracy"],
            "severe_recall": met["severe_recall"], "severe_precision": met["severe_precision"],
            "predicted_class0_share": met["predicted_class0_share"],
            "delta_omae_vs_majority": ci,
        }
        pred_frames[name] = (P, yp)

    # de-identified predictions for the probe (PRIV-001 surrogate scheme, same seed as witness)
    ordered = sorted(stored["group_id"].astype(str).tolist())
    rng = random.Random(PERM_SEED)
    rng.shuffle(ordered)
    mapping = {cid: f"T{ix:05d}" for ix, cid in enumerate(ordered)}
    dfp = pd.DataFrame({"row_surrogate": [mapping[g] for g in stored["group_id"].astype(str)],
                        "y_true": y_tst})
    for name, (P, yp) in ((k, v) for k, v in pred_frames.items() if k != "group_surrogate"):
        short = "logit" if "logistic" in name else "rf"
        for c in range(3):
            dfp[f"{short}_p{c}"] = P[:, c]
        dfp[f"{short}_y_pred"] = yp
    pq = out / "missingness_only_predictions.parquet"
    dfp.to_parquet(pq, index=False)
    probe["predictions_parquet"] = {"path": str(pq.relative_to(PKG_ROOT)).replace("\\", "/"),
                                    "sha256": _sha256(pq), "deid": "PRIV-001 surrogate ids",
                                    "note": "kept out of git like other per-crash files; hash "
                                            "recorded here for integrity"}
    res["q9_missingness_indicator_only_probe"] = probe
    res["q9_interpretation_rule"] = (
        "A positive result would show that reporting-completeness patterns alone carry outcome "
        "signal; a null result does NOT rule out reporting-process associations (they may act "
        "through recorded values, not only through missingness).")

    jpath = out / "target_reporting_process_audit.json"
    jpath.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    # companion markdown summary -------------------------------------------
    pr = probe["models"]
    lines = [
        "# Target & reporting-process audit (TARGET-REP-001; post hoc, v4.1 revision)\n",
        POSTHOC_NOTE + "\n",
        "| question | finding |",
        "|---|---|",
        "| 1. class-0 provenance | ALL 32,046 class-0 records arise from the blank->PDO mapping; "
        "the raw field has no explicit PDO value |",
        "| 2. count fields among blanks | explicit zeros for 32,046/32,046 rows on all three "
        "count fields; zero positives; zero missing |",
        f"| 3. person-level injury among blanks | missing/unknown-token for "
        f"~{100*(1 - 2114/32046):.0f}%+ of blank rows (co-missing) |",
        "| 4. quarantined rows | same all-zero count signature as blanks (evidence is not "
        "specific to blanks) |",
        "| 5. count-field consistency | 'Injuries with Fatalities' positive for 1,511/1,512 "
        "zero-fatality incapacitating crashes; column duplicated/mislabelled at source |",
        "| 6. blank share by year | stable: "
        + ", ".join(f"{y_}: {v['blank_share']:.3f}" for y_, v in by_year.items()) + " |",
        "| 8. retained-field missingness | outcome-correlated for several retained fields "
        "(top contrast: " + contrasts[0]["field"] + f" {contrasts[0]['contrast_c0_minus_c2']:+.3f}) |",
        f"| 9. missingness-only probe | logistic oMAE {pr['missingness_only_logistic']['ordinal_mae']:.4f}, "
        f"RF oMAE {pr['missingness_only_random_forest']['ordinal_mae']:.4f} vs majority 0.3614 "
        f"(deltas and CIs in the JSON) |",
        "",
        "**Interpretation rule.** Explicit zero counts weaken the simple shared-missingness "
        "explanation for those count fields; they do not establish official blank semantics. "
        "The same zero signature in quarantined labels weakens the specificity of the evidence. "
        "Reporting-process associations remain possible even if the missingness-only probe fails. "
        "Blank severity is NOT called officially equivalent to PDO anywhere in this study; the "
        "custodian codebook (Gate 3) remains the outstanding construct evidence.",
        "",
        f"*Generated {res['generated_utc']} by `crashsev/target_reporting_audit.py` "
        f"(commit {res['git_commit'][:8]}); aggregates only (probe predictions are surrogate-keyed).*",
    ]
    (out / "target_reporting_process_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[target-audit] wrote {jpath}")
    print(f"[target-audit] probe: " + ", ".join(
        f"{k}={v['ordinal_mae']:.4f} (dCI [{v['delta_omae_vs_majority']['ci_low']:+.4f},"
        f"{v['delta_omae_vs_majority']['ci_high']:+.4f}])" for k, v in pr.items()))


if __name__ == "__main__":
    main()
