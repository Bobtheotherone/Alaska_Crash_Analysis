"""crashsev.target_sensitivity — bounded blank-severity mapping sensitivity (TARGET-001, v4).

The single consequential target-mapping inference is **blank severity = property-damage-only**
(evidence-checked: 0.0% of blank rows carry any injury/fatality count — but that is internal
evidence, not an official codebook). This module runs the prespecified bounded sensitivity the
paper previously only *described*: rebuild the development cohort under the fail-closed
alternative (**blank → quarantined/excluded**, which is what the mapping does absent the
evidenced override) and repeat the rolling-origin development comparison for a compact model
set. It is **development-years only** — the held-out final year is never touched, so this is a
protocol-safe diagnostic, not a second governed benchmark.

Interpretation limits (stated in the output): excluding blanks removes ~2/3 of class-0 rows,
so the alternative cohort has a different size and prevalence — the comparison shows whether
the *direction* of the primary contrast (candidate vs majority baseline on ordinal MAE, under
the prespecified posterior-median rule) survives the mapping choice, not that the magnitudes
are comparable.

Run (from ``remediation/``):
    python -m crashsev.target_sensitivity --data _local_data/modeling_table_09_12.csv \
        --config configs/route_r_09_12.yml --out experiment
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cli as CLI
from . import contracts as K
from . import metrics as M
from . import models as Models
from . import preprocessing as pp
from . import target as T

PKG_ROOT = Path(__file__).resolve().parents[1]

MODELS_COMPACT = ["majority", "ordinal_median", "prior_probability",
                  "random_forest_unweighted", "ordinal_random_forest"]


def _dev_frame(data_path: str, cfg: dict, blank_excluded: bool):
    """Development-years frame + target under the chosen blank policy (dev rows only)."""
    schema = K.load_schema(); mapping = K.load_target_mapping(); ledger = K.load_ledger()
    raw = CLI.load_any(data_path)
    years = pd.to_numeric(raw[cfg["year_col"]], errors="coerce")
    final_years = [int(v) for v in cfg["final_test_years"]]
    dev_mask = (~years.isin(final_years)).to_numpy() & years.notna().to_numpy()
    dev_raw = raw.loc[dev_mask].reset_index(drop=True)
    del raw

    blank_policy = None if blank_excluded else mapping.blank_maps_to
    _, y_all, audit = T.map_severity(
        dev_raw[schema.target_col],
        text_to_kabco=mapping.text_to_kabco,
        numeric_code_map=mapping.numeric_code_map,
        quarantine_labels=mapping.quarantine_labels,
        blank_maps_to=blank_policy,
        kabco_to_ordinal=mapping.kabco_to_ordinal,
    )
    keep = y_all.notna().to_numpy()
    df = dev_raw.loc[keep].reset_index(drop=True)
    y = y_all[keep].astype(int).reset_index(drop=True)
    df, _ = CLI.clean_numeric_sentinels(df, schema)
    allowed = CLI.select_allowed_features(ledger, cfg, df)
    allowed, _ = CLI.drop_uninformative(df, allowed, schema.string_missing_tokens)
    return df[allowed].copy(), y, pd.to_numeric(df[cfg["year_col"]], errors="coerce"), audit


def _rolling_eval(X, y, years, cfg, seed=42):
    """Rolling-origin dev evaluation of the compact model set under the primary decision rule."""
    reg = Models.build_registry(include_optional=False, seed=seed)
    uniq = sorted(int(v) for v in years.dropna().unique())
    out = {}
    for name in MODELS_COMPACT:
        spec = reg[name]
        maes = []
        for i in range(1, len(uniq)):
            tr = np.where(years.isin(uniq[:i]).to_numpy())[0]
            va = np.where((years == uniq[i]).to_numpy())[0]
            ytr = y.iloc[tr]
            if ytr.nunique() < int(cfg["n_classes"]):
                continue
            num_f, cat_f, _ = pp.split_feature_types(
                X.iloc[tr], categorical_max_cardinality=cfg["categorical_max_cardinality"])
            pipe, _ = CLI.fit_model(spec, X.iloc[tr], ytr, num_f, cat_f, cfg)
            yp_arg, proba = CLI.predict_aligned(pipe, X.iloc[va], cfg["n_classes"])
            yp = CLI.decide_labels(proba, yp_arg, cfg)
            maes.append(M.ordinal_mae_from_cm(
                M.confusion_matrix_from_preds(y.iloc[va].to_numpy(), yp, cfg["n_classes"])))
        out[name] = float(np.mean(maes)) if maes else float("nan")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Blank-mapping target sensitivity (TARGET-001, dev-only).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(PKG_ROOT / "experiment"))
    args = ap.parse_args(argv)
    cfg = CLI.load_config(args.config)

    results = {}
    cohorts = {}
    for label, excluded in (("primary_blank_is_PDO", False), ("alternative_blank_excluded", True)):
        X, y, yrs, audit = _dev_frame(args.data, cfg, blank_excluded=excluded)
        present = sorted(int(v) for v in y.unique())
        cohorts[label] = {"n_usable_dev": int(len(y)),
                          "prevalence": {str(k): round(float(v), 4)
                                         for k, v in y.value_counts(normalize=True).sort_index().items()},
                          "n_quarantined_dev": int(audit.n_quarantined),
                          "classes_present": present,
                          "estimand_intact": present == list(range(int(cfg["n_classes"])))}
        if cohorts[label]["estimand_intact"]:
            results[label] = _rolling_eval(X, y, yrs, cfg)
            print(f"[target-sensitivity] {label}: n={len(y)}  " +
                  "  ".join(f"{m}={v:.4f}" for m, v in results[label].items()))
        else:
            results[label] = None
            print(f"[target-sensitivity] {label}: n={len(y)}  ESTIMAND COLLAPSES — "
                  f"classes present {present} of {cfg['n_classes']}; comparison not computable")

    lines = [
        "# Target-mapping sensitivity — blank = PDO vs blank = excluded (TARGET-001, v4)\n",
        "Development years ONLY (the held-out final year is never touched). Rolling-origin folds,",
        f"prespecified decision rule `{cfg.get('decision_rule', 'argmax')}`, feature tier "
        f"`{cfg.get('feature_tier', 'broad')}`, seed 42, compact model set.\n",
        "| mapping | n usable (dev) | classes present | prevalence 0/1/2 | "
        + " | ".join(MODELS_COMPACT) + " |",
        "|---|---|---|---|" + "---|" * len(MODELS_COMPACT),
    ]
    for label in results:
        c = cohorts[label]
        prev = "/".join(str(c["prevalence"].get(k, 0.0)) for k in ("0", "1", "2"))
        if results[label] is None:
            vals = " | ".join(["—"] * len(MODELS_COMPACT))
        else:
            vals = " | ".join(f"{results[label][m]:.4f}" for m in MODELS_COMPACT)
        lines.append(f"| {label} | {c['n_usable_dev']:,} | {c['classes_present']} | {prev} | {vals} |")
    prim, alt = results["primary_blank_is_PDO"], results["alternative_blank_excluded"]
    lines.append("")
    if alt is None:
        c = cohorts["alternative_blank_excluded"]
        lines += [
            "**Finding — the blank policy CONSTITUTES class 0 in this extract.** Under the",
            "fail-closed alternative (blank → excluded), the usable development cohort keeps only",
            f"{c['n_usable_dev']:,} rows with classes {c['classes_present']} present "
            f"(prevalence {'/'.join(str(c['prevalence'].get(k, 0.0)) for k in ('0', '1', '2'))}):",
            "**class 0 (none/PDO) exists in this extract almost entirely through the blank→PDO",
            "inference**, so the alternative mapping does not yield a comparable 3-class study —",
            "it changes the estimand itself (a 2-class minor-vs-severe problem on a third of the",
            "rows). The prespecified 'direction check' is therefore **not computable**, and the",
            "correct conclusion is stronger than a robustness pass or fail: the blank=PDO decision",
            "is not a marginal coding choice but the definition of the majority class, which makes",
            "obtaining the official Alaska codebook (Gate 3, `PROVENANCE_ACQUISITION_PLAN.md`)",
            "material to the study's construct validity — exactly as the internal evidence check",
            "(0.0% injury/fatality contamination of blanks) already suggested, now with the",
            "consequence quantified.",
        ]
    else:
        d_prim = prim["ordinal_random_forest"] - prim["majority"]
        d_alt = alt["ordinal_random_forest"] - alt["majority"]
        lines.append(
            f"**Direction check (ordinal RF minus majority, dev rolling mean):** "
            f"primary mapping {d_prim:+.4f}; blank-excluded mapping {d_alt:+.4f} — "
            + ("the candidate improves on the baseline under BOTH mappings."
               if (d_prim < 0 and d_alt < 0) else
               "the contrast direction DEPENDS on the mapping — a material construct-validity flag."))
    lines += [
        "",
        f"*Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by "
        f"`crashsev/target_sensitivity.py`; development data only; no per-crash rows.*",
    ]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "target_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[target-sensitivity] wrote {out / 'target_sensitivity.md'}")


if __name__ == "__main__":
    main()
