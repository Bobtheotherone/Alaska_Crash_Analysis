"""Generate the hyperparameter ledger from the model registry and the frozen run config.

Instantiates ``crashsev.models.build_registry`` with the frozen run's seed, extracts each
estimator's constructor parameters (via ``get_params`` where available), and records the
shared preprocessing/protocol constants from ``experiment/final_results.json``'s embedded
config. Emits ``experiment/hyperparameter_ledger.json``.

The point of this artifact (audit-completion directive §6.7): the manuscript's
hyperparameter appendix must be generated from configuration and code, not from memory.
All configurations are FIXED — the corrected protocol runs no hyperparameter search; the
development cross-validation selects among these fixed configurations only.

Deterministic: no timestamps; provenance is the frozen config hash. Usage:
    python tools/gen_hyperparameter_ledger.py [--check]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REM))

import inspect  # noqa: E402

from crashsev import metrics as Me  # noqa: E402
from crashsev import models as Mo  # noqa: E402

LOG_LOSS_EPS = inspect.signature(Me.multiclass_log_loss).parameters["eps"].default

OUT = REM / "experiment" / "hyperparameter_ledger.json"
FR = json.loads((REM / "experiment" / "final_results.json").read_text("utf-8"))

# Parameters worth recording per estimator family (sklearn defaults omitted unless set).
KEY_PARAMS = {
    "RandomForestClassifier": ["n_estimators", "max_depth", "class_weight", "random_state"],
    "DecisionTreeClassifier": ["criterion", "max_depth", "min_samples_split",
                               "min_samples_leaf", "class_weight", "random_state"],
    "LogisticRegression": ["max_iter", "C", "class_weight", "solver", "random_state"],
    "XGBClassifier": ["n_estimators", "learning_rate", "max_depth", "subsample",
                      "colsample_bytree", "objective", "tree_method", "eval_metric",
                      "random_state"],
    "ExplainableBoostingClassifier": ["random_state"],
    "DummyClassifier": ["strategy"],
    "ProportionalOddsClassifier": ["max_iter", "l2", "class_weight"],
    "OrdinalMedianClassifier": [],
}


def describe(est) -> dict:
    name = type(est).__name__
    entry = {"estimator": name}
    if name == "FrankHallOrdinalClassifier":
        base = est.base_estimator
        entry["structure"] = "K-1 cumulative binary classifiers P(y>k); base estimator below"
        entry["base"] = describe(base) if base is not None else {"estimator": "LogisticRegression",
                                                                 "note": "default max_iter=1000"}
        return entry
    params = est.get_params() if hasattr(est, "get_params") else vars(est)
    for k in KEY_PARAMS.get(name, sorted(params)):
        if k in params:
            entry[k] = params[k]
    return entry


def build() -> dict:
    cfg = FR["config"]
    reg = Mo.build_registry(include_optional=True, seed=cfg["seed"])
    models = {}
    for key, spec in reg.items():
        est = spec.make_estimator()
        models[key] = {
            "kind": spec.kind,
            "class_weight_mode": spec.class_weight_mode,
            "scale_numeric": spec.scale_numeric,
            **describe(est),
        }
    return {
        "description": ("Fixed model configurations of the governed benchmark, extracted from "
                        "crashsev.models.build_registry with the frozen run seed. No "
                        "hyperparameter search was run in the corrected protocol; development "
                        "CV selected among these fixed configurations only."),
        "run_id": FR["run_id"],
        "config_sha256": FR["config_sha256"],
        "shared_protocol_constants": {
            "seed": cfg["seed"],
            "n_classes": cfg["n_classes"],
            "decision_rule": cfg["decision_rule"],
            "primary_metric": cfg["primary_metric"],
            "feature_tier": cfg["feature_tier"],
            "bootstrap_resamples": cfg["bootstrap_resamples"],
            "one_hot_min_frequency": cfg["min_frequency"],
            "categorical_max_cardinality": cfg["categorical_max_cardinality"],
            "calibration": {"enabled": cfg["calibrate"], "method": cfg["calibration_method"],
                            "folds": cfg["calibration_folds"]},
            "cv_strategy": cfg["cv_strategy"],
            "cv_seeds": cfg["cv_seeds"],
            "log_loss_probability_clip": LOG_LOSS_EPS,
        },
        "hyperparameter_search": "none (fixed configurations; see description)",
        "models": models,
    }


def main() -> int:
    art = build()
    payload = json.dumps(art, indent=2, default=str) + "\n"
    if "--check" in sys.argv[1:]:
        if not OUT.exists() or OUT.read_text("utf-8") != payload:
            print("[FAIL] hyperparameter ledger stale or missing; regenerate")
            return 1
        print("[PASS] hyperparameter_ledger.json is current")
        return 0
    OUT.write_text(payload, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
