"""Generate the machine-readable metric-convention sidecar (R-007 / R-008 / E-019; extended
by the r3 submission-readiness pass with F1-CONV-001, BOOT-CONV-001, and FH-CONV-001).

Records — extracted from the implementation and the frozen configuration, never from
memory — the exact conventions behind the published probability metrics:

  * ranked probability score: UNNORMALIZED cumulative form (no division by K-1);
  * log loss: the exact probability-clipping epsilon and rule;
  * ECE: top-label definition, bin count, and bin-edge convention;
  * multiclass Brier score definition;
  * F1 zero-division: the count-based REPORTING convention used by every manuscript
    table versus the NaN STORAGE convention preserved inside the frozen bundles;
  * paired bootstrap: interval type, level, seed, resampling unit, refit status;
  * Frank-Hall probability construction: monotonicity repair, flooring, renormalization.

The governed artifacts (final_results.json, run bundles) are frozen byte-for-byte, so
these conventions live in a separate committed sidecar, ``experiment/metric_conventions.json``,
referenced by the manuscript and enforced by tests/test_metric_conventions.py (which
recomputes stored values under the declared convention and fails on any silent switch).

Emits deterministic JSON (no timestamps). Usage:
    python tools/gen_metric_conventions.py [--check]
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

REM = Path(__file__).resolve().parents[1]
OUT = REM / "experiment" / "metric_conventions.json"
sys.path.insert(0, str(REM))

from crashsev import calibration as C  # noqa: E402
from crashsev import metrics as M  # noqa: E402
from crashsev import models as Mo  # noqa: E402
from crashsev import uncertainty as U  # noqa: E402


def build() -> dict:
    fr = json.loads((REM / "experiment" / "final_results.json").read_text("utf-8"))
    k = int(fr["config"]["n_classes"])

    eps = inspect.signature(M.multiclass_log_loss).parameters["eps"].default
    n_bins = inspect.signature(C.expected_calibration_error).parameters["n_bins"].default

    rps_src = inspect.getsource(M.ranked_probability_score)
    # the implementation must not divide the threshold sum by (K-1); guard the extraction
    assert "/ (n_classes - 1)" not in rps_src and "/(n_classes-1)" not in rps_src, \
        "ranked_probability_score now normalizes; regenerate conventions AND the manuscript"

    # F1-CONV-001: metrics must expose both conventions with the frozen storage default
    f1_default = inspect.signature(M.per_class_prf_from_cm).parameters["zero_division"].default
    assert f1_default == "nan", \
        "pipeline storage default changed; frozen bundles would no longer self-reproduce"

    # BOOT-CONV-001: extract the interval mechanics from the implementation
    boot_src = inspect.getsource(U.paired_difference_ci)
    assert "nanpercentile" in boot_src, "paired interval no longer percentile-based; update everything"
    boot_alpha = inspect.signature(U.paired_difference_ci).parameters["alpha"].default

    # FH-CONV-001: guard the monotonicity repair, floor, and renormalization
    fh_src = inspect.getsource(Mo.FrankHallOrdinalClassifier)
    assert "np.minimum.accumulate" in fh_src, "Frank-Hall monotonicity repair changed"
    assert "1e-9" in fh_src, "Frank-Hall probability floor changed"

    return {
        "applies_to": "experiment/final_results.json results.*.{rps,log_loss,brier,ece} "
                      "and every probability metric derived from the frozen predictions",
        "n_classes": k,
        "rps_definition": "mean unnormalized sum of squared cumulative errors over K-1 thresholds",
        "rps_normalization": "none",
        "alternative_normalized_divisor": k - 1,
        "rps_source": "crashsev/metrics.py::ranked_probability_score — "
                      "mean_i sum_{k=0}^{K-2} (F_ik - O_ik)^2 with no division by K-1; "
                      "for K=3 these values are twice the K-1-normalized alternative",
        "log_loss_probability_clipping": f"probabilities clipped to [{eps:g}, 1-{eps:g}] "
                                         "and then row-renormalized before taking the log",
        "log_loss_epsilon": eps,
        "log_loss_source": "crashsev/metrics.py::multiclass_log_loss (eps keyword default)",
        "hard_one_hot_unclipped_behavior": "infinite when an observed class has probability zero",
        "ece_definition": "top-label ECE: bin the maximum predicted probability (confidence); "
                          "count-weighted mean of |mean confidence - accuracy| over non-empty bins",
        "ece_n_bins": n_bins,
        "ece_bin_edges": "numpy.linspace(0, 1, n_bins+1) equal-width edges; first bin closed "
                         "[0, 1/n_bins], later bins half-open (lo, hi]; empty bins skipped",
        "ece_source": "crashsev/calibration.py::expected_calibration_error (n_bins keyword default)",
        "brier_definition": "multiclass Brier score: mean over rows of the sum over classes of "
                            "squared (predicted probability - one-hot outcome) error",
        "brier_source": "crashsev/metrics.py::multiclass_brier",
        "f1_reporting_convention": {
            "id": "F1-CONV-001",
            "reporting": "count-based zero_division='zero': precision undefined (em dash) when "
                         "TP+FP=0; recall a measured 0 when TP=0 with positive support; "
                         "F1 = 2TP/(2TP+FP+FN), a measured 0 for a never-predicted class with "
                         "positive support; macro-F1 averages class F1 values INCLUDING zeros",
            "storage": "frozen run bundles store zero_division='nan' (METRIC-001): F1 and "
                       "macro-F1 are flagged NaN whenever a class is never predicted; preserved "
                       "byte-for-byte so a pipeline rerun reproduces the frozen artifacts",
            "pipeline_default": "nan",
            "source": "crashsev/metrics.py::per_class_prf_from_cm / macro_f1_from_cm "
                      "(zero_division keyword)",
            "conversion_rule": "reporting values derive deterministically from the stored "
                               "confusion matrices / per-class counts; no frozen artifact is "
                               "modified",
        },
        "paired_bootstrap": {
            "id": "BOOT-CONV-001",
            "interval_type": "percentile (numpy.nanpercentile at 100*alpha/2 and 100*(1-alpha/2); "
                             "default linear quantile interpolation)",
            "confidence_level": 1.0 - boot_alpha,
            "alpha": boot_alpha,
            "n_resamples": int(fr["config"]["bootstrap_resamples"]),
            "seed": int(fr["config"]["seed"]),
            "rng": "numpy.random.default_rng(seed) (PCG64)",
            "resampling_unit": "crash (one row per crash; crash-level and row-level coincide)",
            "pairing": "the SAME resampled index set is applied to both models' stored predictions",
            "models_refit_within_resamples": False,
            "conditions_on": "fitted models, frozen protocol, observed cohort, working target "
                             "definition (case-sampling variation only)",
            "degenerate_resamples": "impossible: every replicate concatenates exactly n "
                                    "single-row groups",
            "source": "crashsev/uncertainty.py::paired_difference_ci / bootstrap_metric_ci; "
                      "seed and resample count from the frozen run configuration",
        },
        "frank_hall_probability_construction": {
            "id": "FH-CONV-001",
            "decomposition": "K-1 binary classifiers estimate exceedance probabilities "
                             "P(Y>k|x), k=0..K-2",
            "monotonicity": "NOT guaranteed by the separate binary fits; ENFORCED by a running "
                            "cumulative minimum over k (np.minimum.accumulate) plus clipping "
                            "to [0,1]",
            "class_recovery": "differencing: p_0 = 1 - g_0; p_k = g_{k-1} - g_k; "
                              "p_{K-1} = g_{K-2} (nonnegative after the repair)",
            "negative_probabilities": "impossible after the repair; a 1e-9 floor is applied "
                                      "before renormalization",
            "row_renormalization": True,
            "degenerate_training_fold": "a threshold whose training rows are single-sided is "
                                        "replaced by the constant empirical exceedance rate",
            "applies_to": "frank_hall_logistic AND ordinal_random_forest (weighted and "
                          "unweighted) — one shared implementation",
            "source": "crashsev/models.py::FrankHallOrdinalClassifier (_cum_gt/predict_proba)",
        },
    }


def main() -> int:
    payload = json.dumps(build(), indent=2) + "\n"
    if "--check" in sys.argv[1:]:
        if not OUT.exists() or OUT.read_text("utf-8") != payload:
            print("[FAIL] metric_conventions.json stale or missing; regenerate")
            return 1
        print("[PASS] metric_conventions.json is current")
        return 0
    OUT.write_text(payload, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
