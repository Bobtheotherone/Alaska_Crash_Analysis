"""
Lightweight GPU backend wrapper for the multilevel random forest model.

This module lazily imports RAPIDS cuML so that CPU-only environments can
continue to import and run the project without the GPU dependencies
installed.  The public surface mirrors the sklearn RandomForestClassifier
methods used by MultiLevelRandomForestClassifier: ``fit`` and
``predict_proba``.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:  # pragma: no cover - only used if sklearn is available
    from sklearn.exceptions import NotFittedError
except Exception:  # pragma: no cover - fallback to avoid hard import dependency
    class NotFittedError(RuntimeError):
        """Fallback NotFittedError when sklearn is unavailable."""


class GpuRandomForestClassifier:
    """
    Thin wrapper around a GPU-based RandomForest implementation.

    The implementation is intentionally minimal to keep behaviour aligned
    with the CPU sklearn RandomForestClassifier that the ordinal wrapper
    uses today.  Parameters unsupported by the GPU backend (e.g.
    class_weight, n_jobs) are ignored rather than causing import failures.
    """

    # Conservative subset of parameters that map cleanly onto cuML.
    _SUPPORTED_PARAMS = {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "random_state",
        "bootstrap",
        "n_streams",
    }

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = dict(params)
        self._model = None

    @staticmethod
    def _import_backend():
        """
        Import the GPU backend lazily to avoid hard dependencies for
        CPU-only users.
        """
        try:
            from cuml.ensemble import RandomForestClassifier as CumlRandomForestClassifier
        except Exception as exc:  # pragma: no cover - exercised in GPU environments
            raise ImportError(
                "cuML RandomForestClassifier is unavailable. Install RAPIDS/cuML "
                "to enable GPU training for the multilevel random forest."
            ) from exc

        return CumlRandomForestClassifier

    def _prepare_params(self) -> Dict[str, Any]:
        """
        Filter provided params down to those the GPU backend understands.

        Unsupported parameters are silently dropped so that callers can
        continue passing the same kwargs used for the CPU sklearn backend.
        """
        filtered = {k: v for k, v in self.params.items() if v is not None}
        return {k: filtered[k] for k in self._SUPPORTED_PARAMS if k in filtered}

    @staticmethod
    def _to_backend_array(arr: Any) -> Any:
        """Convert pandas / list inputs into numpy arrays for the backend."""
        if hasattr(arr, "values"):
            return np.asarray(arr.values)
        return np.asarray(arr)

    @staticmethod
    def _to_numpy(proba: Any) -> np.ndarray:
        """Best-effort conversion of backend outputs to numpy."""
        try:
            import cupy as cp  # type: ignore
        except Exception:  # pragma: no cover - CPU / non-cupy environments
            cp = None

        if hasattr(proba, "to_numpy"):
            return proba.to_numpy()
        if cp is not None and isinstance(proba, cp.ndarray):  # pragma: no cover - GPU only
            return cp.asnumpy(proba)
        if hasattr(proba, "get"):  # cupy / GPU arrays sometimes expose .get()
            try:
                return proba.get()
            except Exception:
                pass
        return np.asarray(proba)

    def fit(self, X: Any, y: Any) -> "GpuRandomForestClassifier":
        CumlRF = self._import_backend()
        params = self._prepare_params()

        X_arr = self._to_backend_array(X)
        y_arr = self._to_backend_array(y)

        self._model = CumlRF(**params)
        self._model.fit(X_arr, y_arr)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise NotFittedError("GpuRandomForestClassifier is not fitted yet.")

        X_arr = self._to_backend_array(X)
        proba = self._model.predict_proba(X_arr)
        return self._to_numpy(proba)


__all__ = ["GpuRandomForestClassifier"]
