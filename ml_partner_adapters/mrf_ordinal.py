"""
Adapter module providing Peyton's ``MultiLevelRandomForestClassifier`` under
a stable import path for the Django backend.

The implementation below is copied verbatim (aside from minor formatting
and import adjustments) from the Peyton ML repository, whose frozen snapshot
lives under
``peyton_original/Machine Learning Models/MultilevelRandomForest/MRF.py``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

class MultiLevelRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi-level Random Forest for ordered classes.

    For K ordered classes c0 < c1 < ... < c_{K-1}, we train K-1 binary
    RandomForests. For threshold j, the target is 1 if y > c_j else 0.

    At prediction time, for each sample we find the first threshold where
    the model predicts 0 and assign that class; if all thresholds are 1,
    we assign the highest class.
    Essentially ordinal random forest classification via multiple binary classifiers.
    """

    def __init__(
        self,
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        thresholds=None,  # None => use 0.5 for all levels

    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.thresholds = thresholds  # scalar or list


    def fit(self, X, y):
        y = np.asarray(y)

        #sort unique labels so we know the order 0 < 1 < 2 etc.
        classes = np.unique(y)
        classes = np.sort(classes)
        self.classes_ = classes

        if classes.size < 2:
            self.rf_list_ = []
            return self

        self.rf_list_ = []

        for j, c_j in enumerate(classes[:-1]):
            y_bin = (y > c_j).astype(int)

            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
                #different seed per level so trees differ slightly
                random_state=None if self.random_state is None else self.random_state + j,
            )
            rf.fit(X, y_bin)
            self.rf_list_.append(rf)

        return self

    def predict(self, X):
        if not hasattr(self, "rf_list_"):
            raise NotFittedError("MultiLevelRandomForestClassifier is not fitted yet.")

        classes = self.classes_
        n_classes = classes.size

        # trivial case: only one class
        if n_classes == 1 or len(self.rf_list_) == 0:
            return np.full(shape=(X.shape[0],), fill_value=classes[0])

        level_probs = []
        for rf in self.rf_list_:
            proba = rf.predict_proba(X)
            # positive class is index 1
            level_probs.append(proba[:, 1])

        level_probs = np.vstack(level_probs)  # shape: (K-1, n_samples)
        n_levels, n_samples = level_probs.shape
        y_pred = np.empty(n_samples, dtype=classes.dtype)

        # For each sample, walk thresholds in order
        for i in range(n_samples):
            assigned = classes[-1]  # default to highest class
            for j in range(n_levels):
                prob_pos = level_probs[j, i]
                thr = self.thresholds[j]
                # 0 = "not more severe than this threshold"
                if prob_pos < thr:
                    assigned = classes[j]
                    break
            y_pred[i] = assigned

        return y_pred

    def get_feature_importances(self):
        """
        Aggregate feature importances across all levels by taking their mean.
        Assumes all underlying RFs have the same number/order of features.
        """
        if not hasattr(self, "rf_list_") or len(self.rf_list_) == 0:
            raise NotFittedError("MultiLevelRandomForestClassifier is not fitted yet.")

        importances = [rf.feature_importances_ for rf in self.rf_list_]
        return np.mean(importances, axis=0)

__all__ = ["MultiLevelRandomForestClassifier"]
