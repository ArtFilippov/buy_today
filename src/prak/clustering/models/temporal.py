"""Equal-sized, consecutive temporal groups and their full-refit strategy."""

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin

from prak.clustering.distances.temporal_distances import TimestampDistance
from prak.clustering.models import ModelSnapshot
from prak.schema import ROW_KEY, SORT_KEY


class TemporalClustering(ClusterMixin, BaseEstimator):
    """Split time/key-sorted Olist positions, returning labels in input order."""

    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        if (
            isinstance(self.n_clusters, (bool, np.bool_))
            or not isinstance(self.n_clusters, Integral)
            or self.n_clusters <= 0
        ):
            raise ValueError("n_clusters must be a positive integer")
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("X must be a DataFrame with unique column names")
        if not set(SORT_KEY).issubset(X.columns):
            raise ValueError(f"X must contain {SORT_KEY}")
        times = X[SORT_KEY[0]]
        if not pd.api.types.is_datetime64_any_dtype(times.dtype):
            raise ValueError(f"{SORT_KEY[0]} must contain datetime values")
        if X[list(SORT_KEY)].isna().any().any():
            raise ValueError("Temporal sorting columns must not contain missing values")
        if X.duplicated(list(ROW_KEY)).any():
            raise ValueError(f"Duplicate row keys: {ROW_KEY}")
        if len(X) < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")

        # A positional index is essential: caller indexes may be nonunique.
        order = X.reset_index(drop=True).sort_values(list(SORT_KEY)).index.to_numpy()
        labels = np.empty(len(X), dtype=np.int64)
        for cluster_id, positions in enumerate(np.array_split(order, self.n_clusters)):
            labels[positions] = cluster_id
        self.labels_ = labels
        self.n_features_in_ = X.shape[1]
        if all(isinstance(column, str) for column in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        return self


def train_temporal(X: pd.DataFrame, *, n_clusters=20, distance=None) -> ModelSnapshot:
    """Refit on the supplied accumulated dataset and replace all assignments.

    This is the temporal baseline's policy, not a requirement of other models.
    A learned distance can also be prepared here, before independent evaluation.
    """
    model = TemporalClustering(n_clusters=n_clusters).fit(X)
    distance = TimestampDistance() if distance is None else distance
    distance.fit(X)
    labels = X[list(ROW_KEY)].copy()
    labels["cluster_id"] = model.labels_
    return ModelSnapshot(model=model, distance=distance, labels=labels)
