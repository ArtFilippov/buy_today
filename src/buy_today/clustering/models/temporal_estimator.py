"""Numerical adapter assigning equal-sized consecutive temporal groups."""

from numbers import Integral
from typing import Any, override

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin

from buy_today.estimation import Clusterable
from buy_today.schema import ROW_KEY, SORT_KEY


def _validate_frame(frame: object) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("X must be a DataFrame with unique column names")
    if not set(SORT_KEY).issubset(frame.columns):
        raise ValueError(f"X must contain {SORT_KEY}")
    times = frame[SORT_KEY[0]]
    if not pd.api.types.is_datetime64_any_dtype(times.dtype):
        raise ValueError(f"{SORT_KEY[0]} must contain datetime values")
    if frame[list(SORT_KEY)].isna().any().any():
        raise ValueError("Temporal sorting columns must not contain missing values")
    if frame.duplicated(list(ROW_KEY)).any():
        raise ValueError(f"Duplicate row keys: {ROW_KEY}")
    return frame


class TemporalClustering(ClusterMixin, BaseEstimator, Clusterable[object, NDArray[np.int64]]):
    """Split time/key-sorted Olist positions, returning labels in input order.

    Args:
        n_clusters (int, default=20): Number of consecutive temporal groups.
    """

    def __init__(self, n_clusters: int = 20) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        # Annotations do not create fitted attributes before the first fit.
        self.labels_: NDArray[np.int64]
        self.n_features_in_: int
        self.feature_names_in_: NDArray[np.object_]

    @override
    def _fit(self, X: object, y: object = None) -> None:
        del y  # Unsupervised estimator: targets never affect temporal assignments.
        if (
            isinstance(self.n_clusters, (bool, np.bool_))
            or not isinstance(self.n_clusters, Integral)
            or self.n_clusters <= 0
        ):
            raise ValueError("n_clusters must be a positive integer")
        frame = _validate_frame(X)
        if len(frame) < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")

        # A positional index is essential: caller indexes may be nonunique.
        ordered = pd.DataFrame.reset_index(frame, drop=True).sort_values(list(SORT_KEY))
        order = pd.Index.to_numpy(ordered.index)
        labels = np.empty(len(frame), dtype=np.int64)
        for cluster_id, positions in enumerate(np.array_split(order, self.n_clusters)):
            labels[positions] = cluster_id
        self.labels_ = labels
        self.n_features_in_ = frame.shape[1]
        columns: list[object] = list(frame.columns)
        if all(isinstance(column, str) for column in columns):
            self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

    @override
    def fit_predict(self, X: object, y: object = None, **kwargs: Any) -> NDArray[np.int64]:
        _ = self.fit(X, y, **kwargs)
        return self.labels_
