"""Absolute differences in purchase time, in seconds."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from buy_today.schema import SORT_KEY


def _seconds(X) -> np.ndarray:
    column = SORT_KEY[0]
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique or column not in X:
        raise ValueError(f"X must be a DataFrame containing {column}")
    times = X[column]
    if not pd.api.types.is_datetime64_dtype(times.dtype) or times.isna().any():
        raise ValueError(f"{column} must contain nonmissing timezone-naive datetimes")
    # Convert explicitly: pandas datetime arrays can have ns, us, ms or s units.
    # Floating point subtraction also avoids int64 timedelta overflow.
    return times.to_numpy(dtype="datetime64[ns]").astype(np.float64) / 1e9


class TimestampDistance(BaseEstimator):
    def fit(self, X, y=None):
        """Deliberate no-op: pairwise needs no fitted state or training data."""
        return self

    def pairwise(self, X, Y=None):
        x = _seconds(X)
        y = x if Y is None else _seconds(Y)
        return np.abs(x[:, None] - y[None, :])
