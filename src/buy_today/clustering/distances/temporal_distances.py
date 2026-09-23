"""Absolute differences in purchase time, in seconds."""

from typing import override

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.base import BaseEstimator

from buy_today.schema import SORT_KEY
from buy_today.estimation import Distance


def _seconds(frame: object) -> NDArray[np.float64]:
    column = SORT_KEY[0]
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or column not in frame:
        raise ValueError(f"X must be a DataFrame containing {column}")
    times = frame[column]
    if not pd.api.types.is_datetime64_dtype(times.dtype) or times.isna().any():
        raise ValueError(f"{column} must contain nonmissing timezone-naive datetimes")
    # Convert explicitly: pandas datetime arrays can have ns, us, ms or s units.
    # Floating point subtraction also avoids int64 timedelta overflow.
    return times.to_numpy(dtype="datetime64[ns]").astype(np.float64) / 1e9


class TimestampDistance(BaseEstimator, Distance[object, NDArray[np.float64]]):
    def __init__(self) -> None:
        super().__init__()

    @override
    def _fit(self, X: object, y: object = None) -> None:
        """Deliberate no-op: pairwise needs no fitted state or training data.

        Args:
            X (object): Unused training input, accepted for estimator compatibility.
            y (object, default=None): Unused target, accepted for estimator compatibility.

        """
        del X, y

    @staticmethod
    @override
    def pairwise(X: object, Y: object = None) -> NDArray[np.float64]:
        x = _seconds(X)
        y = x if Y is None else _seconds(Y)
        return np.abs(x[:, None] - y[None, :])
