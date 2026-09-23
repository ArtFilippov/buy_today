"""Category and relative-price preferences for synthetic purchase histories."""

from typing import override

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.base import BaseEstimator

from buy_today.estimation import Distance

REAL_DTYPE_KINDS = "iuf"


def _features(frame: object) -> tuple[NDArray[np.str_], NDArray[np.float64]]:
    columns = {"product_category_name", "price"}
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Expected a DataFrame with unique columns")
    if not columns.issubset(frame.columns):
        raise ValueError("Expected product_category_name and price")
    categories = frame["product_category_name"]
    if categories.isna().any() or not pd.api.types.is_string_dtype(categories):
        raise ValueError("Categories must be nonmissing strings")
    if categories.str.strip().eq("").any():
        raise ValueError("Categories must be nonempty strings")
    prices = frame["price"]
    if not pd.api.types.is_numeric_dtype(prices) or prices.dtype.kind not in REAL_DTYPE_KINDS:
        raise ValueError("Prices must be real numbers")
    values = prices.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be finite and positive")
    return categories.to_numpy(dtype=np.str_), np.log(values)


class CategoryPriceDistance(BaseEstimator, Distance[object, NDArray[np.float64]]):
    """Category mismatch plus absolute log-price gap, in dimensionless units.

    Sampling with temperature 0.1 models a narrow, single-category price
    preference. The distance uses neither product IDs nor user/anchor labels.
    Equal category and price can give zero distance for distinct products.
    """

    def __init__(self) -> None:
        super().__init__()

    @override
    def _fit(self, X: object, y: object = None) -> None:
        """No learned state; validation occurs at the pairwise boundary.

        Args:
            X (object): Unused training data.
            y (object, default=None): Unused estimator target.
        """
        del X, y

    @staticmethod
    @override
    def pairwise(X: object, Y: object = None) -> NDArray[np.float64]:
        categories, prices = _features(X)
        other_categories, other_prices = (categories, prices) if Y is None else _features(Y)
        mismatch = categories[:, None] != other_categories[None, :]
        return mismatch.astype(np.float64) + np.abs(prices[:, None] - other_prices[None, :])
