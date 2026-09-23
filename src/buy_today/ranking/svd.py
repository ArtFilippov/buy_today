"""Truncated SVD of train purchase counts over the complete candidate catalog."""

from typing import override

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_is_fitted

from buy_today.ranking.data import RankingData, check_integer, check_ranking_data
from buy_today.estimation import Ranker

_MIN_PRODUCTS = 2


def _purchase_counts(
    data: RankingData[pd.DataFrame], users: NDArray[np.object_], catalog: NDArray[np.object_]
) -> csr_matrix:
    user_ids: list[str] = data.interactions["user_id"].tolist()
    product_ids: list[str] = data.interactions["product_id"].tolist()
    user_indices = users.searchsorted(user_ids)
    product_indices = catalog.searchsorted(product_ids)
    return csr_matrix(
        (
            np.ones(len(data.interactions), dtype=np.float32),
            (
                user_indices,
                product_indices,
            ),
        ),
        shape=(len(users), len(catalog)),
    )


class SVDRanker(
    BaseEstimator, Ranker[RankingData[pd.DataFrame], NDArray[np.object_]]
):
    """Rank by the dot product of user (X @ V) and item (V) factors.

    fit accepts RankingData and rebuilds a float32 CSR count matrix on every
    call. IDs are sorted before factorization, making the seeded fit independent
    of input row order. No centering, binarization or bought-item filtering is
    applied. Exactly equal scores are ordered lexicographically by product_id.

    The catalog must have at least two items (TruncatedSVD's feature minimum),
    with 1 <= n_components <= min(number of users, catalog size).

    Args:
        n_components (object, default=32): Integer latent dimension, validated on fit.
        n_iter (object, default=7): Positive number of randomized SVD iterations.
        random_state (object, default=42): Integer seed, validated on fit.
    """

    def __init__(
        self, n_components: object = 32, n_iter: object = 7, random_state: object = 42
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.user_ids_: NDArray[np.object_]
        self.catalog_: NDArray[np.object_]
        self.user_factors_: NDArray[np.float32]
        self.item_factors_: NDArray[np.float32]

    @override
    def _fit(self, X: RankingData[pd.DataFrame], y: object = None) -> None:
        del y
        check_ranking_data(X)
        parameters = (
            check_integer(self.random_state, "random_state", 0, 2**32 - 1),
            check_integer(self.n_iter, "n_iter", 1, np.inf),
        )
        user_ids = np.asarray(sorted(set(X.interactions["user_id"])), dtype=object)
        catalog = np.asarray(sorted(X.catalog.product_id), dtype=object)
        if len(catalog) < _MIN_PRODUCTS:
            raise ValueError("SVD requires at least two products in catalog")
        decomposition = TruncatedSVD(
            n_components=check_integer(
                self.n_components, "n_components", 1, min(len(user_ids), len(catalog))
            ),
            random_state=parameters[0],
            n_iter=parameters[1],
        )
        counts = _purchase_counts(X, user_ids, catalog)
        # Identical users (including a single user) have zero total variance.
        # sklearn's unused explained_variance_ratio_ can then be undefined.
        with np.errstate(divide="ignore", invalid="ignore"):
            user_factors = decomposition.fit_transform(counts)
        item_factors = np.ascontiguousarray(decomposition.components_.T)
        # Null columns must have exactly zero scores, even at full/rank-deficient
        # decompositions where roundoff can leave tiny nonzero components.
        item_factors[np.equal(csr_matrix.getnnz(counts, axis=0), 0)] = 0

        self.user_ids_ = user_ids
        self.catalog_ = catalog
        self.user_factors_ = user_factors
        self.item_factors_ = item_factors

    @override
    def predict(self, user_id: object, k: object) -> NDArray[np.object_]:
        check_is_fitted(
            self,
            [
                "catalog_",
                "user_ids_",
                "user_factors_",
                "item_factors_",
            ],
        )
        size = check_integer(k, "k", 1, len(self.catalog_))
        if not isinstance(user_id, str) or user_id not in self.user_ids_:
            raise ValueError(f"Unknown user_id: {user_id!r}")
        scores = self.item_factors_ @ self.user_factors_[np.searchsorted(self.user_ids_, user_id)]
        # Stable sorting preserves catalog order on ties, including at the K
        # boundary, and gives the same prefix for every requested K.
        order = np.argsort(-scores, kind="stable")
        return self.catalog_[order[:size]].copy()
