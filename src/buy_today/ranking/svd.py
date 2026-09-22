"""Truncated SVD of train purchase counts over the complete candidate catalog."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_is_fitted

from buy_today.ranking.data import _check_integer, check_ranking_data


class SVDRanker(BaseEstimator):
    """Rank by the dot product of user (X @ V) and item (V) factors.

    fit accepts RankingData and rebuilds a float32 CSR count matrix on every
    call. IDs are sorted before factorization, making the seeded fit independent
    of input row order. No centering, binarization or bought-item filtering is
    applied. Exactly equal scores are ordered lexicographically by product_id.

    The catalog must have at least two items (TruncatedSVD's feature minimum),
    with 1 <= n_components <= min(number of users, catalog size).
    """

    def __init__(self, n_components=32, n_iter=7, random_state=42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        check_ranking_data(X)
        _check_integer(self.random_state, "random_state", 0, 2**32 - 1)
        _check_integer(self.n_iter, "n_iter", 1, np.inf)
        user_ids = np.asarray(sorted(X.interactions.user_id.unique()), dtype=object)
        catalog = np.asarray(sorted(X.catalog.product_id), dtype=object)
        if len(catalog) < 2:
            raise ValueError("SVD requires at least two products in catalog")
        _check_integer(self.n_components, "n_components", 1, min(len(user_ids), len(catalog)))
        counts = csr_matrix(
            (
                np.ones(len(X.interactions), dtype=np.float32),
                (
                    np.searchsorted(user_ids, X.interactions.user_id.to_numpy()),
                    np.searchsorted(catalog, X.interactions.product_id.to_numpy()),
                ),
            ),
            shape=(len(user_ids), len(catalog)),
        )
        decomposition = TruncatedSVD(
            n_components=int(self.n_components), n_iter=int(self.n_iter),
            random_state=int(self.random_state),
        )
        # Identical users (including a single user) have zero total variance.
        # sklearn's unused explained_variance_ratio_ can then be undefined.
        with np.errstate(divide="ignore", invalid="ignore"):
            user_factors = decomposition.fit_transform(counts)
        item_factors = np.ascontiguousarray(decomposition.components_.T)
        # Null columns must have exactly zero scores, even at full/rank-deficient
        # decompositions where roundoff can leave tiny nonzero components.
        item_factors[counts.getnnz(axis=0) == 0] = 0

        self.user_ids_ = user_ids
        self.catalog_ = catalog
        self.user_factors_ = user_factors
        self.item_factors_ = item_factors
        self._user_indices_ = {user: index for index, user in enumerate(user_ids)}
        return self

    def predict(self, user_id, k):
        check_is_fitted(self, [
            "catalog_", "user_ids_", "user_factors_", "item_factors_", "_user_indices_",
        ])
        _check_integer(k, "k", 1, len(self.catalog_))
        if not isinstance(user_id, str) or user_id not in self._user_indices_:
            raise ValueError(f"Unknown user_id: {user_id!r}")
        scores = self.item_factors_ @ self.user_factors_[self._user_indices_[user_id]]
        # Stable sorting preserves catalog order on ties, including at the K
        # boundary, and gives the same prefix for every requested K.
        order = np.argsort(-scores, kind="stable")
        return self.catalog_[order[:k]].copy()
