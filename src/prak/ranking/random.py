"""A per-user deterministic random permutation of the full catalog."""

import hashlib

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from prak.ranking.data import _check_integer, check_ranking_data


class RandomRanker(BaseEstimator):
    """Register training users and candidates; previously bought items stay eligible.

    fit accepts RankingData, with repeated purchases preserved by the data
    interface. predict(user_id, k) requires a known user and 1 <= k <= catalog
    size. A stable SHA-256 seed per user makes prefixes independent of call
    order, input row order, process hash randomization and requested k.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y=None):
        _check_integer(self.random_state, "random_state", 0, 2**32 - 1)
        check_ranking_data(X)
        self.user_ids_ = np.asarray(sorted(X.interactions.user_id.unique()), dtype=object)
        self.catalog_ = np.asarray(sorted(X.catalog.product_id), dtype=object)
        self._user_set_ = frozenset(self.user_ids_)
        self.random_state_ = int(self.random_state)
        return self

    def predict(self, user_id, k):
        check_is_fitted(self, ["catalog_", "user_ids_", "random_state_", "_user_set_"])
        _check_integer(k, "k", 1, len(self.catalog_))
        if not isinstance(user_id, str) or user_id not in self._user_set_:
            raise ValueError(f"Unknown user_id: {user_id!r}")
        user_seed = int.from_bytes(hashlib.sha256(user_id.encode("utf-8")).digest(), "big")
        rng = np.random.default_rng(np.random.SeedSequence([self.random_state_, user_seed]))
        return rng.permutation(self.catalog_)[:k].copy()
