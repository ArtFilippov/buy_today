"""A per-user deterministic random permutation of the full catalog."""

import hashlib
from typing import override

import numpy as np
import pandas as pd
from numpy.random import Generator, PCG64, SeedSequence
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from buy_today.ranking.data import RankingData, check_integer, check_ranking_data
from buy_today.estimation import Ranker


class RandomRanker(
    BaseEstimator, Ranker[RankingData[pd.DataFrame], NDArray[np.object_]]
):
    """Register training users and candidates; previously bought items stay eligible.

    fit accepts RankingData, with repeated purchases preserved by the data
    interface. predict(user_id, k) requires a known user and 1 <= k <= catalog
    size. A stable SHA-256 seed per user makes prefixes independent of call
    order, input row order, process hash randomization and requested k.

    Args:
        random_state (object, default=42): Integer seed validated when fitting.
    """

    def __init__(self, random_state: object = 42) -> None:
        super().__init__()
        self.random_state = random_state
        self.user_ids_: NDArray[np.object_]
        self.catalog_: NDArray[np.object_]
        self._user_set_: frozenset[str]
        self.random_state_: int

    @override
    def _fit(self, X: RankingData[pd.DataFrame], y: object = None) -> None:
        del y  # sklearn's optional target is not used by unsupervised rankers.
        seed = check_integer(self.random_state, "random_state", 0, 2**32 - 1)
        check_ranking_data(X)
        self.user_ids_ = np.asarray(sorted(set(X.interactions["user_id"])), dtype=object)
        self.catalog_ = np.asarray(sorted(X.catalog.product_id), dtype=object)
        self._user_set_ = frozenset(self.user_ids_)
        self.random_state_ = seed

    @override
    def predict(self, user_id: object, k: object) -> NDArray[np.object_]:
        check_is_fitted(self, ["catalog_", "user_ids_", "random_state_", "_user_set_"])
        size = check_integer(k, "k", 1, len(self.catalog_))
        if not isinstance(user_id, str) or user_id not in self._user_set_:
            raise ValueError(f"Unknown user_id: {user_id!r}")
        user_seed = int.from_bytes(hashlib.sha256(user_id.encode("utf-8")).digest(), "big")
        rng = Generator(PCG64(SeedSequence([self.random_state_, user_seed])))
        return Generator.permutation(rng, self.catalog_)[:size].copy()
