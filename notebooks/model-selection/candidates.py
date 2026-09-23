"""Experimental distances and train-only baselines for reproducible selection."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize


class ProductDistance(BaseEstimator):
    """Weighted category mismatch, log-price gap and optional elapsed days."""

    def __init__(self, category_weight=1.0, price_weight=1.0, time_scale_days=None):
        self.category_weight = category_weight
        self.price_weight = price_weight
        self.time_scale_days = time_scale_days

    def fit(self, X, y=None):
        return self

    def pairwise(self, X, Y=None):
        Y = X if Y is None else Y
        category = (
            X.product_category_name.to_numpy()[:, None]
            != Y.product_category_name.to_numpy()[None, :]
        )
        price = np.abs(
            np.log(X.price.to_numpy())[:, None] - np.log(Y.price.to_numpy())[None, :]
        )
        result = self.category_weight * category + self.price_weight * price
        if self.time_scale_days is not None:
            x = X.order_purchase_timestamp.to_numpy(dtype="datetime64[s]").astype(float)
            y = Y.order_purchase_timestamp.to_numpy(dtype="datetime64[s]").astype(float)
            result += np.abs(x[:, None] - y[None, :]) / (86400 * self.time_scale_days)
        return result


class FrequencyRanker(BaseEstimator):
    """Global popularity or personal frequency with popularity tie-breaking."""

    def __init__(self, personal=False):
        self.personal = personal

    def fit(self, X, y=None):
        self.catalog_ = np.asarray(sorted(X.catalog.product_id), dtype=object)
        self.users_ = np.asarray(sorted(X.interactions.user_id.unique()), dtype=object)
        self.counts_ = csr_matrix((
            np.ones(len(X.interactions), dtype=np.float32),
            (self.users_.searchsorted(X.interactions.user_id),
             self.catalog_.searchsorted(X.interactions.product_id)),
        ), shape=(len(self.users_), len(self.catalog_)))
        self.popularity_ = np.asarray(self.counts_.sum(axis=0)).ravel()
        return self

    def predict(self, user_id, k):
        scores = self.popularity_ / (self.popularity_.max() + 1)
        if self.personal:
            scores = scores + self.counts_[self.users_.searchsorted(user_id)].toarray().ravel()
        return self.catalog_[np.argsort(-scores, kind="stable")[:k]].copy()


class UserKNNRanker(FrequencyRanker):
    """Cosine neighbors over binary interactions; weighted purchase counts."""

    def __init__(self, n_neighbors=30):
        super().__init__()
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        super().fit(X, y)
        binary = self.counts_.copy()
        binary.data[:] = 1
        unit = normalize(binary)
        similarity = (unit @ unit.T).toarray()
        np.fill_diagonal(similarity, 0)
        neighbors = np.argsort(-similarity, axis=1, kind="stable")[:, :self.n_neighbors]
        weights = csr_matrix((
            np.take_along_axis(similarity, neighbors, axis=1).ravel(),
            (np.repeat(np.arange(len(self.users_)), neighbors.shape[1]), neighbors.ravel()),
        ), shape=similarity.shape)
        self.scores_ = (weights @ self.counts_).tocsr()
        return self

    def predict(self, user_id, k):
        scores = self.scores_[self.users_.searchsorted(user_id)].toarray().ravel()
        return self.catalog_[np.argsort(-scores, kind="stable")[:k]].copy()


def dataset_quality(batch, events, anchors):
    """Quality uses train only; held-out events are reserved for evaluation."""
    train = events.loc[events.split == "train"]
    unique = train.groupby("user_id").product_id.nunique()
    counts = train.groupby(["user_id", "product_id"]).size()
    maximum = counts.groupby(level=0).max() / train.groupby("user_id").size()
    category_by_key = batch.set_index(["order_id", "order_item_id"]).product_category_name
    keys = pd.MultiIndex.from_frame(train[["order_id", "order_item_id"]])
    categories = pd.Series(category_by_key.reindex(keys).to_numpy(), index=train.index)
    source_share = batch.product_category_name.value_counts(normalize=True)
    generated_share = categories.value_counts(normalize=True).reindex(source_share.index, fill_value=0)
    anchor_keys = pd.MultiIndex.from_frame(anchors[["order_id", "order_item_id"]])
    anchor_categories = pd.Series(
        category_by_key.reindex(anchor_keys).to_numpy(), index=anchors.user_id,
    )
    return {
        "n_source_rows": len(batch), "n_catalog": batch.product_id.nunique(),
        "n_users": len(anchors), "unique_train_median": float(unique.median()),
        "unique_train_p10": float(unique.quantile(.1)),
        "mean_top_product_share": float(maximum.mean()),
        "train_catalog_coverage": train.product_id.nunique() / batch.product_id.nunique(),
        "category_tvd": float((source_share - generated_share).abs().sum() / 2),
        "same_category_share": float((categories.to_numpy() == train.user_id.map(anchor_categories).to_numpy()).mean()),
    }
