"""Model-independent binary top-K metrics on held-out purchases."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prak.ranking.data import RankingData, _check_integer, check_ranking_data


@dataclass(frozen=True)
class RankingEvaluation:
    metrics: dict[str, float]
    per_user: pd.DataFrame


def evaluate_ranker(model, X: RankingData, *, k=10) -> RankingEvaluation:
    """Evaluate an already fitted model using only predict(user_id, k).

    X contains held-out purchases and the candidate catalog. Relevance is the
    set of unique held-out products, including those unseen in training. Every
    represented user has equal weight. Empty inputs and invalid recommendations
    are errors, rather than silently changing the metric denominator.
    """
    check_ranking_data(X)
    _check_integer(k, "k", 1, len(X.catalog))
    catalog = set(X.catalog.product_id)
    discounts = 1 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    ideal = np.cumsum(discounts)
    rows = []
    for user_id, purchases in X.interactions.groupby("user_id", sort=True):
        relevant = set(purchases.product_id)
        recommendations = model.predict(user_id, k)
        if not isinstance(recommendations, np.ndarray) or recommendations.shape != (k,):
            raise ValueError("predict must return an array of shape (k,)")
        if not all(isinstance(item, str) and item in catalog for item in recommendations):
            raise ValueError("Recommendations must contain string product IDs from catalog")
        if len(set(recommendations)) != k:
            raise ValueError("Recommendations must contain unique products")
        hits = np.asarray([item in relevant for item in recommendations], dtype=np.float64)
        rows.append({
            "user_id": user_id, "n_relevant": len(relevant), "hits": int(hits.sum()),
            "recall_at_k": float(hits.sum() / len(relevant)),
            "ndcg_at_k": float(hits @ discounts / ideal[min(k, len(relevant)) - 1]),
        })
    per_user = pd.DataFrame(rows)
    metrics = {name: float(per_user[name].mean()) for name in ("recall_at_k", "ndcg_at_k")}
    return RankingEvaluation(metrics, per_user)
