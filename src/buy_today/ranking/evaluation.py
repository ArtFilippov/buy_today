"""Model-independent binary top-K metrics on held-out purchases."""

from typing import TypeIs

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from buy_today.ranking.data import RankingData, check_integer, check_ranking_data
from buy_today.ranking.domain import RankingEvaluation, Predictor


def _is_array(value: object) -> TypeIs[NDArray[np.generic]]:
    return isinstance(value, np.ndarray)


def _recommendations(model: Predictor[object], user: str, k: int, catalog: set[str]) -> list[str]:
    recommendations = model.predict(user, k)
    if not _is_array(recommendations) or recommendations.shape != (k,):
        raise ValueError("predict must return an array of shape (k,)")
    # ndarray object elements are deliberately validated at the model boundary.
    items: list[object] = list(recommendations)
    valid = [item for item in items if isinstance(item, str) and item in catalog]
    if len(valid) != k:
        raise ValueError("Recommendations must contain string product IDs from catalog")
    if len(set(valid)) != k:
        raise ValueError("Recommendations must contain unique products")
    return valid


def _user_metrics(
    user: str, relevant: set[str], recommendations: list[str], discounts: NDArray[np.float64]
) -> dict[str, str | int | float]:
    hits = np.asarray([item in relevant for item in recommendations], dtype=np.float64)
    ideal = np.cumsum(discounts)
    return {
        "user_id": user,
        "n_relevant": len(relevant),
        "hits": int(np.sum(hits)),
        "recall_at_k": float(np.sum(hits) / len(relevant)),
        "ndcg_at_k": float(hits @ discounts / ideal[min(len(discounts), len(relevant)) - 1]),
    }


def evaluate_ranker(
    model: Predictor[object], X: RankingData[pd.DataFrame], *, k: object = 10
) -> RankingEvaluation[pd.DataFrame]:
    """Evaluate an already fitted model using only predict(user_id, k).

    X contains held-out purchases and the candidate catalog. Relevance is the
    set of unique held-out products, including those unseen in training. Every
    represented user has equal weight. Empty inputs and invalid recommendations
    are errors, rather than silently changing the metric denominator.

    Args:
        model (Predictor[object]): Fitted predictor returning candidate identifiers.
        X (RankingData[pd.DataFrame]): Held-out interactions and the complete catalog.
        k (object, default=10): Positive integer cutoff, no larger than the catalog.

    Returns:
        RankingEvaluation[pd.DataFrame]: Macro averages and metrics for each represented user.
    """
    check_ranking_data(X)
    size = check_integer(k, "k", 1, len(X.catalog))
    catalog = set(X.catalog.product_id)
    discounts = 1 / np.log2(np.arange(2, size + 2, dtype=np.float64))
    per_user = pd.DataFrame(
        [
            _user_metrics(
                str(user_id),
                set(purchases.product_id),
                _recommendations(model, str(user_id), size, catalog),
                discounts,
            )
            for user_id, purchases in X.interactions.groupby("user_id", sort=True)
        ]
    )
    metrics = {name: float(per_user[name].mean()) for name in ("recall_at_k", "ndcg_at_k")}
    return RankingEvaluation(metrics, per_user)
