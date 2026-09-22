"""Binary held-out ranking metrics through the predict-only interface."""

from math import log2

import numpy as np
import pandas as pd
import pytest

from buy_today.ranking.data import RankingData
from buy_today.ranking.evaluation import RankingEvaluation, evaluate_ranker
from buy_today.ranking.random import RandomRanker


class PredictOnlyRanker:
    """A model with no fit method, training data, catalog or sklearn attributes."""

    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def predict(self, user_id, k, /):
        self.calls.append((user_id, k))
        return self.predictions[user_id]


@pytest.fixture
def held_out():
    return RankingData(
        pd.DataFrame({"user_id": ["u"] * 3, "product_id": ["a", "b", "a"]}),
        pd.DataFrame({"product_id": ["a", "b", "c", "d", "e"]}),
    )


def test_binary_metrics_and_macro_average_use_only_predict():
    data = RankingData(
        pd.DataFrame({
            "user_id": ["many"] * 6 + ["few"] * 2,
            "product_id": ["a", "a", "a", "b", "c", "d", "cold", "cold"],
        }),
        pd.DataFrame({"product_id": ["a", "b", "c", "d", "cold", "miss"]}),
    )
    interactions = data.interactions.copy(deep=True)
    catalog = data.catalog.copy(deep=True)
    model = PredictOnlyRanker({
        "many": np.array(["a", "miss", "c"]),
        "few": np.array(["miss", "cold", "a"]),
    })

    result = evaluate_ranker(model, data, k=np.int64(3))

    # many: four unique relevant items, hits at ranks 1 and 3; IDCG stops at K=3.
    # few: one unique relevant item, hit at rank 2; IDCG has only its rank-1 term.
    rank_two_discount = 1 / log2(3)
    many_ndcg = (1 + 1 / 2) / (1 + rank_two_discount + 1 / 2)
    few_ndcg = rank_two_discount
    expected = pd.DataFrame([
        {"user_id": "few", "n_relevant": 1, "hits": 1, "recall_at_k": 1.0, "ndcg_at_k": few_ndcg},
        {"user_id": "many", "n_relevant": 4, "hits": 2, "recall_at_k": 0.5, "ndcg_at_k": many_ndcg},
    ]).set_index("user_id")
    assert isinstance(result, RankingEvaluation)
    pd.testing.assert_frame_equal(
        result.per_user.set_index("user_id").sort_index(), expected,
        check_exact=False, rtol=1e-12, atol=1e-12,
    )
    assert result.metrics == pytest.approx({
        "recall_at_k": 0.75,
        "ndcg_at_k": (many_ndcg + few_ndcg) / 2,
    })
    assert sorted(model.calls) == [("few", 3), ("many", 3)]
    pd.testing.assert_frame_equal(data.interactions, interactions)
    pd.testing.assert_frame_equal(data.catalog, catalog)
    np.testing.assert_array_equal(model.predictions["many"], ["a", "miss", "c"])
    np.testing.assert_array_equal(model.predictions["few"], ["miss", "cold", "a"])


@pytest.mark.parametrize(("recommendations", "score", "hits"), [
    (["c", "d", "e"], 0.0, 0),
    (["b", "a", "c"], 1.0, 2),
], ids=["no-hits", "perfect-hits"])
def test_no_hit_and_perfect_hit_metrics(held_out, recommendations, score, hits):
    model = PredictOnlyRanker({"u": np.array(recommendations)})

    result = evaluate_ranker(model, held_out, k=3)

    assert result.metrics == pytest.approx({"recall_at_k": score, "ndcg_at_k": score})
    assert result.per_user.n_relevant.tolist() == [2]
    assert result.per_user.hits.tolist() == [hits]


def test_ground_truth_keeps_items_absent_from_train_and_previously_bought_items():
    catalog = pd.DataFrame({"product_id": ["seen", "cold"]})
    train = RankingData(pd.DataFrame({"user_id": ["u"], "product_id": ["seen"]}), catalog)
    model = RandomRanker().fit(train)
    held_out = RankingData(
        pd.DataFrame({"user_id": ["u"] * 3, "product_id": ["cold", "seen", "cold"]}),
        catalog,
    )

    result = evaluate_ranker(model, held_out, k=2)

    assert result.per_user.n_relevant.tolist() == [2]
    assert result.per_user.hits.tolist() == [2]
    assert result.metrics == pytest.approx({"recall_at_k": 1.0, "ndcg_at_k": 1.0})


@pytest.mark.parametrize(("prediction", "message"), [
    pytest.param(np.array(["a", "a"]), "unique", id="duplicate"),
    pytest.param(np.array(["a", "unknown"]), "catalog", id="outside-catalog"),
    pytest.param(np.array(["a", 7], dtype=object), "string", id="nonstring"),
    pytest.param(["a", "b"], "array", id="not-an-array"),
    pytest.param(np.array(["a"]), "shape", id="short"),
    pytest.param(np.array(["a", "b", "c"]), "shape", id="long"),
    pytest.param(np.array([["a", "b"]]), "shape", id="two-dimensional"),
])
def test_invalid_predictions_are_rejected(held_out, prediction, message):
    model = PredictOnlyRanker({"u": prediction})
    with pytest.raises(ValueError, match=message):
        evaluate_ranker(model, held_out, k=2)
    assert model.calls == [("u", 2)]


@pytest.mark.parametrize("k", [0, -1, 6, True, 1.5, "2", None])
def test_invalid_k_is_rejected_before_predicting(held_out, k):
    model = PredictOnlyRanker({"u": np.array(["a", "b"])})
    with pytest.raises(ValueError, match="k must"):
        evaluate_ranker(model, held_out, k=k)
    assert model.calls == []


def test_empty_holdout_is_rejected_before_predicting(held_out):
    model = PredictOnlyRanker({})
    empty = RankingData(held_out.interactions.iloc[:0], held_out.catalog)
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_ranker(model, empty, k=1)
    assert model.calls == []
