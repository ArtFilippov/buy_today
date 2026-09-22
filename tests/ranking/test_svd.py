import json
import os
import subprocess
import sys
import textwrap

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from prak.ranking import RankingData, SVDRanker


COUNTS = np.array([
    [4, 1, 0, 1, 0, 0],
    [3, 2, 0, 0, 0, 0],
    [0, 0, 3, 1, 2, 0],
    [0, 1, 4, 0, 1, 0],
])


def purchases(counts):
    return RankingData(
        pd.DataFrame([
            (f"u{user}", f"p{item}")
            for user, row in enumerate(counts)
            for item, count in enumerate(row) for _ in range(count)
        ], columns=["user_id", "product_id"]),
        pd.DataFrame({"product_id": [f"p{i}" for i in range(len(counts[0]))]}),
    )


@pytest.fixture
def training_data():
    return purchases(COUNTS)


def test_truncated_scores_and_rankings_match_dense_svd(training_data):
    model = SVDRanker(n_components=2).fit(training_data)
    left, singular, right = np.linalg.svd(COUNTS.astype(float), full_matrices=False)
    expected = (left[:, :2] * singular[:2]) @ right[:2]
    actual = model.user_factors_ @ model.item_factors_.T
    np.testing.assert_allclose(actual, expected, atol=2e-5)
    assert model.user_factors_.shape == (4, 2)
    assert model.item_factors_.shape == (6, 2)
    assert model.user_factors_.dtype == model.item_factors_.dtype == np.float32
    for index, user in enumerate(model.user_ids_):
        order = np.argsort(-expected[index], kind="stable")
        np.testing.assert_array_equal(model.predict(user, 3), model.catalog_[order[:3]])


def test_repeated_purchases_are_counts_and_bought_items_remain_candidates():
    # Rank one, so the approximation must recover these counts exactly.
    counts = [[2, 6, 0, 0], [1, 3, 0, 0]]
    model = SVDRanker(n_components=1).fit(purchases(counts))
    np.testing.assert_allclose(model.user_factors_ @ model.item_factors_.T, counts, atol=1e-5)
    for user in model.user_ids_:
        np.testing.assert_array_equal(model.predict(user, 4), ["p1", "p0", "p2", "p3"])


def test_exact_score_ties_negative_scores_and_k_boundary(training_data):
    model = SVDRanker(n_components=2).fit(training_data)
    # Controlled factors isolate the ranking policy from numerical SVD error.
    model.user_factors_[0] = [2, 1]
    model.item_factors_[:] = [[1, 1], [0, 3], [2, -1], [0, -1], [-1, 0], [0, 0]]
    expected = ["p0", "p1", "p2", "p5", "p3", "p4"]
    for k in (6, 2, 3, 1, 5):
        np.testing.assert_array_equal(model.predict("u0", k), expected[:k])


@pytest.mark.parametrize("counts,n_components", [
    ([[3, 1, 0]], 1),                 # single user
    ([[3, 1, 0], [3, 1, 0]], 2),     # zero variance, rank deficient
    ([[3, 1], [1, 4], [2, 1]], 2),   # full item dimension
    (COUNTS, 4),                     # full user dimension
])
def test_small_and_full_dimension_matrices(counts, n_components):
    model = SVDRanker(n_components=n_components).fit(purchases(counts))
    actual = model.user_factors_ @ model.item_factors_.T
    np.testing.assert_allclose(actual, counts, atol=2e-5)
    cold = np.asarray(counts).sum(axis=0) == 0
    np.testing.assert_array_equal(model.item_factors_[cold], 0)
    assert np.isfinite(actual).all()
    for user in model.user_ids_:
        assert set(model.predict(user, len(model.catalog_))) == set(model.catalog_)


def test_sklearn_parameters_clone_and_refit_replace_state(training_data):
    model = SVDRanker()
    assert model.get_params() == {"n_components": 32, "n_iter": 7, "random_state": 42}
    with pytest.raises(NotFittedError):
        model.predict("u0", 1)
    assert model.set_params(n_components=2, n_iter=5, random_state=19) is model
    assert model.fit(training_data, y=np.zeros(len(training_data.interactions))) is model
    check_is_fitted(model)
    copied = clone(model)
    assert copied.get_params() == model.get_params()
    with pytest.raises(NotFittedError):
        copied.predict("u0", 1)
    copied.fit(training_data)
    np.testing.assert_array_equal(copied.predict("u0", 6), model.predict("u0", 6))

    replacement = RankingData(
        pd.DataFrame({"user_id": ["новый", "новый"], "product_id": ["x", "y"]}),
        pd.DataFrame({"product_id": ["z", "y", "x"]}),
    )
    assert model.set_params(n_components=1, random_state=23).fit(replacement) is model
    fresh = clone(model).fit(replacement)
    assert model.user_factors_.shape == (1, 1)
    assert model.item_factors_.shape == (3, 1)
    np.testing.assert_array_equal(model.predict("новый", 3), fresh.predict("новый", 3))
    with pytest.raises(ValueError, match="Unknown user"):
        model.predict("u0", 1)


def test_row_and_call_order_reproducibility_and_input_isolation(training_data):
    interactions, catalog = training_data.interactions.copy(), training_data.catalog.copy()
    original = SVDRanker(n_components=2).fit(training_data)
    shuffled = SVDRanker(n_components=2).fit(RankingData(
        interactions.sample(frac=1, random_state=17), catalog.iloc[::-1],
    ))
    np.testing.assert_array_equal(original.user_factors_, shuffled.user_factors_)
    np.testing.assert_array_equal(original.item_factors_, shuffled.item_factors_)
    pd.testing.assert_frame_equal(training_data.interactions, interactions)
    pd.testing.assert_frame_equal(training_data.catalog, catalog)
    expected = {user: original.predict(user, 6) for user in original.user_ids_}
    training_data.interactions.loc[:, "user_id"] = "replacement"
    training_data.catalog.loc[:, "product_id"] = "replacement"
    original.predict("u0", 6)[:] = "changed by caller"
    for user in reversed(original.user_ids_):
        for k in (2, 6, 1, 4):
            np.testing.assert_array_equal(original.predict(user, k), expected[user][:k])
            np.testing.assert_array_equal(shuffled.predict(user, k), expected[user][:k])


@pytest.mark.parametrize("parameter,value", [
    ("n_components", 0), ("n_components", 5), ("n_components", 1.5),
    ("n_components", True), ("n_components", None),
    ("n_iter", 0), ("n_iter", -1), ("n_iter", 2.5), ("n_iter", "7"),
    ("n_iter", np.bool_(True)), ("n_iter", None),
    ("random_state", -1), ("random_state", 2**32), ("random_state", True),
    ("random_state", 1.5), ("random_state", None),
])
def test_invalid_parameters_fail_on_fit(training_data, parameter, value):
    model = SVDRanker(n_components=2).set_params(**{parameter: value})
    with pytest.raises(ValueError, match=parameter):
        model.fit(training_data)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_numpy_integer_parameters_and_seed_boundaries(training_data, seed):
    model = SVDRanker(
        n_components=np.int64(2), n_iter=np.int64(7), random_state=np.int64(seed),
    ).fit(training_data)
    assert len(model.predict("u0", np.int64(3))) == 3


def test_dimension_limit_checks_both_axes_and_single_item_catalog():
    with pytest.raises(ValueError, match="n_components"):
        SVDRanker(n_components=3).fit(purchases([[1, 2], [2, 1], [1, 1]]))
    with pytest.raises(ValueError, match="at least two products"):
        SVDRanker(n_components=1).fit(purchases([[1], [2]]))


@pytest.mark.parametrize("k", [0, -1, 7, True, np.bool_(True), 1.5, "3", None])
def test_invalid_k(training_data, k):
    model = SVDRanker(n_components=2).fit(training_data)
    with pytest.raises(ValueError, match="k must"):
        model.predict("u0", k)


@pytest.mark.parametrize("user", ["unknown", 0, None, ["u0"]])
def test_unknown_or_nonstring_user(training_data, user):
    model = SVDRanker(n_components=2).fit(training_data)
    with pytest.raises(ValueError, match="Unknown user"):
        model.predict(user, 1)


def test_fit_validates_ranking_data(training_data):
    with pytest.raises(ValueError, match="RankingData"):
        SVDRanker(n_components=2).fit(training_data.interactions)
    with pytest.raises(ValueError, match="outside catalog"):
        SVDRanker(n_components=2).fit(RankingData(
            training_data.interactions, training_data.catalog.iloc[1:],
        ))


def test_persistence_and_fresh_fit_across_process_hash_seeds(training_data, tmp_path):
    model = SVDRanker(n_components=2).fit(training_data)
    expected = {user: model.predict(user, 6).tolist() for user in model.user_ids_}
    path = tmp_path / "svd.joblib"
    joblib.dump((model, training_data), path)
    script = textwrap.dedent("""
        import json
        import sys
        import joblib
        from sklearn.base import clone

        restored, data = joblib.load(sys.argv[1])
        fresh = clone(restored).fit(data)
        print(json.dumps([
            {user: model.predict(user, len(data.catalog)).tolist()
             for user in reversed(model.user_ids_)}
            for model in (restored, fresh)
        ]))
    """)
    for seed in ("1", "98765"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)], cwd=tmp_path,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True, text=True, check=True, timeout=30,
        )
        assert json.loads(result.stdout) == [expected, expected]
