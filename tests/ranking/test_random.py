import json
import os
import subprocess
import sys
import textwrap

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
import fixture_types as ft
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from buy_today.ranking.data import RankingData
from buy_today.ranking.random import RandomRanker


@pytest.fixture
def training_data() -> RankingData:
    return RankingData(
        pd.DataFrame(
            {
                "user_id": ["007", "NA", "007", "пользователь"],
                "product_id": ["item_00", "item_01", "item_00", "item_02"],
            }
        ),
        pd.DataFrame({"product_id": [f"item_{index:02d}" for index in range(20)]}),
    )


def test_sklearn_parameters_fit_and_clone(training_data: RankingData):
    model = RandomRanker()
    assert model.get_params() == {"random_state": 42}
    assert model.set_params(random_state=19) is model
    with pytest.raises(NotFittedError):
        model.predict("007", 1)

    assert model.fit(training_data, y=np.zeros(len(training_data.interactions))) is model
    check_is_fitted(model)
    assert model.get_params() == {"random_state": 19}
    copied = clone(model)
    assert copied is not model
    assert copied.get_params() == model.get_params()
    with pytest.raises(NotFittedError):
        copied.predict("007", 1)
    copied.fit(training_data)
    np.testing.assert_array_equal(copied.predict("007", 20), model.predict("007", 20))


def test_refit_replaces_users_catalog_and_seed(training_data: RankingData):
    model = RandomRanker().fit(training_data)
    replacement = RankingData(
        pd.DataFrame({"user_id": ["007", "new"], "product_id": ["x", "y"]}),
        pd.DataFrame({"product_id": ["x", "y", "z"]}),
    )

    assert model.set_params(random_state=23).fit(replacement) is model
    fresh = RandomRanker(random_state=23).fit(replacement)
    for user in ("007", "new"):
        assert set(model.predict(user, 3)) == {"x", "y", "z"}
        np.testing.assert_array_equal(model.predict(user, 3), fresh.predict(user, 3))
    with pytest.raises(ValueError, match="Unknown user"):
        model.predict("NA", 1)


def test_per_user_permutations_and_prefixes_are_independent_of_call_order(
    training_data: RankingData,
):
    first = RandomRanker().fit(training_data)
    second = RandomRanker().fit(training_data)
    users = training_data.interactions.user_id.unique().tolist()
    size = len(training_data.catalog)
    full: dict[str, NDArray[np.object_]] = {}
    for user in users:
        prefix = first.predict(user, 5)
        recommendations = first.predict(user, size)
        assert isinstance(recommendations, np.ndarray)
        assert recommendations.shape == (size,)
        assert len(set(recommendations)) == size
        # Includes both previously bought products and products never bought by anyone.
        assert set(recommendations) == set(training_data.catalog.product_id)
        np.testing.assert_array_equal(prefix, recommendations[:5])
        full[user] = recommendations

    assert len({tuple(items) for items in full.values()}) == len(users)
    for user in reversed(users):
        for k in (1, 7, size, 3):
            np.testing.assert_array_equal(second.predict(user, k), full[user][:k])
            np.testing.assert_array_equal(first.predict(user, k), full[user][:k])


def test_input_row_order_does_not_change_rankings_or_mutate_inputs(training_data: RankingData):
    interactions = training_data.interactions.copy(deep=True)
    catalog = training_data.catalog.copy(deep=True)
    reordered = RankingData(
        interactions.sample(frac=1, random_state=17),
        catalog.iloc[::-1],
    )
    original = RandomRanker().fit(training_data)
    shuffled = RandomRanker().fit(reordered)

    for user in interactions.user_id.unique():
        np.testing.assert_array_equal(original.predict(user, 20), shuffled.predict(user, 20))
    pd.testing.assert_frame_equal(training_data.interactions, interactions)
    pd.testing.assert_frame_equal(training_data.catalog, catalog)


def test_fitted_model_is_isolated_from_training_tables_and_returned_arrays(
    training_data: RankingData,
):
    model = RandomRanker().fit(training_data)
    expected = {
        user: model.predict(user, 20) for user in training_data.interactions.user_id.unique()
    }
    training_data.interactions.loc[:, "user_id"] = "replacement-user"
    training_data.interactions.loc[:, "product_id"] = "replacement-product"
    training_data.catalog.loc[:, "product_id"] = "replacement-product"
    returned = model.predict("007", 20)
    returned[:] = "changed-by-caller"

    for user, recommendations in expected.items():
        np.testing.assert_array_equal(model.predict(user, 20), recommendations)
    with pytest.raises(ValueError, match="Unknown user"):
        model.predict("replacement-user", 1)


def test_changing_seed_changes_rankings(training_data: RankingData):
    first = RandomRanker(random_state=42).fit(training_data)
    changed = RandomRanker(random_state=43).fit(training_data)
    for user in training_data.interactions.user_id.unique():
        assert not np.array_equal(first.predict(user, 20), changed.predict(user, 20))


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_numpy_integer_parameters_and_seed_boundaries(training_data: RankingData, seed: int):
    model = RandomRanker(random_state=np.int64(seed)).fit(training_data)
    assert set(model.predict("007", np.int64(20))) == set(training_data.catalog.product_id)


@pytest.mark.parametrize("seed", [None, -1, 2**32, True, np.bool_(False), 1.5, "42"])
def test_invalid_random_state_is_rejected_on_fit(training_data: RankingData, seed: object):
    with pytest.raises(ValueError, match="random_state"):
        RandomRanker(random_state=seed).fit(training_data)


@pytest.mark.parametrize("k", [0, -1, 21, True, np.bool_(True), 1.5, "3", None])
def test_invalid_k_is_rejected(training_data: RankingData, k: object):
    model = RandomRanker().fit(training_data)
    with pytest.raises(ValueError, match="k must"):
        model.predict("007", k)


@pytest.mark.parametrize("user", ["unknown", "7", 7, None])
def test_unknown_or_nonstring_user_is_rejected(training_data: RankingData, user: object):
    model = RandomRanker().fit(training_data)
    with pytest.raises(ValueError, match="Unknown user"):
        model.predict(user, 1)


def test_fit_validates_catalog_inclusion(training_data: RankingData):
    invalid = RankingData(training_data.interactions, training_data.catalog.iloc[1:])
    with pytest.raises(ValueError, match="outside catalog"):
        RandomRanker().fit(invalid)


def test_persistence_and_fresh_fit_reproduce_rankings_across_process_hash_seeds(
    training_data: RankingData, tmp_path: ft.Path
):
    model = RandomRanker().fit(training_data)
    model.predict("007", 2)
    expected = {
        user: model.predict(user, 20).tolist()
        for user in training_data.interactions.user_id.unique()
    }
    path = tmp_path / "ranker.joblib"
    joblib.dump((model, training_data), path)
    script = textwrap.dedent("""
        import json
        import sys

        import joblib
        from sklearn.base import clone

        restored, data = joblib.load(sys.argv[1])
        fresh = clone(restored).fit(data)
        users = data.interactions.user_id.unique().tolist()[::-1]
        print(json.dumps([
            {user: model.predict(user, len(data.catalog)).tolist() for user in users}
            for model in (restored, fresh)
        ]))
    """)

    for hash_seed in ("1", "98765"):
        completed = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            cwd=tmp_path,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        assert json.loads(completed.stdout) == [expected, expected]
