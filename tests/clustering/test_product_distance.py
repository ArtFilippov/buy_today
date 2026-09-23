from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from buy_today.clustering.distances import CategoryPriceDistance
from buy_today.generation.histories import generate_histories


def test_distance_is_scale_invariant_symmetric_and_rectangular():
    frame = pd.DataFrame({"product_category_name": ["a", "a", "b"], "price": [10., 20., 10.]})
    distance = CategoryPriceDistance()
    expected = np.array(
        [[0, np.log(2), 1], [np.log(2), 0, 1 + np.log(2)], [1, 1 + np.log(2), 0]]
    )
    np.testing.assert_allclose(distance.pairwise(frame), expected)
    np.testing.assert_allclose(distance.pairwise(frame.iloc[[2, 0]], frame), expected[[2, 0]])
    np.testing.assert_allclose(distance.pairwise(frame.assign(price=frame.price * 100)), expected)
    assert distance.pairwise(frame.iloc[:0], frame).shape == (0, 3)
    assert clone(distance.fit(frame)).get_params() == {}


@pytest.mark.parametrize("price", [0., -1., np.nan, np.inf, "10", True, 1j])
def test_distance_rejects_undefined_price_geometry(price: object):
    frame = pd.DataFrame({"product_category_name": ["a"], "price": [price]})
    with pytest.raises(ValueError):
        CategoryPriceDistance().pairwise(frame)


@pytest.mark.parametrize("category", [None, "", " ", 123])
def test_distance_rejects_invalid_category(category: object):
    frame = pd.DataFrame({"product_category_name": [category], "price": [10.]})
    with pytest.raises(ValueError):
        CategoryPriceDistance().pairwise(frame)


def test_serialized_distance_reproduces_histories(working_frame: pd.DataFrame, tmp_path: Path):
    distance = CategoryPriceDistance()
    joblib.dump(distance, tmp_path / "distance.joblib")
    first = generate_histories(
        working_frame, distance=distance, batch_index=0, n_users=3, temperature=.1, random_state=73,
    )
    second = generate_histories(
        working_frame, distance=joblib.load(tmp_path / "distance.joblib"),
        batch_index=0, n_users=3, temperature=.1, random_state=73,
    )
    pd.testing.assert_frame_equal(first.events, second.events)
    pd.testing.assert_frame_equal(first.anchors, second.anchors)
