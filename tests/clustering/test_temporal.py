from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from buy_today.clustering.distances import TimestampDistance
from buy_today.clustering.models.temporal import TemporalClustering
from buy_today.schema import SORT_KEY


def test_sorted_equal_groups_preserve_input_order_and_do_not_mutate(working_frame: pd.DataFrame):
    # All ties: both parts of the row key must decide the order, not the index.
    working_frame[SORT_KEY[0]] = pd.Timestamp("2018-01-01")
    frame = working_frame.sample(frac=1, random_state=7)
    frame.index = [99] * len(frame)
    before = frame.copy(deep=True)
    model = TemporalClustering(n_clusters=5)
    actual = model.fit_predict(frame)
    expected_by_key = dict(
        zip(
            working_frame[["order_id", "order_item_id"]].itertuples(index=False, name=None),
            [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
        )
    )
    expected = [
        expected_by_key[key]
        for key in frame[["order_id", "order_item_id"]].itertuples(index=False, name=None)
    ]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, model.labels_)
    pd.testing.assert_frame_equal(frame, before)


def test_time_precedes_keys_and_refit_replaces_state(working_frame: pd.DataFrame):
    frame = working_frame.copy()
    frame[SORT_KEY[0]] = frame[SORT_KEY[0]].iloc[::-1].to_numpy()
    model = TemporalClustering(n_clusters=3)
    assert model.fit(frame) is model
    np.testing.assert_array_equal(model.labels_, [2] * 4 + [1] * 4 + [0] * 4)
    assert model.n_features_in_ == 35
    np.testing.assert_array_equal(model.feature_names_in_, frame.columns)
    assert model.get_params() == {"n_clusters": 3}
    copied = clone(model)
    assert not hasattr(copied, "labels_")
    assert copied.get_params() == model.get_params()
    assert model.set_params(n_clusters=2) is model
    model.fit(frame.iloc[:5, :2].assign(order_purchase_timestamp=frame[SORT_KEY[0]].iloc[:5]))
    assert model.n_features_in_ == 3
    assert len(model.feature_names_in_) == 3
    assert sorted(np.bincount(model.labels_)) == [2, 3]


@pytest.mark.parametrize("n_clusters", [0, -1, 1.5, "2", True, np.bool_(True), 13])
def test_invalid_number_of_clusters(working_frame: pd.DataFrame, n_clusters: Any):
    model = TemporalClustering(n_clusters=n_clusters)
    with pytest.raises(ValueError):
        model.fit(working_frame)


@pytest.mark.parametrize(
    "problem", ["missing_time", "nat", "text_time", "duplicate", "missing_key"]
)
def test_invalid_temporal_data(working_frame: pd.DataFrame, problem: str):
    if problem == "missing_time":
        working_frame = working_frame.drop(columns=SORT_KEY[0])
    elif problem == "nat":
        working_frame.loc[0, SORT_KEY[0]] = pd.NaT
    elif problem == "text_time":
        working_frame[SORT_KEY[0]] = "2018-01-01"
    elif problem == "duplicate":
        working_frame.iloc[1] = working_frame.iloc[0]
    else:
        working_frame.loc[0, "order_id"] = pd.NA
    with pytest.raises(ValueError):
        TemporalClustering(n_clusters=2).fit(working_frame)


def test_distance_is_noop_clonable_and_works_without_fit():
    distance = TimestampDistance()
    assert distance.fit(object(), y=object()) is distance
    assert vars(distance) == {}
    assert clone(distance).get_params() == {}
    x = pd.DataFrame(
        {
            SORT_KEY[0]: pd.to_datetime(
                [
                    "2018-01-01 00:00:04",
                    "2018-01-01 00:00:00",
                    "2018-01-01 00:00:04",
                ]
            )
        }
    )
    y = pd.DataFrame(
        {
            SORT_KEY[0]: pd.to_datetime(
                [
                    "2018-01-02 00:00:00",
                    "2017-12-31 23:59:59",
                ]
            )
        }
    )
    matrix = TimestampDistance().pairwise(x)
    assert matrix.dtype == np.float64
    np.testing.assert_array_equal(matrix, [[0, 4, 0], [4, 0, 4], [0, 4, 0]])
    np.testing.assert_array_equal(distance.pairwise(x, y), [[86396, 5], [86400, 1], [86396, 5]])
    assert distance.pairwise(x.iloc[:0], y).shape == (0, 2)


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_distance_respects_datetime_units(unit: str):
    frame = pd.DataFrame(
        {SORT_KEY[0]: np.array(["2018-01-01", "2018-01-02"], dtype=f"datetime64[{unit}]")}
    )
    assert TimestampDistance().pairwise(frame)[0, 1] == 86400


@pytest.mark.parametrize(
    "values", [[pd.NaT], ["invalid"], [1], pd.date_range("2018-01-01", periods=2, tz="UTC")]
)
def test_distance_rejects_invalid_dates(values: object):
    with pytest.raises(ValueError):
        TimestampDistance().pairwise(pd.DataFrame({SORT_KEY[0]: values}))
