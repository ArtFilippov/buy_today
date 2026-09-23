from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

from buy_today.clustering.evaluation import evaluate_clustering
from buy_today.clustering.contracts import FeatureMatrix


class DistanceOnly:
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[FeatureMatrix] = []

    def fit(self, *args: object) -> None:
        raise AssertionError("Evaluation must not fit distances")

    def pairwise(self, features: FeatureMatrix) -> NDArray[Any]:
        self.inputs.append(features.copy())
        values = np.asarray(features)[:, 0]
        return np.abs(values[:, None] - values[None, :])


@pytest.mark.parametrize("as_frame", [False, True])
def test_independent_distance_only_evaluation_and_known_silhouette(as_frame: bool) -> None:
    values = np.array([[0], [1], [10], [11]])
    features = pd.DataFrame(values, columns=["arbitrary_feature"]) if as_frame else values
    distance = DistanceOnly()
    result = evaluate_clustering(
        X=features, labels=[3, 3, 9, 9], distance=distance, new_rows=[True] * 4
    )
    assert len(distance.inputs) == 1
    assert result.silhouette == pytest.approx((1 - 1 / 10.5 + 1 - 1 / 9.5) / 2)
    assert result.silhouette == silhouette_score(
        result.distances, result.labels, metric="precomputed"
    )
    assert result.silhouette_reason is None
    np.testing.assert_array_equal(result.sample_positions, np.arange(4))


@pytest.mark.parametrize(
    ("new", "old", "limit", "expected_new", "expected_old"),
    [
        (1300, 2200, 1000, 500, 500),
        (20, 2200, 1000, 20, 980),
        (2200, 20, 1000, 980, 20),
        (1500, 0, 1000, 1000, 0),
        (0, 1500, 1000, 0, 1000),
        (1200, 1200, 999, 500, 499),
        (3, 2, 1000, 3, 2),
        (3, 2, 1, 1, 0),
    ],
)
def test_quota_reproducibility_alignment_and_sampling_before_distances(
    new: int, old: int, limit: int, expected_new: int, expected_old: int
) -> None:
    features = np.arange(new + old).reshape(-1, 1)
    labels = np.arange(new + old) % 3
    mask = np.arange(new + old) < new
    distance = DistanceOnly()
    first = evaluate_clustering(
        features, labels, distance=distance, new_rows=mask, max_evaluation_rows=limit
    )
    second = evaluate_clustering(
        features, labels[::-1], distance=distance, new_rows=mask, max_evaluation_rows=limit
    )
    np.testing.assert_array_equal(first.sample_positions, second.sample_positions)
    assert first.new_count == expected_new
    assert first.previous_count == expected_old
    assert len(np.unique(first.sample_positions)) == expected_new + expected_old
    assert first.distances.shape == (expected_new + expected_old,) * 2
    assert len(distance.inputs) == 2
    np.testing.assert_array_equal(distance.inputs[0], features[first.sample_positions])
    np.testing.assert_array_equal(first.labels, labels[first.sample_positions])
    np.testing.assert_array_equal(first.new_rows, mask[first.sample_positions])


@pytest.mark.parametrize("labels", [[1], [0, 0, 0], [0, 1, 2], [0, 1]])
def test_undefined_silhouette_has_reason(labels: list[int]) -> None:
    result = evaluate_clustering(
        np.arange(len(labels)).reshape(-1, 1),
        labels,
        distance=DistanceOnly(),
        new_rows=[True] * len(labels),
    )
    assert result.silhouette is None
    assert result.silhouette_reason is not None
    assert "Силуэт не определён" in result.silhouette_reason
    assert result.distances.shape == (len(labels), len(labels))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"labels": [1.0, 1.0, 2.0]},
        {"labels": [True, False, True]},
        {"labels": [0, 1]},
        {"labels": [[0], [0], [1]]},
        {"new_rows": [0, 1, 1]},
        {"new_rows": [True]},
        {"max_evaluation_rows": 0},
        {"max_evaluation_rows": 1.5},
        {"max_evaluation_rows": True},
        {"max_evaluation_rows": np.bool_(True)},
        {"random_state": -1},
        {"random_state": 2**32},
        {"random_state": True},
        {"random_state": np.bool_(False)},
        {"X": np.array([])},
        {"X": np.ones(3)},
        {"X": np.ones((0, 1))},
    ],
)
def test_invalid_inputs_are_errors_before_pairwise(kwargs: dict[str, Any]) -> None:
    distance = DistanceOnly()
    args: dict[str, Any] = {
        "X": np.arange(3).reshape(-1, 1),
        "labels": [0, 0, 1],
        "new_rows": [True] * 3,
        "distance": distance,
    }
    args.update(kwargs)
    with pytest.raises(ValueError):
        evaluate_clustering(**args)
    assert not distance.inputs


@pytest.mark.parametrize("name", ["max_evaluation_row", "thread_limit"])
def test_unknown_options_are_rejected_before_pairwise(name: str) -> None:
    distance = DistanceOnly()
    options: dict[str, Any] = {name: 3}
    with pytest.raises(TypeError, match=name):
        evaluate_clustering(
            np.ones((3, 1)), [0, 0, 1], distance=distance, new_rows=[True] * 3, **options
        )
    assert not distance.inputs


def test_numpy_integer_sampling_parameters_remain_supported() -> None:
    options: dict[str, Any] = {"max_evaluation_rows": np.int64(4), "random_state": np.int64(17)}
    result = evaluate_clustering(
        np.arange(12).reshape(-1, 1),
        np.arange(12) // 4,
        distance=DistanceOnly(),
        new_rows=np.arange(12) >= 6,
        **options,
    )
    assert len(result.sample_positions) == 4
    assert result.new_count == result.previous_count == 2


@pytest.mark.parametrize(
    "matrix",
    [
        [[0, 1]],
        [[0, -1], [-1, 0]],
        [[0, np.nan], [np.nan, 0]],
        [[0, np.inf], [np.inf, 0]],
        [[0, 2], [1, 0]],
        [[1, 2], [2, 0]],
        [[0, 1j], [1j, 0]],
        [["0", "1"], ["1", "0"]],
    ],
)
def test_invalid_distances_are_errors_even_with_one_label(matrix: ArrayLike) -> None:
    class InvalidDistance:
        def pairwise(self, features: FeatureMatrix) -> ArrayLike:
            return matrix

    with pytest.raises(ValueError):
        evaluate_clustering(
            np.ones((2, 1)), [0, 0], distance=InvalidDistance(), new_rows=[True, True]
        )


def test_technical_errors_propagate() -> None:
    class BrokenDistance:
        def pairwise(self, features: FeatureMatrix) -> ArrayLike:
            raise RuntimeError("distance unavailable")

    with pytest.raises(RuntimeError, match="distance unavailable"):
        evaluate_clustering(
            np.ones((3, 1)), [0, 0, 1], distance=BrokenDistance(), new_rows=[True] * 3
        )
