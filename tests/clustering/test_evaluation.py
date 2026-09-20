import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

from prak.clustering.evaluation import evaluate_clustering


class DistanceOnly:
    def __init__(self):
        self.inputs = []

    def fit(self, *args):
        raise AssertionError("Evaluation must not fit distances")

    def pairwise(self, X):
        self.inputs.append(X.copy())
        values = np.asarray(X)[:, 0]
        return np.abs(values[:, None] - values[None, :])


@pytest.mark.parametrize("as_frame", [False, True])
def test_independent_distance_only_evaluation_and_known_silhouette(as_frame):
    values = np.array([[0], [1], [10], [11]])
    X = pd.DataFrame(values, columns=["arbitrary_feature"]) if as_frame else values
    distance = DistanceOnly()
    result = evaluate_clustering(X, [3, 3, 9, 9], distance=distance, new_rows=[True] * 4)
    assert len(distance.inputs) == 1
    assert result.silhouette == pytest.approx((1 - 1 / 10.5 + 1 - 1 / 9.5) / 2)
    assert result.silhouette == silhouette_score(result.distances, result.labels, metric="precomputed")
    assert result.silhouette_reason is None
    np.testing.assert_array_equal(result.sample_positions, np.arange(4))


@pytest.mark.parametrize(("new", "old", "limit", "expected_new", "expected_old"), [
    (1300, 2200, 1000, 500, 500),
    (20, 2200, 1000, 20, 980),
    (2200, 20, 1000, 980, 20),
    (1500, 0, 1000, 1000, 0),
    (0, 1500, 1000, 0, 1000),
    (1200, 1200, 999, 500, 499),
    (3, 2, 1000, 3, 2),
    (3, 2, 1, 1, 0),
])
def test_quota_reproducibility_alignment_and_sampling_before_distances(new, old, limit, expected_new, expected_old):
    X = np.arange(new + old).reshape(-1, 1)
    labels = np.arange(new + old) % 3
    mask = np.arange(new + old) < new
    distance = DistanceOnly()
    args = dict(distance=distance, new_rows=mask, max_evaluation_rows=limit)
    first = evaluate_clustering(X, labels, **args)
    second = evaluate_clustering(X, labels[::-1], **args)
    np.testing.assert_array_equal(first.sample_positions, second.sample_positions)
    assert first.new_count == expected_new
    assert first.previous_count == expected_old
    assert len(np.unique(first.sample_positions)) == expected_new + expected_old
    assert first.distances.shape == (expected_new + expected_old,) * 2
    assert len(distance.inputs) == 2
    np.testing.assert_array_equal(distance.inputs[0], X[first.sample_positions])
    np.testing.assert_array_equal(first.labels, labels[first.sample_positions])
    np.testing.assert_array_equal(first.new_rows, mask[first.sample_positions])


@pytest.mark.parametrize("labels", [[1], [0, 0, 0], [0, 1, 2], [0, 1]])
def test_undefined_silhouette_has_reason(labels):
    result = evaluate_clustering(
        np.arange(len(labels)).reshape(-1, 1), labels, distance=DistanceOnly(), new_rows=[True] * len(labels),
    )
    assert result.silhouette is None
    assert "Силуэт не определён" in result.silhouette_reason
    assert result.distances.shape == (len(labels), len(labels))


@pytest.mark.parametrize("kwargs", [
    {"labels": [1.0, 1.0, 2.0]}, {"labels": [True, False, True]},
    {"labels": [0, 1]}, {"labels": [[0], [0], [1]]},
    {"new_rows": [0, 1, 1]}, {"new_rows": [True]},
    {"max_evaluation_rows": 0}, {"max_evaluation_rows": 1.5}, {"max_evaluation_rows": True},
    {"random_state": -1}, {"random_state": 2**32}, {"random_state": True},
    {"X": np.array([])}, {"X": np.ones(3)}, {"X": np.ones((0, 1))},
])
def test_invalid_inputs_are_errors_before_pairwise(kwargs):
    distance = DistanceOnly()
    args = dict(X=np.arange(3).reshape(-1, 1), labels=[0, 0, 1], new_rows=[True] * 3, distance=distance)
    args.update(kwargs)
    with pytest.raises(ValueError):
        evaluate_clustering(**args)
    assert distance.inputs == []


@pytest.mark.parametrize("matrix", [
    [[0, 1]], [[0, -1], [-1, 0]], [[0, np.nan], [np.nan, 0]],
    [[0, np.inf], [np.inf, 0]], [[0, 2], [1, 0]], [[1, 2], [2, 0]],
    [[0, 1j], [1j, 0]], [["0", "1"], ["1", "0"]],
])
def test_invalid_distances_are_errors_even_with_one_label(matrix):
    class InvalidDistance:
        def pairwise(self, X):
            return matrix

    with pytest.raises(ValueError):
        evaluate_clustering(np.ones((2, 1)), [0, 0], distance=InvalidDistance(), new_rows=[True, True])


def test_technical_errors_propagate():
    class BrokenDistance:
        def pairwise(self, X):
            raise RuntimeError("distance unavailable")

    with pytest.raises(RuntimeError, match="distance unavailable"):
        evaluate_clustering(np.ones((3, 1)), [0, 0, 1], distance=BrokenDistance(), new_rows=[True] * 3)
