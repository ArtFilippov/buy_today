from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from prak.auto_eda.drift import (
    DriftResult,
    DriftThresholds,
    evaluate_drift,
    ks_distance,
    total_variation_distance,
)


def test_threshold_defaults_are_immutable():
    thresholds = DriftThresholds()
    assert (thresholds.price, thresholds.category, thresholds.state) == (0.10,) * 3
    with pytest.raises(FrozenInstanceError):
        thresholds.price = 0.5


@pytest.mark.parametrize("name", ["price", "category", "state"])
@pytest.mark.parametrize("value", [0, 0.25, np.float64(0.5), 1])
def test_valid_thresholds_include_endpoints(name, value):
    assert getattr(DriftThresholds(**{name: value}), name) == value


@pytest.mark.parametrize("name", ["price", "category", "state"])
@pytest.mark.parametrize("value", [
    np.nan, np.inf, -np.inf, -0.01, 1.01, None, "0.1", 0.1j,
])
def test_invalid_thresholds_raise_value_error(name, value):
    with pytest.raises(ValueError, match=rf"Порог {name}.*конечное число в \[0, 1\]"):
        DriftThresholds(**{name: value})


@pytest.mark.parametrize(("reference", "batch", "expected"), [
    ([1, 2, 2, 4], [2, 4, 2, 1], 0.0),
    ([3, 1, 1], [1, 3, 1, 1, 3, 1], 0.0),
    ([1, 1], [1, 1, 1], 0.0),
    ([1, 1, 2], [1, 2, 2, 3], 5 / 12),
    ([1, 2], [3, 4, 4], 1.0),
    ([1, 2], [2, 3], 0.5),
    ([1] * 6 + [2] * 4, [1] * 5 + [2] * 5, 0.1),
])
def test_ks_distance_with_ties_unequal_sizes_and_shifts(reference, batch, expected):
    reference, batch = pd.Series(reference), pd.Series(batch)
    assert ks_distance(reference, batch) == pytest.approx(expected)
    assert ks_distance(batch, reference) == pytest.approx(expected)


@pytest.mark.parametrize(("reference", "batch", "expected"), [
    (["a", "a", "b"], ["b", "a", "a", "a", "b", "a"], 0.0),
    (["a", "a"], ["a", "a", "a"], 0.0),
    (["a", "a", "b"], ["a", "c"], 0.5),
    (["a", "b"], ["c", "c", "d"], 1.0),
    (["a"] * 6 + ["b"] * 4, ["a"] * 5 + ["b"] * 5, 0.1),
])
def test_tvd_uses_each_samples_size_and_union_of_values(reference, batch, expected):
    reference, batch = pd.Series(reference), pd.Series(batch)
    assert total_variation_distance(reference, batch) == pytest.approx(expected)
    assert total_variation_distance(batch, reference) == pytest.approx(expected)


@pytest.mark.parametrize("distance", [ks_distance, total_variation_distance])
@pytest.mark.parametrize("empty_reference", [False, True])
def test_distances_reject_empty_samples(distance, empty_reference):
    reference, batch = pd.Series([1.0]), pd.Series([], dtype="float64")
    if empty_reference:
        reference, batch = batch, reference
    with pytest.raises(ValueError, match="непустые выборки"):
        distance(reference, batch)


def test_identical_distributions_return_zero_metrics_and_no_drift(working_frame):
    result = evaluate_drift(working_frame, working_frame.copy())
    assert isinstance(result, DriftResult)
    assert list(result.metrics.columns) == ["Признак", "Мера", "Значение", "Порог", "Дрейф"]
    assert result.metrics["Признак"].tolist() == [
        "price", "product_category_name", "customer_state",
    ]
    assert result.metrics["Мера"].tolist() == ["KS D", "TVD", "TVD"]
    assert result.metrics["Значение"].tolist() == [0.0, 0.0, 0.0]
    assert result.metrics["Порог"].tolist() == [0.1, 0.1, 0.1]
    assert result.metrics["Дрейф"].dtype == bool
    assert not result.metrics["Дрейф"].any()
    assert result.drift_detected is False


@pytest.mark.parametrize(("feature", "shifted"), [
    ("price", 1000.0), ("product_category_name", "new_category"), ("customer_state", "ZZ"),
])
def test_any_single_feature_can_trigger_overall_drift(working_frame, feature, shifted):
    batch = working_frame.copy()
    batch.loc[:, feature] = shifted
    result = evaluate_drift(working_frame, batch)
    metrics = result.metrics.set_index("Признак")
    assert metrics.at[feature, "Значение"] == 1.0
    assert metrics["Дрейф"].to_dict() == {
        name: name == feature for name in metrics.index
    }
    assert (metrics.drop(index=feature)["Значение"] == 0.0).all()
    assert result.drift_detected is True


@pytest.mark.parametrize(("feature", "threshold_name", "old", "new"), [
    ("price", "price", 10.0, 20.0),
    ("product_category_name", "category", "known", "new"),
    ("customer_state", "state", "SP", "RJ"),
])
@pytest.mark.parametrize(("threshold", "expected"), [
    (np.nextafter(0.1, 0.0), True), (0.1, True), (np.nextafter(0.1, 1.0), False),
])
def test_threshold_comparison_is_greater_than_or_equal(
    working_frame, feature, threshold_name, old, new, threshold, expected,
):
    reference = working_frame.iloc[:10].copy()
    reference.loc[:, feature] = old
    batch = reference.copy()
    batch.loc[9, feature] = new
    result = evaluate_drift(
        reference, batch, thresholds=DriftThresholds(**{threshold_name: threshold}),
    )
    metric = result.metrics.set_index("Признак").loc[feature]
    assert metric["Значение"] == 0.1
    assert metric["Порог"] == threshold
    assert bool(metric["Дрейф"]) is expected
    assert result.drift_detected is expected


def test_custom_thresholds_are_applied_to_corresponding_features(working_frame):
    batch = working_frame.copy()
    batch.loc[:2, "price"] = 1000.0
    batch.loc[:2, "product_category_name"] = "new"
    batch.loc[:2, "customer_state"] = "ZZ"
    result = evaluate_drift(
        working_frame, batch, thresholds=DriftThresholds(price=0.25, category=0.5, state=0.75),
    )
    assert result.metrics["Значение"].tolist() == [0.25, 0.25, 0.25]
    assert result.metrics["Порог"].tolist() == [0.25, 0.5, 0.75]
    assert result.metrics["Дрейф"].tolist() == [True, False, False]
    assert result.drift_detected is True


@pytest.mark.parametrize("threshold", [0, 1])
def test_endpoint_thresholds_are_inclusive(working_frame, threshold):
    batch = working_frame.copy()
    if threshold == 1:
        batch.loc[:, "price"] = 1000.0
        batch.loc[:, "product_category_name"] = "new"
        batch.loc[:, "customer_state"] = "ZZ"
    result = evaluate_drift(
        working_frame, batch, thresholds=DriftThresholds(threshold, threshold, threshold),
    )
    assert result.metrics["Значение"].tolist() == [float(threshold)] * 3
    assert result.metrics["Дрейф"].tolist() == [True, True, True]
    assert result.drift_detected is True


def test_new_and_disappearing_values_in_tail_beyond_top_ten():
    top_ten = [f"common_{index:02d}" for index in range(10) for _ in range(4)]
    reference_values = top_ten + ["old_tail_a"] * 2 + ["old_tail_b"] * 2
    batch_values = top_ten + ["new_tail"] * 4
    reference = pd.DataFrame({
        "price": 10.0, "product_category_name": reference_values, "customer_state": reference_values,
    })
    batch = pd.DataFrame({
        "price": 10.0, "product_category_name": batch_values, "customer_state": batch_values,
    })
    result = evaluate_drift(reference, batch, thresholds=DriftThresholds(category=0.05, state=0.05))
    assert result.metrics["Значение"].tolist() == pytest.approx([0.0, 4 / 44, 4 / 44])
    assert result.metrics["Дрейф"].tolist() == [False, True, True]
    assert result.drift_detected is True


def test_results_are_repeatable_order_invariant_and_do_not_mutate_inputs(working_frame):
    reference = working_frame.copy()
    batch = working_frame.iloc[:5].copy()
    batch.loc[0, "price"] = 1000.0
    batch.loc[0, "product_category_name"] = "new"
    batch.loc[0, "customer_state"] = "ZZ"
    reference_before, batch_before = reference.copy(deep=True), batch.copy(deep=True)
    expected = evaluate_drift(reference, batch)
    for result in (
        evaluate_drift(reference, batch),
        evaluate_drift(reference.iloc[::-1], batch.iloc[::-1]),
        evaluate_drift(pd.concat([reference] * 3), pd.concat([batch] * 2)),
    ):
        pd.testing.assert_frame_equal(result.metrics, expected.metrics)
        assert result.drift_detected is expected.drift_detected
    pd.testing.assert_frame_equal(reference, reference_before)
    pd.testing.assert_frame_equal(batch, batch_before)
