import numpy as np
import pandas as pd
import pytest

from buy_today.history_quality import evaluate_history_quality


def reference_rows(categories: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": ["order"] * len(categories),
        "order_item_id": np.arange(1, len(categories) + 1),
        "product_category_name": categories,
    })


def histories(items: list[list[int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.DataFrame([
        {"user_id": f"user_{user}", "order_id": "order", "order_item_id": item}
        for user, history in enumerate(items) for item in history
    ])
    anchors = events.groupby("user_id", sort=True).first().reset_index()
    return events, anchors


def test_concentration_distinguishes_99_1_from_50_50_at_identical_global_frequencies() -> None:
    reference = reference_rows(["A", "B"])
    concentrated = evaluate_history_quality(
        *histories([[1] * 99 + [2], [2] * 99 + [1]]), reference,
    )
    balanced = evaluate_history_quality(*histories([[1, 2] * 50, [2, 1] * 50]), reference)
    for result, top, effective in ((concentrated, 0.99, 1 / 0.9802), (balanced, 0.5, 2.0)):
        metrics = result["metrics"]
        assert not metrics["category_frequency_tvd"]
        assert metrics["user_top_category_share_mean"] == pytest.approx(top)
        assert metrics["user_effective_category_count_mean"] == pytest.approx(effective)
        assert metrics["anchor_category_event_share"] == pytest.approx(top)
        assert metrics["user_anchor_category_share_mean"] == pytest.approx(top)
    assert concentrated["metrics"]["users_top_category_share_ge_0_99"] == 1
    assert not balanced["metrics"]["users_top_category_share_ge_0_80"]
    assert concentrated["iid"] == balanced["iid"]
    assert concentrated["metrics"]["top_category_share_lift_vs_iid"] > 1.5
    assert concentrated["metrics"]["effective_category_count_ratio_vs_iid"] < 0.6


def test_linear_quantiles_and_inclusive_threshold_shares() -> None:
    result = evaluate_history_quality(
        *histories([[1] * count + [2] * (100 - count) for count in (50, 80, 95, 99)]),
        reference_rows(["A", "B"]),
    )
    metrics = result["metrics"]
    assert metrics["user_top_category_share_mean"] == pytest.approx(0.81)
    for percentile, expected in ((50, 0.875), (90, 0.978), (95, 0.984), (99, 0.9888)):
        assert metrics[f"user_top_category_share_p{percentile}"] == pytest.approx(expected)
    for threshold, expected in ((80, 0.75), (90, 0.5), (95, 0.5), (99, 0.25)):
        assert metrics[f"users_top_category_share_ge_0_{threshold}"] == expected
    assert metrics["user_anchor_category_share_p95"] == pytest.approx(0.984)
    assert metrics["users_anchor_category_share_ge_0_95"] == 0.5
    assert metrics["users_anchor_category_share_ge_0_99"] == 0.25
    assert metrics["user_effective_category_count_p10"] == pytest.approx(0.7 / .9802 + .3 / .905)
    assert metrics["user_effective_category_count_p50"] == pytest.approx((1 / .905 + 1 / .68) / 2)


def test_event_weights_differ_from_user_weights_and_anchor_is_not_top_category() -> None:
    events, anchors = histories([[1] * 9 + [2], [2, 3]])
    anchors["order_item_id"] = [2, 1]
    result = evaluate_history_quality(events, anchors, reference_rows(["A", "B", "C", "D"]))
    metrics = result["metrics"]
    assert metrics["user_top_category_share_mean"] == pytest.approx(0.7)
    assert metrics["user_effective_category_count_mean"] == pytest.approx((1 / .82 + 2) / 2)
    assert metrics["anchor_category_event_share"] == pytest.approx(1 / 12)
    assert metrics["user_anchor_category_share_mean"] == pytest.approx(0.05)
    assert metrics["category_frequency_tvd"] == pytest.approx(0.5)
    assert result["n_events"] == 12
    assert result["n_users"] == 2
    assert result["n_reference_rows"] == 4


def test_iid_matches_known_expectation_for_unequal_lengths_and_ratios_of_means() -> None:
    # With two uniform categories, N=1 gives top=effective=1. For N=2,
    # P(same)=1/2, so E[top]=3/4 and E[effective]=3/2.
    result = evaluate_history_quality(
        *histories([[1], [1, 2]]), reference_rows(["A", "B"]), iid_repeats=20000,
    )
    baseline = result["iid"]
    assert baseline["user_top_category_share_mean"] == pytest.approx(.875, abs=.003)
    assert baseline["user_effective_category_count_mean"] == pytest.approx(1.25, abs=.006)
    assert result["metrics"]["top_category_share_lift_vs_iid"] == (
        .75 / baseline["user_top_category_share_mean"]
    )
    assert result["metrics"]["effective_category_count_ratio_vs_iid"] == (
        1.5 / baseline["user_effective_category_count_mean"]
    )


def test_repeatable_independent_of_row_order_without_mutating_inputs_or_global_rng() -> None:
    events, anchors = histories([[1, 1, 1, 2], [3, 2], [1]])
    reference = reference_rows(["A", "B", "C"])
    originals = [table.copy(deep=True) for table in (events, anchors, reference)]
    initial_rng_state = np.random.get_state()
    np.random.seed(71)
    expected_random = np.random.random()
    np.random.seed(71)
    result = evaluate_history_quality(events, anchors, reference, random_state=17)
    assert np.random.random() == expected_random
    assert result == evaluate_history_quality(events, anchors, reference, random_state=17)
    assert result == evaluate_history_quality(
        events.iloc[::-1], anchors.iloc[::-1], reference.iloc[::-1], random_state=17,
    )
    for actual, original in zip((events, anchors, reference), originals):
        pd.testing.assert_frame_equal(actual, original)
    changed_seed = evaluate_history_quality(events, anchors, reference, random_state=18)
    assert (
        result["iid"]["user_top_category_share_mean"]
        != changed_seed["iid"]["user_top_category_share_mean"]
    )
    np.random.set_state(initial_rng_state)


@pytest.mark.parametrize("items,categories", [([[1, 1, 1]], ["A"]), ([[1], [2]], ["A", "B"])])
def test_degenerate_valid_data_has_finite_metrics_and_unit_iid_ratios(
    items: list[list[int]], categories: list[str],
) -> None:
    result = evaluate_history_quality(*histories(items), reference_rows(categories))
    assert all(np.isfinite(value) for value in result["metrics"].values())
    for name in ("top_category_share_lift_vs_iid", "effective_category_count_ratio_vs_iid"):
        assert result["metrics"][name] == 1


@pytest.mark.parametrize("table", ["events", "anchors", "reference"])
@pytest.mark.parametrize("problem", ["empty", "missing", "blank", "unknown_key"])
def test_invalid_tables_cannot_silently_drop_observations(table: str, problem: str) -> None:
    events, anchors = histories([[1, 2]])
    tables = {"events": events, "anchors": anchors, "reference": reference_rows(["A", "B"])}
    frame = tables[table].copy()
    if problem == "empty":
        frame = frame.iloc[:0]
    elif problem == "missing":
        frame.loc[0, "order_id"] = None
    elif problem == "blank":
        frame.loc[0, "order_id"] = " "
    else:
        frame.loc[0, "order_item_id"] = 99
    tables[table] = frame
    with pytest.raises(ValueError):
        evaluate_history_quality(tables["events"], tables["anchors"], tables["reference"])


@pytest.mark.parametrize("table", ["anchors", "reference"])
def test_duplicate_identity_is_an_error(table: str) -> None:
    events, anchors = histories([[1, 2]])
    reference = reference_rows(["A", "B"])
    if table == "anchors":
        anchors = pd.concat([anchors, anchors])
    else:
        reference = pd.concat([reference, reference])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_history_quality(events, anchors, reference)


@pytest.mark.parametrize("category", [None, "", "   "])
def test_missing_category_is_an_error_even_for_unsampled_rows(category: str | None) -> None:
    reference = reference_rows(["A", "B"])
    reference.loc[1, "product_category_name"] = category
    with pytest.raises(ValueError):
        evaluate_history_quality(*histories([[1]]), reference)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_iid_repeats(value: int) -> None:
    with pytest.raises(ValueError, match="iid_repeats"):
        evaluate_history_quality(*histories([[1]]), reference_rows(["A"]), iid_repeats=value)


@pytest.mark.parametrize("value", [-1, 2**32, 1.5, True])
def test_invalid_iid_seed(value: int) -> None:
    with pytest.raises(ValueError, match="random_state"):
        evaluate_history_quality(*histories([[1]]), reference_rows(["A"]), random_state=value)
