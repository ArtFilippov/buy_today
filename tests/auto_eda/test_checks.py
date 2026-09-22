import numpy as np
import pandas as pd
import pytest

from buy_today.auto_eda.checks import check_combination, check_dataset
from buy_today.schema import read_dataset


def test_complete_dataset_passes_with_repeated_products_and_tied_times(working_frame):
    # Delivery before approval is not a violation of purchase-time sorting.
    working_frame["order_approved_at"] += pd.Timedelta(days=10)
    checks = check_dataset(working_frame)
    assert len(checks) == 7
    assert set(checks["Результат"]) == {"OK"}
    assert checks["Фактически"].notna().all()
    assert checks["Ожидается"].notna().all()


@pytest.mark.parametrize("column", ["product_category_name", "order_approved_at", "price"])
def test_missing_values_fail(working_frame, write_dataset, column):
    working_frame.loc[2, column] = None
    with pytest.raises(ValueError, match="Пропуски: фактически 1"):
        check_dataset(read_dataset(write_dataset(working_frame)))


@pytest.mark.parametrize("price", [0, -1, np.inf, -np.inf])
def test_invalid_prices_fail(working_frame, write_dataset, price):
    working_frame.loc[3, "price"] = price
    with pytest.raises(ValueError, match="Конечная положительная цена: фактически 1"):
        check_dataset(read_dataset(write_dataset(working_frame)))


def test_duplicate_key_reports_excess_count(working_frame):
    working_frame.loc[1, "order_item_id"] = 1
    with pytest.raises(ValueError, match=r"Повторы ключа.*фактически 1.*= 0"):
        check_dataset(working_frame)


def test_unsorted_purchases_fail(working_frame):
    with pytest.raises(ValueError, match="переходов назад во времени = 0"):
        check_dataset(working_frame.iloc[::-1])


def test_empty_csv_fails(working_frame, write_dataset):
    with pytest.raises(ValueError, match="Непустой датасет: фактически 0"):
        check_dataset(read_dataset(write_dataset(working_frame.iloc[:0])))


@pytest.mark.parametrize("change", ["reorder", "missing", "extra"])
def test_strict_csv_columns(working_frame, write_dataset, change):
    if change == "reorder":
        working_frame = working_frame[working_frame.columns[::-1]]
    elif change == "missing":
        working_frame = working_frame.drop(columns="customer_city")
    else:
        working_frame["extra"] = 1
    with pytest.raises(ValueError, match="35 working dataset columns"):
        read_dataset(write_dataset(working_frame))


@pytest.mark.parametrize(("column", "value"), [
    ("order_item_id", "not-an-integer"), ("order_item_id", "1.5"),
    ("price", "not-a-price"), ("order_approved_at", "2018-02-30 00:00:00"),
])
def test_strict_csv_values(working_frame, write_dataset, column, value):
    working_frame[column] = working_frame[column].astype("string")
    working_frame.loc[0, column] = value
    with pytest.raises(ValueError):
        read_dataset(write_dataset(working_frame))


def test_dataframe_types_are_checked(working_frame):
    working_frame["order_item_id"] = working_frame["order_item_id"].astype("float64")
    with pytest.raises(ValueError, match="Типы и даты.*order_item_id.*float64"):
        check_dataset(working_frame)


@pytest.mark.parametrize("new_orders", [False, True])
def test_combination_accepts_repeated_products_with_distinct_row_keys(working_frame, new_orders):
    reference = working_frame.copy()
    batch = working_frame.copy()
    if new_orders:
        batch["order_id"] = "new_" + batch["order_id"]
    else:
        batch["order_item_id"] += 2
    reference_before, batch_before = reference.copy(deep=True), batch.copy(deep=True)
    dataset_checks = check_dataset(reference)
    check_dataset(batch)
    checks = check_combination(reference, batch)
    assert list(checks.columns) == list(dataset_checks.columns)
    assert len(checks) == 1
    assert "объединении" in checks.at[0, "Проверка"]
    assert checks.at[0, "Результат"] == "OK"
    assert checks.at[0, "Фактически"] == "0"
    assert checks.at[0, "Ожидается"] == "число повторений сверх первого = 0"
    pd.testing.assert_frame_equal(reference, reference_before)
    pd.testing.assert_frame_equal(batch, batch_before)


@pytest.mark.parametrize(("start", "stop"), [(0, 12), (4, 9)])
@pytest.mark.parametrize("changed_values", [False, True])
def test_combination_rejects_full_and_partial_key_overlap(
    working_frame, start, stop, changed_values,
):
    batch = working_frame.copy()
    batch["order_id"] = "new_" + batch["order_id"]
    batch.loc[start:stop - 1, "order_id"] = working_frame.loc[start:stop - 1, "order_id"]
    if changed_values:
        batch["price"] += 1.0
        batch.loc[:, "product_id"] = "different_product"
    check_dataset(working_frame)
    check_dataset(batch)
    with pytest.raises(
        ValueError,
        match=rf"Повторы ключа.*объединении: фактически {stop - start};.*сверх первого = 0",
    ):
        check_combination(working_frame, batch)


@pytest.mark.parametrize(("reference_rows", "batch_rows", "duplicates"), [
    ([0, 0, 0], [2, 3], 2),
    ([0, 1], [2, 2, 2, 2], 3),
    ([0, 0], [2, 2, 2], 3),
    ([0, 0, 0], [0, 0, 2, 2], 5),
])
def test_combination_counts_internal_and_cross_input_repetitions_beyond_first(
    working_frame, reference_rows, batch_rows, duplicates,
):
    # Check directly: internal duplicates must not be mistaken for a disjoint union.
    with pytest.raises(
        ValueError,
        match=rf"Повторы ключа.*объединении: фактически {duplicates};.*сверх первого = 0",
    ):
        check_combination(working_frame.iloc[reference_rows], working_frame.iloc[batch_rows])
