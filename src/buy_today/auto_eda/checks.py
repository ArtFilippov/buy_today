"""Fail-fast checks shared by reports and the reference lifecycle."""

import numpy as np
import pandas as pd

from buy_today.schema import COLUMNS, DTYPES, ROW_KEY


def check_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the check table, raising ValueError on the first violated condition.

    Call read_dataset first for CSV inputs. Equal purchase timestamps are valid;
    dates of approval, shipping and delivery need not form an ordered chain.

    Args:
        frame (pd.DataFrame): Positions with the working dataset schema.

    Returns:
        pd.DataFrame: Successful checks in execution order.
    """
    records: list[dict[str, str]] = []

    def check(name: str, passed: bool, actual: str | int, expected: str) -> None:
        if not passed:
            raise ValueError(f"{name}: фактически {actual}; ожидается {expected}")
        records.append(
            {
                "Проверка": name,
                "Результат": "OK",
                "Фактически": str(actual),
                "Ожидается": expected,
            }
        )

    check(
        "Схема и порядок колонок",
        tuple(frame.columns) == COLUMNS,
        "35 колонок, порядок совпадает"
        if tuple(frame.columns) == COLUMNS
        else str(list(frame.columns)),
        "35 колонок в порядке schema.COLUMNS",
    )
    mismatched = {
        column: str(frame[column].dtype)
        for column, dtype in DTYPES.items()
        if frame[column].dtype != pd.api.types.pandas_dtype(dtype)
    }
    check(
        "Типы и даты",
        not mismatched,
        str(mismatched) if mismatched else "35 типов совпадают",
        "типы schema.DTYPES, включая пять разобранных дат",
    )
    check("Непустой датасет", len(frame) > 0, len(frame), "число строк > 0")
    missing = int(frame.isna().sum().sum())
    check("Пропуски", not missing, missing, "число пропущенных значений = 0")
    duplicates = int(frame.duplicated(list(ROW_KEY)).sum())
    check(
        "Повторы ключа (order_id, order_item_id)",
        not duplicates,
        duplicates,
        "число повторений сверх первого = 0",
    )
    times = frame["order_purchase_timestamp"]
    inversions = int((times.diff() < pd.Timedelta(0)).sum())
    check(
        "Хронологический порядок покупок",
        not inversions,
        inversions,
        "число переходов назад во времени = 0",
    )
    invalid_prices = int((~np.isfinite(frame["price"]) | (frame["price"] <= 0)).sum())
    check(
        "Конечная положительная цена",
        not invalid_prices,
        invalid_prices,
        "число цен, нарушающих 0 < price < inf, = 0",
    )
    return pd.DataFrame(records)


def check_combination(reference: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    """Check combined row keys after check_dataset has validated both inputs.

    Count all repetitions beyond the first, including those within either input.
    Repeated products and distinct items of the same order remain valid.

    Args:
        reference (pd.DataFrame): Validated accumulated positions.
        batch (pd.DataFrame): Validated incoming positions.

    Returns:
        pd.DataFrame: Successful combined-key check.

    Raises:
        ValueError: A row key occurs more than once across the inputs.
    """
    keys = pd.concat([reference[list(ROW_KEY)], batch[list(ROW_KEY)]], ignore_index=True)
    duplicates = int(pd.DataFrame.duplicated(keys).sum())
    name = "Повторы ключа (order_id, order_item_id) в объединении"
    expected = "число повторений сверх первого = 0"
    if duplicates:
        raise ValueError(f"{name}: фактически {duplicates}; ожидается {expected}")
    return pd.DataFrame(
        [
            {
                "Проверка": name,
                "Результат": "OK",
                "Фактически": str(duplicates),
                "Ожидается": expected,
            }
        ]
    )
