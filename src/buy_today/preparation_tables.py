"""Assemble validated Olist positions and record each filtering stage."""

from __future__ import annotations

import pandas as pd

from buy_today.artifacts import PreparationStatistics
from buy_today.schema import COLUMNS, DATE_COLUMNS, DATE_FORMAT, DTYPES, ROW_KEY, SORT_KEY

_GEO_KEY = "geolocation_zip_code_prefix"
_MINIMUM_ROWS = 2


def _require_unique(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{name}: missing key values in {columns}")
    if frame.duplicated(list(columns)).any():
        raise ValueError(f"{name}: duplicate key values in {columns}")


def _require_references(frame: pd.DataFrame, column: str, parent: pd.DataFrame) -> None:
    if not frame[column].isin(parent[column]).all():
        raise ValueError(f"Missing references for {column}")


def row_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": len(frame),
        "orders": int(frame["order_id"].nunique()),
        "products": int(frame["product_id"].nunique()),
    }


def _normalize_zip(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = frame.copy()
    strings = frame[column].astype("string").str
    frame[column] = strings.zfill(5)
    present = frame[column].dropna().str
    if not present.fullmatch(r"[0-9]{5}").all():
        raise ValueError(f"{column}: expected five-digit postal prefixes")
    return frame


def _aggregate_geography(geolocation: pd.DataFrame) -> pd.DataFrame:
    source = _normalize_zip(geolocation, _GEO_KEY).dropna(subset=[_GEO_KEY])
    # Choose city/state together; ties are resolved alphabetically.
    return _merge_geography(_median_coordinates(source), _representative_places(source))


def _median_coordinates(source: pd.DataFrame) -> pd.DataFrame:
    return source.groupby(_GEO_KEY, as_index=False).agg(
        geolocation_lat=("geolocation_lat", "median"),
        geolocation_lng=("geolocation_lng", "median"),
    )


def _representative_places(source: pd.DataFrame) -> pd.DataFrame:
    return (
        source.groupby([_GEO_KEY, "geolocation_city", "geolocation_state"], dropna=False)
        .size()
        .rename("point_count")
        .reset_index()
        .sort_values(
            [_GEO_KEY, "point_count", "geolocation_city", "geolocation_state"],
            ascending=[True, False, True, True],
        )
        .drop_duplicates(_GEO_KEY)
        .drop(columns="point_count")
    )


def _merge_geography(coordinates: pd.DataFrame, places: pd.DataFrame) -> pd.DataFrame:
    return coordinates.merge(places, on=_GEO_KEY, how="left", validate="one_to_one")


def _validate_tables(tables: dict[str, pd.DataFrame]) -> None:
    _require_unique(tables["order_items"], ROW_KEY, "order_items")
    for name, key in (
        ("orders", "order_id"),
        ("customers", "customer_id"),
        ("sellers", "seller_id"),
        ("products", "product_id"),
        ("category_translation", "product_category_name"),
    ):
        _require_unique(tables[name], (key,), name)
    _require_references(tables["order_items"], "order_id", tables["orders"])


def _address_features(addresses: pd.DataFrame, geography: pd.DataFrame, role: str) -> pd.DataFrame:
    zip_column = f"{role}_zip_code_prefix"
    renamed = geography.rename(
        columns={
            column: zip_column if column == _GEO_KEY else f"{role}_{column}"
            for column in geography.columns
        }
    )
    return pd.merge(
        _normalize_zip(addresses, zip_column),
        renamed,
        on=zip_column,
        how="left",
        validate="many_to_one",
    )


def _translated_products(products: pd.DataFrame, translation: pd.DataFrame) -> pd.DataFrame:
    products = products.copy()
    category_map = translation.set_index("product_category_name")["product_category_name_english"]
    products["product_category_name"] = (
        products["product_category_name"]
        .map(category_map)
        .fillna(products["product_category_name"])
    )
    return products


def _join_features(selected: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    for name, key in (
        ("customers", "customer_id"),
        ("sellers", "seller_id"),
        ("products", "product_id"),
    ):
        _require_references(selected, key, tables[name])
    geography = _aggregate_geography(tables["geolocation"])
    for role in ("customer", "seller"):
        addresses = _address_features(tables[f"{role}s"], geography, role)
        selected = pd.merge(
            selected, addresses, on=f"{role}_id", how="left", validate="many_to_one"
        )
    products = _translated_products(tables["products"], tables["category_translation"])
    return pd.merge(selected, products, on="product_id", how="left", validate="many_to_one").loc[
        :, list(COLUMNS)
    ]


def assemble(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, PreparationStatistics]:
    _validate_tables(tables)
    items, orders = tables["order_items"], tables["orders"]
    delivered_orders = orders.loc[orders["order_status"].eq("delivered")].copy()
    for column in DATE_COLUMNS:
        delivered_orders[column] = pd.to_datetime(
            delivered_orders[column], format=DATE_FORMAT, errors="raise"
        )
    # Count before filtering orders: a cancelled sale still counts.
    seller_counts = items.groupby("product_id")["seller_id"].nunique()
    delivered_items = items.merge(
        delivered_orders, on="order_id", how="inner", validate="many_to_one"
    )
    selected = _single_seller_items(delivered_items, seller_counts)
    assembled = _join_features(selected, tables)
    return assembled, {
        "stages": {
            "source_items": row_counts(items),
            "delivered_items": row_counts(delivered_items),
            "single_seller_items": row_counts(selected),
            "assembled": row_counts(assembled),
        },
        "excluded_non_delivered_orders": len(orders) - len(delivered_orders),
        "multi_seller_products": int((seller_counts > 1).sum()),
        "missing_before_cleaning": {
            str(column): int(count) for column, count in pd.isna(assembled).sum().items()
        },
    }


def _single_seller_items(items: pd.DataFrame, seller_counts: pd.Series[int]) -> pd.DataFrame:
    products = seller_counts.index[seller_counts.eq(1)]
    return items.loc[items["product_id"].isin(products)].copy()


def clean(assembled: pd.DataFrame, statistics: PreparationStatistics, minimum: int) -> pd.DataFrame:
    cleaned = assembled.dropna(subset=list(COLUMNS)).astype(DTYPES)
    if len(cleaned) < _MINIMUM_ROWS:
        raise ValueError("At least two complete rows are required after cleaning")
    statistics["stages"]["cleaned"] = row_counts(cleaned)
    statistics["rows_dropped_missing"] = len(assembled) - len(cleaned)
    category_counts = cleaned["product_category_name"].value_counts()
    retained = category_counts.index[category_counts >= minimum]
    working = (
        cleaned.loc[cleaned["product_category_name"].isin(retained)]
        .sort_values(list(SORT_KEY))
        .reset_index(drop=True)
    )
    if len(working) < _MINIMUM_ROWS:
        raise ValueError(
            "At least two rows are required after category filtering; "
            + f"min_category_count={minimum}, retained rows={len(working)}"
        )
    statistics["stages"]["category_filtered"] = row_counts(working)
    statistics["rows_dropped_rare_categories"] = len(cleaned) - len(working)
    statistics["categories_before_filtering"] = len(category_counts)
    statistics["categories_after_filtering"] = len(retained)
    return working
