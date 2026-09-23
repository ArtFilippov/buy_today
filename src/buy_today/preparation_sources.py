"""Read the seven required Olist source tables with stable CSV types."""

from pathlib import Path
from typing import NamedTuple

import pandas as pd

from buy_today.schema import CSV_DTYPES, DATE_COLUMNS, ROW_KEY


class _Source(NamedTuple):
    filename: str
    columns: tuple[str, ...]


SOURCES = {
    "order_items": _Source(
        "olist_order_items_dataset.csv",
        (*ROW_KEY, "product_id", "seller_id", "price", "freight_value"),
    ),
    "orders": _Source(
        "olist_orders_dataset.csv", ("order_id", "customer_id", "order_status", *DATE_COLUMNS)
    ),
    "customers": _Source(
        "olist_customers_dataset.csv",
        (
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
        ),
    ),
    "sellers": _Source(
        "olist_sellers_dataset.csv",
        (
            "seller_id",
            "seller_zip_code_prefix",
            "seller_city",
            "seller_state",
        ),
    ),
    "geolocation": _Source(
        "olist_geolocation_dataset.csv",
        (
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
            "geolocation_city",
            "geolocation_state",
        ),
    ),
    "products": _Source(
        "olist_products_dataset.csv",
        (
            "product_id",
            "product_category_name",
            "product_name_lenght",
            "product_description_lenght",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ),
    ),
    "category_translation": _Source(
        "product_category_name_translation.csv",
        ("product_category_name", "product_category_name_english"),
    ),
}
_SOURCE_DTYPES = CSV_DTYPES | {
    "order_item_id": "Int64",
    "seller_id": "string",
    "geolocation_zip_code_prefix": "string",
    "geolocation_lat": "float64",
    "geolocation_lng": "float64",
    "geolocation_city": "string",
    "geolocation_state": "string",
    "product_category_name_english": "string",
    **{column: "string" for column in DATE_COLUMNS},
}


def read_sources(raw_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(
            raw_dir / source.filename,
            usecols=list(source.columns),
            dtype={column: _SOURCE_DTYPES[column] for column in source.columns},
            encoding="utf-8",
            float_precision="round_trip",
        )
        for name, source in SOURCES.items()
    }


def source_metadata(raw_dir: Path, tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    return {
        "raw_dir": str(raw_dir),
        "files": [
            {"path": source.filename, "rows": len(tables[name])} for name, source in SOURCES.items()
        ],
    }
