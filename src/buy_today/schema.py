"""The working Olist dataset: ordered columns, types and CSV reading."""

from pathlib import Path

import pandas as pd


DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_COLUMNS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)
ROW_KEY = ("order_id", "order_item_id")
SORT_KEY = ("order_purchase_timestamp", *ROW_KEY)

# Keep Olist's original spelling of the two *_lenght columns.
DTYPES = {
    "order_id": "string",
    "order_item_id": "int64",
    "customer_id": "string",
    "customer_unique_id": "string",
    "product_id": "string",
    "price": "float64",
    "freight_value": "float64",
    "order_status": "string",
    **{column: "datetime64[ns]" for column in DATE_COLUMNS},
    "customer_zip_code_prefix": "string",
    "customer_city": "string",
    "customer_state": "string",
    "customer_geolocation_lat": "float64",
    "customer_geolocation_lng": "float64",
    "customer_geolocation_city": "string",
    "customer_geolocation_state": "string",
    "seller_zip_code_prefix": "string",
    "seller_city": "string",
    "seller_state": "string",
    "seller_geolocation_lat": "float64",
    "seller_geolocation_lng": "float64",
    "seller_geolocation_city": "string",
    "seller_geolocation_state": "string",
    "product_category_name": "string",
    "product_name_lenght": "float64",
    "product_description_lenght": "float64",
    "product_photos_qty": "float64",
    "product_weight_g": "float64",
    "product_length_cm": "float64",
    "product_height_cm": "float64",
    "product_width_cm": "float64",
}
COLUMNS = tuple(DTYPES)
CSV_DTYPES = {column: dtype for column, dtype in DTYPES.items() if column not in DATE_COLUMNS}


def read_dataset(path: Path | str) -> pd.DataFrame:
    """Read a prepared CSV, restoring its exact schema, dates and numeric values.

    Raises ValueError for unexpected columns or unparseable values. This reads
    the schema; checks such as completeness and data drift belong to the caller.

    Args:
        path (Path | str): Prepared CSV to read.

    Returns:
        pd.DataFrame: Positions with schema column order and restored dtypes.

    Raises:
        ValueError: Columns or values do not match the working schema.
    """
    frame = pd.read_csv(path, dtype=CSV_DTYPES, encoding="utf-8", float_precision="round_trip")
    if tuple(frame.columns) != COLUMNS:
        raise ValueError(
            "Expected the 35 working dataset columns in schema order; "
            + f"got {len(frame.columns)} columns: {list(frame.columns)}"
        )
    for column in DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], format=DATE_FORMAT, errors="raise")
    return pd.DataFrame.astype(frame, DTYPES)
