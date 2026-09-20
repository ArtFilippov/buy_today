from pathlib import Path

import pandas as pd
import pytest

from prak.schema import DATE_COLUMNS, DATE_FORMAT, DTYPES


@pytest.fixture
def working_frame() -> pd.DataFrame:
    """Complete positions with tied purchase times and distinct item keys."""
    values = {
        column: [1 if dtype in ("float64", "int64") else "value"] * 12
        for column, dtype in DTYPES.items()
    }
    for column in DATE_COLUMNS:
        values[column] = pd.date_range("2018-01-01", periods=12, freq="h")
    values.update({
        "order_id": [f"order_{index // 2:03d}" for index in range(12)],
        "order_item_id": [1, 2] * 6,
        "order_purchase_timestamp": pd.date_range("2018-01-01", periods=6, freq="h").repeat(2),
        "customer_zip_code_prefix": ["00123"] * 12,
        "seller_zip_code_prefix": ["00456"] * 12,
        "price": [100.12345678912345] * 12,
        "product_category_name": [f"category_{index:02d}" for index in reversed(range(12))],
        "customer_state": ["SP"] * 8 + ["RJ"] * 3 + ["MG"],
    })
    return pd.DataFrame(values).astype(DTYPES)


@pytest.fixture
def new_batch(working_frame) -> pd.DataFrame:
    """A later, disjoint batch with the same distributions and repeated products."""
    frame = working_frame.copy()
    frame["order_id"] = "new_" + frame["order_id"]
    for column in DATE_COLUMNS:
        frame[column] += pd.Timedelta(days=1)
    return frame


@pytest.fixture
def write_dataset(tmp_path):
    def write(frame: pd.DataFrame, name: str = "input ' batch.csv") -> Path:
        path = tmp_path / name
        frame.to_csv(path, index=False, encoding="utf-8", date_format=DATE_FORMAT)
        return path

    return write


@pytest.fixture
def raw_tables() -> dict[str, pd.DataFrame]:
    """Small, complete Olist sources with tied times and padded postal codes."""
    order_ids = [f"order_{index:03d}" for index in range(12)]
    purchase_times = [
        (pd.Timestamp("2018-01-01") + pd.Timedelta(hours=index // 2)).isoformat(sep=" ")
        for index in range(12)
    ]
    return {
        "olist_order_items_dataset.csv": pd.DataFrame({
            "order_id": order_ids,
            "order_item_id": 1,
            "product_id": "product_a",
            "seller_id": "seller_a",
            "price": 100.12345678912345,
            "freight_value": 5.99,
            # Missing values in excluded columns must not remove any rows.
            "shipping_limit_date": None,
        }),
        "olist_orders_dataset.csv": pd.DataFrame({
            "order_id": order_ids,
            "customer_id": "customer_a",
            "order_status": "delivered",
            "order_purchase_timestamp": purchase_times,
            "order_approved_at": "2018-01-02 00:00:00",
            "order_delivered_carrier_date": "2018-01-03 00:00:00",
            "order_delivered_customer_date": "2018-01-04 00:00:00",
            "order_estimated_delivery_date": "2018-01-05 00:00:00",
        }),
        "olist_customers_dataset.csv": pd.DataFrame([{
            "customer_id": "customer_a",
            "customer_unique_id": "unique_customer_a",
            "customer_zip_code_prefix": "00123",
            "customer_city": "customer city",
            "customer_state": "SP",
        }]),
        "olist_sellers_dataset.csv": pd.DataFrame([{
            "seller_id": "seller_a",
            "seller_zip_code_prefix": "00456",
            "seller_city": "seller city",
            "seller_state": "RJ",
        }]),
        "olist_geolocation_dataset.csv": pd.DataFrame({
            "geolocation_zip_code_prefix": ["00123"] * 4 + ["00456"],
            "geolocation_lat": [1.0, 9.0, 3.0, 5.0, -22.0],
            "geolocation_lng": [2.0, 8.0, 10.0, 4.0, -45.0],
            "geolocation_city": ["zeta", "alpha", "zeta", "alpha", "seller geo"],
            "geolocation_state": ["AA", "ZZ", "AA", "ZZ", "RJ"],
        }),
        "olist_products_dataset.csv": pd.DataFrame([{
            "product_id": "product_a",
            "product_category_name": "brinquedos",
            "product_name_lenght": 30.0,
            "product_description_lenght": 100.0,
            "product_photos_qty": 2.0,
            "product_weight_g": 500.0,
            "product_length_cm": 20.0,
            "product_height_cm": 5.0,
            "product_width_cm": 10.0,
        }]),
        "product_category_name_translation.csv": pd.DataFrame([{
            "product_category_name": "brinquedos",
            "product_category_name_english": "toys",
        }]),
    }


@pytest.fixture
def raw_category_tables(raw_tables):
    """Build complete sources with specified order-item counts per category."""
    def build(category_counts: dict[str, int]) -> dict[str, pd.DataFrame]:
        tables = {name: frame.copy() for name, frame in raw_tables.items()}
        product_ids = [
            f"product_{index}" for index, count in enumerate(category_counts.values())
            for _ in range(count)
        ]
        order_ids = [f"order_{index:05d}" for index in range(len(product_ids))]
        for filename in ("olist_order_items_dataset.csv", "olist_orders_dataset.csv"):
            tables[filename] = (
                tables[filename].iloc[[0] * len(order_ids)].reset_index(drop=True)
                .assign(order_id=order_ids)
            )
        tables["olist_order_items_dataset.csv"]["product_id"] = product_ids
        tables["olist_orders_dataset.csv"]["order_purchase_timestamp"] = pd.date_range(
            "2018-01-01", periods=len(order_ids), freq="h",
        ).strftime(DATE_FORMAT)
        tables["olist_products_dataset.csv"] = (
            tables["olist_products_dataset.csv"].iloc[[0] * len(category_counts)]
            .reset_index(drop=True)
            .assign(
                product_id=[f"product_{index}" for index in range(len(category_counts))],
                product_category_name=list(category_counts),
            )
        )
        return tables

    return build


@pytest.fixture
def write_raw(tmp_path):
    def write(tables: dict[str, pd.DataFrame]) -> Path:
        directory = tmp_path / "raw source"
        directory.mkdir(exist_ok=True)
        for filename, frame in tables.items():
            frame.to_csv(directory / filename, index=False)
        return directory

    return write
