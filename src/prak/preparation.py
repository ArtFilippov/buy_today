"""Build the working Olist dataset and its chronological batch stream."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from prak.schema import COLUMNS, CSV_DTYPES, DATE_COLUMNS, DATE_FORMAT, DTYPES
from prak.schema import ROW_KEY, SORT_KEY


_SOURCES = {
    "order_items": (
        "olist_order_items_dataset.csv",
        (*ROW_KEY, "product_id", "seller_id", "price", "freight_value"),
    ),
    "orders": (
        "olist_orders_dataset.csv",
        ("order_id", "customer_id", "order_status", *DATE_COLUMNS),
    ),
    "customers": (
        "olist_customers_dataset.csv",
        (
            "customer_id", "customer_unique_id", "customer_zip_code_prefix",
            "customer_city", "customer_state",
        ),
    ),
    "sellers": (
        "olist_sellers_dataset.csv",
        ("seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"),
    ),
    "geolocation": (
        "olist_geolocation_dataset.csv",
        (
            "geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
            "geolocation_city", "geolocation_state",
        ),
    ),
    "products": (
        "olist_products_dataset.csv",
        (
            "product_id", "product_category_name", "product_name_lenght",
            "product_description_lenght", "product_photos_qty", "product_weight_g",
            "product_length_cm", "product_height_cm", "product_width_cm",
        ),
    ),
    "category_translation": (
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


@dataclass(frozen=True)
class PreparationResult:
    working_dataset_path: Path
    manifest_path: Path
    rows_before_cleaning: int
    rows_after_cleaning: int
    batch_sizes: tuple[int, ...]
    dropped_tail_rows: int


def _require_unique(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{name}: missing key values in {columns}")
    if frame.duplicated(list(columns)).any():
        raise ValueError(f"{name}: duplicate key values in {columns}")


def _require_references(
    frame: pd.DataFrame, column: str, parent: pd.DataFrame
) -> None:
    if not frame[column].isin(parent[column]).all():
        raise ValueError(f"Missing references for {column}")


def _row_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": len(frame),
        "orders": int(frame["order_id"].nunique()),
        "products": int(frame["product_id"].nunique()),
    }


def _normalize_zip(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = frame.copy()
    frame[column] = frame[column].astype("string").str.zfill(5)
    if not frame[column].dropna().str.fullmatch(r"[0-9]{5}").all():
        raise ValueError(f"{column}: expected five-digit postal prefixes")
    return frame


def _aggregate_geography(geolocation: pd.DataFrame) -> pd.DataFrame:
    key = "geolocation_zip_code_prefix"
    source = _normalize_zip(geolocation, key).dropna(subset=[key])
    coordinates = source.groupby(key, as_index=False).agg(
        geolocation_lat=("geolocation_lat", "median"),
        geolocation_lng=("geolocation_lng", "median"),
    )
    # Choose city and state together. Ties are resolved alphabetically.
    places = (
        source.groupby(
            [key, "geolocation_city", "geolocation_state"], dropna=False
        )
        .size().rename("point_count").reset_index()
        .sort_values(
            [key, "point_count", "geolocation_city", "geolocation_state"],
            ascending=[True, False, True, True],
        )
        .drop_duplicates(key)
        .drop(columns="point_count")
    )
    return coordinates.merge(places, on=key, how="left", validate="one_to_one")


def _assemble(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    items, orders = tables["order_items"], tables["orders"]
    _require_unique(items, ROW_KEY, "order_items")
    for name, key in (
        ("orders", "order_id"), ("customers", "customer_id"),
        ("sellers", "seller_id"), ("products", "product_id"),
        ("category_translation", "product_category_name"),
    ):
        _require_unique(tables[name], (key,), name)
    _require_references(items, "order_id", orders)

    delivered_orders = orders.loc[orders["order_status"].eq("delivered")].copy()
    for column in DATE_COLUMNS:
        delivered_orders[column] = pd.to_datetime(
            delivered_orders[column], format=DATE_FORMAT, errors="raise"
        )
    # Count sellers before filtering orders: a cancelled sale still counts.
    seller_counts = items.groupby("product_id")["seller_id"].nunique()
    single_seller_products = seller_counts.index[seller_counts.eq(1)]
    delivered_items = items.merge(
        delivered_orders, on="order_id", how="inner", validate="many_to_one"
    )
    selected = delivered_items.loc[
        delivered_items["product_id"].isin(single_seller_products)
    ].copy()
    for name, key in (
        ("customers", "customer_id"), ("sellers", "seller_id"),
        ("products", "product_id"),
    ):
        _require_references(selected, key, tables[name])

    geo_by_zip = _aggregate_geography(tables["geolocation"])
    address_features = {}
    for role in ("customer", "seller"):
        zip_column = f"{role}_zip_code_prefix"
        addresses = _normalize_zip(tables[f"{role}s"], zip_column)
        geography = geo_by_zip.rename(columns={
            column: zip_column if column == "geolocation_zip_code_prefix"
            else f"{role}_{column}"
            for column in geo_by_zip.columns
        })
        address_features[role] = addresses.merge(
            geography, on=zip_column, how="left", validate="many_to_one"
        )

    products = tables["products"].copy()
    category_map = tables["category_translation"].set_index(
        "product_category_name"
    )["product_category_name_english"]
    products["product_category_name"] = (
        products["product_category_name"].map(category_map)
        .fillna(products["product_category_name"])
    )
    assembled = (
        selected
        .merge(address_features["customer"], on="customer_id", how="left",
               validate="many_to_one")
        .merge(address_features["seller"], on="seller_id", how="left",
               validate="many_to_one")
        .merge(products, on="product_id", how="left", validate="many_to_one")
        .loc[:, list(COLUMNS)]
    )
    statistics = {
        "stages": {
            "source_items": _row_counts(items),
            "delivered_items": _row_counts(delivered_items),
            "single_seller_items": _row_counts(selected),
            "assembled": _row_counts(assembled),
        },
        "excluded_non_delivered_orders": len(orders) - len(delivered_orders),
        "multi_seller_products": int(seller_counts.gt(1).sum()),
        "missing_before_cleaning": {
            column: int(count) for column, count in assembled.isna().sum().items()
        },
    }
    return assembled, statistics


def _file_metadata(frame: pd.DataFrame, path: str) -> dict:
    timestamps = frame["order_purchase_timestamp"]
    return {
        "path": path,
        "rows": len(frame),
        "period_start": timestamps.iloc[0].isoformat(),
        "period_end": timestamps.iloc[-1].isoformat(),
    }


def prepare_data(
    raw_dir: Path | str, output_dir: Path | str, batch_size: int = 5000
) -> PreparationResult:
    """Prepare seven raw CSVs and overwrite the dataset, batches and manifest.

    The first batch contains N // 2 rows; later batches contain batch_size rows.
    An incomplete tail stays in working_dataset.csv but is omitted from the
    stream. The manifest lists the current batches; state.json resets to index 0.
    At least two complete rows and a positive integer batch_size are required.
    Invalid keys, references, source formats or I/O failures raise immediately.
    """
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()
    tables = {
        name: pd.read_csv(
            raw_dir / filename,
            usecols=list(columns),
            dtype={column: _SOURCE_DTYPES[column] for column in columns},
            encoding="utf-8",
            float_precision="round_trip",
        )
        for name, (filename, columns) in _SOURCES.items()
    }
    assembled, statistics = _assemble(tables)
    working = (
        assembled.dropna(subset=list(COLUMNS))
        .sort_values(list(SORT_KEY))
        .reset_index(drop=True)
        .astype(DTYPES)
    )
    if len(working) < 2:
        raise ValueError("At least two complete rows are required after cleaning")
    statistics["stages"]["cleaned"] = _row_counts(working)
    statistics["rows_dropped_missing"] = len(assembled) - len(working)

    (output_dir / "batches").mkdir(parents=True, exist_ok=True)
    working_path = output_dir / "working_dataset.csv"
    working.to_csv(working_path, index=False, encoding="utf-8", date_format=DATE_FORMAT)

    first_size = len(working) // 2
    spans = [(0, first_size)] + [
        (start, start + batch_size)
        for start in range(first_size, len(working) - batch_size + 1, batch_size)
    ]
    batches = []
    for index, (start, stop) in enumerate(spans):
        batch = working.iloc[start:stop]
        batch_id = f"batch_{index:03d}"
        relative_path = f"batches/{batch_id}.csv"
        batch.to_csv(
            output_dir / relative_path, index=False,
            encoding="utf-8", date_format=DATE_FORMAT,
        )
        batches.append({"id": batch_id, **_file_metadata(batch, relative_path)})
    dropped_tail_rows = len(working) - spans[-1][1]
    statistics["dropped_tail_rows"] = dropped_tail_rows

    manifest = {
        "format_version": 1,
        "sources": {
            "raw_dir": str(raw_dir),
            "files": [
                {"path": filename, "rows": len(tables[name])}
                for name, (filename, _) in _SOURCES.items()
            ],
        },
        "schema": {"columns": list(COLUMNS), "dtypes": DTYPES},
        "parameters": {
            "batch_size": batch_size,
            "first_batch_rule": "N // 2",
            "drop_last": True,
            "sort_key": list(SORT_KEY),
            "drop_missing": "any_of_selected_columns",
        },
        "statistics": statistics,
        "working_dataset": _file_metadata(working, "working_dataset.csv"),
        "batches": batches,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "state.json").write_text(
        json.dumps({"next_batch_index": 0}, indent=2) + "\n", encoding="utf-8"
    )
    return PreparationResult(
        working_dataset_path=working_path,
        manifest_path=manifest_path,
        rows_before_cleaning=len(assembled),
        rows_after_cleaning=len(working),
        batch_sizes=tuple(batch["rows"] for batch in batches),
        dropped_tail_rows=dropped_tail_rows,
    )
