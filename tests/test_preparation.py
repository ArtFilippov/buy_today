import json
from typing import Any

import fixture_types as ft
import pandas as pd
import pytest

from buy_today.preparation import prepare_data
from buy_today.schema import COLUMNS, DATE_COLUMNS, DTYPES, read_dataset


def test_complete_dataset_and_geography(
    raw_tables: ft.RawTables, write_raw: ft.RawWriter, tmp_path: ft.Path
) -> None:
    result = prepare_data(
        write_raw(raw_tables),
        tmp_path / "stream",
        batch_size=3,
        min_category_count=1,
    )
    working = read_dataset(result.working_dataset_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert working.shape == (12, 35)
    assert tuple(working.columns) == COLUMNS
    assert working.dtypes.astype(str).to_dict() == DTYPES
    assert not working.isna().any().any()
    assert not working.duplicated(["order_id", "order_item_id"]).any()
    assert working["customer_zip_code_prefix"].eq("00123").all()
    assert working["seller_zip_code_prefix"].eq("00456").all()
    assert working["price"].eq(100.12345678912345).all()
    assert working["product_category_name"].eq("toys").all()
    assert working["customer_geolocation_lat"].eq(4.0).all()
    assert working["customer_geolocation_lng"].eq(6.0).all()
    # Alphabetical tie-break must choose the pair alpha/ZZ, not alpha/AA.
    assert working["customer_geolocation_city"].eq("alpha").all()
    assert working["customer_geolocation_state"].eq("ZZ").all()
    assert working["seller_geolocation_lat"].eq(-22.0).all()
    assert working["seller_geolocation_lng"].eq(-45.0).all()
    assert all(pd.api.types.is_datetime64_dtype(working[c]) for c in DATE_COLUMNS)
    assert result.rows_before_cleaning == result.rows_after_cleaning == 12
    assert result.rows_after_category_filtering == 12
    assert result.categories_before_filtering == result.categories_after_filtering == 1
    assert not manifest["statistics"]["rows_dropped_missing"]
    assert not manifest["statistics"]["rows_dropped_rare_categories"]
    assert set(manifest["statistics"]["missing_before_cleaning"].values()) == {0}
    assert len(manifest["sources"]["files"]) == 7
    assert manifest["working_dataset"]["rows"] == 12
    assert manifest["schema"] == {"columns": list(COLUMNS), "dtypes": DTYPES}


def test_sellers_counted_before_delivered_filter(
    raw_tables: ft.RawTables, write_raw: ft.RawWriter, tmp_path: ft.Path
) -> None:
    items = raw_tables["olist_order_items_dataset.csv"]
    items.loc[[0, 1], "product_id"] = "product_b"
    items.loc[1, "seller_id"] = "seller_b"
    raw_tables["olist_orders_dataset.csv"].loc[1, "order_status"] = "canceled"
    # A surviving position of the same order must not be lost with product_b.
    sibling = items.iloc[[0]].assign(order_item_id=2, product_id="product_a")
    raw_tables["olist_order_items_dataset.csv"] = pd.concat([items, sibling])
    products = raw_tables["olist_products_dataset.csv"]
    raw_tables["olist_products_dataset.csv"] = pd.concat(
        [products, products.assign(product_id="product_b")]
    )
    sellers = raw_tables["olist_sellers_dataset.csv"]
    raw_tables["olist_sellers_dataset.csv"] = pd.concat(
        [sellers, sellers.assign(seller_id="seller_b")]
    )

    result = prepare_data(write_raw(raw_tables), tmp_path / "stream", min_category_count=1)
    working = read_dataset(result.working_dataset_path)
    stats = json.loads(result.manifest_path.read_text(encoding="utf-8"))["statistics"]

    assert len(working) == 11
    assert working["product_id"].eq("product_a").all()
    assert working.loc[working["order_id"].eq("order_000"), "order_item_id"].tolist() == [2]
    assert "order_001" not in set(working["order_id"])
    assert stats["stages"]["source_items"]["rows"] == 13
    assert stats["stages"]["delivered_items"]["rows"] == 12
    assert stats["stages"]["single_seller_items"]["rows"] == 11
    assert stats["multi_seller_products"] == 1
    assert stats["excluded_non_delivered_orders"] == 1


def test_cleaning_and_untranslated_categories(
    raw_tables: ft.RawTables, write_raw: ft.RawWriter, tmp_path: ft.Path
) -> None:
    orders = raw_tables["olist_orders_dataset.csv"]
    orders.loc[0, ["order_delivered_carrier_date", "order_delivered_customer_date"]] = None
    orders.loc[1, "customer_id"] = "customer_b"
    customers = raw_tables["olist_customers_dataset.csv"]
    raw_tables["olist_customers_dataset.csv"] = pd.concat(
        [
            customers,
            customers.assign(customer_id="customer_b", customer_zip_code_prefix="99999"),
        ]
    )
    products = raw_tables["olist_products_dataset.csv"]
    raw_tables["olist_products_dataset.csv"] = pd.concat(
        [
            products,
            products.assign(product_id="product_b", product_category_name=None),
            products.assign(product_id="product_c", product_category_name="pc_gamer"),
        ]
    )
    items = raw_tables["olist_order_items_dataset.csv"]
    items.loc[2, "product_id"] = "product_b"
    items.loc[3, "product_id"] = "product_c"

    result = prepare_data(write_raw(raw_tables), tmp_path / "stream", min_category_count=1)
    working = read_dataset(result.working_dataset_path)
    stats = json.loads(result.manifest_path.read_text(encoding="utf-8"))["statistics"]

    assert result.rows_before_cleaning == 12
    assert result.rows_after_cleaning == 9
    assert stats["rows_dropped_missing"] == 3
    assert stats["missing_before_cleaning"]["order_delivered_carrier_date"] == 1
    assert stats["missing_before_cleaning"]["order_delivered_customer_date"] == 1
    assert stats["missing_before_cleaning"]["customer_geolocation_lat"] == 1
    assert stats["missing_before_cleaning"]["product_category_name"] == 1
    assert not working.isna().any().any()
    assert working["order_id"].tolist() == [f"order_{i:03d}" for i in range(3, 12)]
    assert working.loc[0, "product_category_name"] == "pc_gamer"


@pytest.mark.parametrize("threshold", [3, 1000])
def test_category_cutoff_and_filtered_batches(
    raw_category_tables: ft.CategoryTables,
    write_raw: ft.RawWriter,
    tmp_path: ft.Path,
    threshold: int,
) -> None:
    tables = raw_category_tables(
        {
            "rare": threshold - 1,
            "boundary": threshold,
            "common": threshold + 1,
        }
    )
    # Exercise the default 1000 as well as an explicit small threshold.
    options = {} if threshold == 1000 else {"min_category_count": threshold}
    output = tmp_path / "stream"
    result = prepare_data(write_raw(tables), output, batch_size=threshold, **options)
    working = read_dataset(result.working_dataset_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    stats = manifest["statistics"]

    assert working["product_category_name"].value_counts().to_dict() == {
        "boundary": threshold,
        "common": threshold + 1,
    }
    assert working["order_id"].tolist() == [
        f"order_{index:05d}" for index in range(threshold - 1, 3 * threshold)
    ]
    assert result.rows_before_cleaning == result.rows_after_cleaning == 3 * threshold
    assert result.rows_after_category_filtering == 2 * threshold + 1
    assert result.categories_before_filtering == stats["categories_before_filtering"] == 3
    assert result.categories_after_filtering == stats["categories_after_filtering"] == 2
    assert not stats["rows_dropped_missing"]
    assert stats["rows_dropped_rare_categories"] == threshold - 1
    assert stats["stages"]["cleaned"]["rows"] == 3 * threshold
    assert stats["stages"]["category_filtered"] == {
        "rows": 2 * threshold + 1,
        "orders": 2 * threshold + 1,
        "products": 2,
    }
    assert manifest["parameters"]["min_category_count"] == threshold
    assert manifest["working_dataset"]["rows"] == 2 * threshold + 1
    assert result.batch_sizes == (threshold, threshold)
    assert result.dropped_tail_rows == 1
    batches = [read_dataset(output / entry["path"]) for entry in manifest["batches"]]
    pd.testing.assert_frame_equal(
        pd.concat(batches, ignore_index=True),
        working.iloc[:-1],
        check_exact=True,
    )

    shuffled = {name: frame.sample(frac=1, random_state=42) for name, frame in tables.items()}
    second_output = tmp_path / "second"
    prepare_data(write_raw(shuffled), second_output, batch_size=threshold, **options)
    for path in ["working_dataset.csv", "manifest.json", "state.json"] + [
        entry["path"] for entry in manifest["batches"]
    ]:
        assert (output / path).read_bytes() == (second_output / path).read_bytes()


def test_category_counts_use_complete_rows(
    raw_category_tables: ft.CategoryTables, write_raw: ft.RawWriter, tmp_path: ft.Path
) -> None:
    tables = raw_category_tables({"becomes_rare": 4, "common": 8})
    tables["olist_orders_dataset.csv"].loc[:1, "order_approved_at"] = None
    result = prepare_data(write_raw(tables), tmp_path / "stream", min_category_count=3)
    working = read_dataset(result.working_dataset_path)
    stats = json.loads(result.manifest_path.read_text(encoding="utf-8"))["statistics"]

    assert result.rows_before_cleaning == 12
    assert result.rows_after_cleaning == 10
    assert result.rows_after_category_filtering == 8
    assert working["product_category_name"].eq("common").all()
    assert stats["rows_dropped_missing"] == stats["rows_dropped_rare_categories"] == 2


def test_category_counts_use_translated_positions(
    raw_category_tables: ft.CategoryTables, write_raw: ft.RawWriter, tmp_path: ft.Path
) -> None:
    tables = raw_category_tables({"brinquedos": 2, "toys": 2, "rare": 1})
    result = prepare_data(write_raw(tables), tmp_path / "stream", min_category_count=3)
    working = read_dataset(result.working_dataset_path)

    assert working["product_category_name"].tolist() == ["toys"] * 4
    assert working["product_id"].nunique() == 2
    assert result.categories_before_filtering == 2
    assert result.categories_after_filtering == 1


def test_all_categories_removed_before_writing(
    raw_tables: ft.RawTables, write_raw: ft.RawWriter, tmp_path: ft.Path
) -> None:
    output = tmp_path / "stream"
    with pytest.raises(ValueError, match="min_category_count=1000, retained rows=0"):
        prepare_data(write_raw(raw_tables), output)
    assert not output.exists()


@pytest.mark.parametrize("threshold", [0, -1, 1.5, True, "1000", None])
def test_invalid_category_threshold_is_rejected_before_reading(
    tmp_path: ft.Path, threshold: Any
) -> None:
    with pytest.raises(ValueError, match="min_category_count must be a positive integer"):
        prepare_data(tmp_path / "absent", tmp_path / "stream", min_category_count=threshold)


@pytest.mark.parametrize(
    ("rows", "batch_size", "sizes", "tail"),
    [
        (2, 5000, [1], 1),
        (2, 1, [1, 1], 0),
        (3, 1, [1, 1, 1], 0),
        (3, 5, [1], 2),
        (6, 3, [3, 3], 0),
        (7, 3, [3, 3], 1),
        (8, 2, [4, 2, 2], 0),
        (9, 2, [4, 2, 2], 1),
        (10, 2, [5, 2, 2], 1),
        (11, 2, [5, 2, 2, 2], 0),
        (12, 3, [6, 3, 3], 0),
    ],
)
def test_chronological_batches(
    raw_tables: ft.RawTables,
    write_raw: ft.RawWriter,
    tmp_path: ft.Path,
    rows: int,
    batch_size: int,
    sizes: list[int],
    tail: int,
) -> None:
    raw_tables["olist_order_items_dataset.csv"] = (
        raw_tables["olist_order_items_dataset.csv"].iloc[:rows].sample(frac=1, random_state=7)
    )
    output = tmp_path / "stream"
    result = prepare_data(write_raw(raw_tables), output, batch_size, min_category_count=1)
    working = read_dataset(result.working_dataset_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    batches = [read_dataset(output / entry["path"]) for entry in manifest["batches"]]

    assert working["order_id"].tolist() == [f"order_{i:03d}" for i in range(rows)]
    assert [len(batch) for batch in batches] == sizes
    assert result.batch_sizes == tuple(sizes)
    assert result.dropped_tail_rows == manifest["statistics"]["dropped_tail_rows"] == tail
    assert [entry["rows"] for entry in manifest["batches"]] == sizes
    assert [entry["id"] for entry in manifest["batches"]] == [
        f"batch_{i:03d}" for i in range(len(sizes))
    ]
    for batch, entry in zip(batches, manifest["batches"], strict=True):
        assert entry["period_start"] == batch["order_purchase_timestamp"].iloc[0].isoformat()
        assert entry["period_end"] == batch["order_purchase_timestamp"].iloc[-1].isoformat()
    pd.testing.assert_frame_equal(
        pd.concat(batches, ignore_index=True), working.iloc[: sum(sizes)], check_exact=True
    )
    assert len(working) == sum(sizes) + tail
    assert json.loads((output / "state.json").read_text()) == {"next_batch_index": 0}


def test_positions_of_one_order_can_cross_batch_boundary(
    raw_tables: ft.RawTables, write_raw: ft.RawWriter, tmp_path: ft.Path
) -> None:
    items = raw_tables["olist_order_items_dataset.csv"]
    items.loc[6, ["order_id", "order_item_id"]] = ["order_005", 2]
    raw_tables["olist_order_items_dataset.csv"] = items.iloc[::-1]
    output = tmp_path / "stream"
    prepare_data(write_raw(raw_tables), output, batch_size=3, min_category_count=1)
    first = read_dataset(output / "batches/batch_000.csv")
    second = read_dataset(output / "batches/batch_001.csv")
    assert first.iloc[-1][["order_id", "order_item_id"]].tolist() == ["order_005", 1]
    assert second.iloc[0][["order_id", "order_item_id"]].tolist() == ["order_005", 2]


def test_reproducibility_with_shuffled_sources(
    raw_tables: ft.RawTables, write_raw: ft.RawWriter, tmp_path: ft.Path
) -> None:
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = prepare_data(write_raw(raw_tables), first_dir, batch_size=2, min_category_count=1)
    shuffled = {name: frame.sample(frac=1, random_state=42) for name, frame in raw_tables.items()}
    prepare_data(write_raw(shuffled), second_dir, batch_size=2, min_category_count=1)
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    paths = ["working_dataset.csv", "manifest.json", "state.json"]
    paths.extend(entry["path"] for entry in manifest["batches"])
    for path in paths:
        assert (first_dir / path).read_bytes() == (second_dir / path).read_bytes()


def test_repreparation_resets_state_and_replaces_manifest(
    raw_tables: ft.RawTables, write_raw: ft.RawWriter, tmp_path: ft.Path
) -> None:
    raw_dir, output = write_raw(raw_tables), tmp_path / "stream"
    first = prepare_data(raw_dir, output, batch_size=1, min_category_count=1)
    assert len(first.batch_sizes) == 7
    (output / "state.json").write_text('{"next_batch_index": 4}')
    (output / "batches/unrelated.csv").write_text("not a batch")
    result = prepare_data(raw_dir, output, batch_size=5, min_category_count=1)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.batch_sizes == (6, 5)
    assert manifest["parameters"]["batch_size"] == 5
    assert [batch["path"] for batch in manifest["batches"]] == [
        "batches/batch_000.csv",
        "batches/batch_001.csv",
    ]
    assert len(read_dataset(output / "batches/batch_001.csv")) == 5
    assert json.loads((output / "state.json").read_text()) == {"next_batch_index": 0}


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_is_rejected_before_reading(tmp_path: ft.Path, batch_size: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        prepare_data(tmp_path / "absent", tmp_path / "stream", batch_size)


@pytest.mark.parametrize("rows", [0, 1])
def test_too_few_rows(
    raw_tables: ft.RawTables, write_raw: ft.RawWriter, tmp_path: ft.Path, rows: int
) -> None:
    raw_tables["olist_order_items_dataset.csv"] = raw_tables["olist_order_items_dataset.csv"].iloc[
        :rows
    ]
    with pytest.raises(ValueError, match="At least two complete rows"):
        prepare_data(write_raw(raw_tables), tmp_path / "stream")


def test_all_rows_removed_by_cleaning(
    raw_tables: ft.RawTables, write_raw: ft.RawWriter, tmp_path: ft.Path
) -> None:
    raw_tables["olist_products_dataset.csv"]["product_category_name"] = None
    with pytest.raises(ValueError, match="At least two complete rows"):
        prepare_data(write_raw(raw_tables), tmp_path / "stream")


@pytest.mark.parametrize(
    "filename",
    [
        "olist_order_items_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_customers_dataset.csv",
        "olist_sellers_dataset.csv",
        "olist_products_dataset.csv",
        "product_category_name_translation.csv",
    ],
)
def test_duplicate_keys_are_errors(
    raw_tables: ft.RawTables, write_raw: ft.RawWriter, tmp_path: ft.Path, filename: str
) -> None:
    raw_tables[filename] = pd.concat([raw_tables[filename], raw_tables[filename].iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate key"):
        prepare_data(write_raw(raw_tables), tmp_path / "stream")


@pytest.mark.parametrize(
    ("filename", "column"),
    [
        ("olist_order_items_dataset.csv", "order_id"),
        ("olist_order_items_dataset.csv", "seller_id"),
        ("olist_order_items_dataset.csv", "product_id"),
        ("olist_orders_dataset.csv", "customer_id"),
    ],
)
def test_missing_references_are_errors(
    raw_tables: ft.RawTables, write_raw: ft.RawWriter, tmp_path: ft.Path, filename: str, column: str
) -> None:
    if column == "seller_id":
        # Keep one seller per product so these rows reach reference validation.
        raw_tables[filename][column] = "unknown"
    else:
        raw_tables[filename].loc[0, column] = "unknown"
    with pytest.raises(ValueError, match=column):
        prepare_data(write_raw(raw_tables), tmp_path / "stream")


@pytest.mark.parametrize(
    ("filename", "column", "value"),
    [
        ("olist_orders_dataset.csv", "order_purchase_timestamp", "not a date"),
        ("olist_customers_dataset.csv", "customer_zip_code_prefix", "bad zip"),
    ],
)
def test_malformed_source_values_fail(
    raw_tables: ft.RawTables,
    write_raw: ft.RawWriter,
    tmp_path: ft.Path,
    filename: str,
    column: str,
    value: str,
) -> None:
    raw_tables[filename].loc[0, column] = value
    with pytest.raises(ValueError):
        prepare_data(write_raw(raw_tables), tmp_path / "stream")
