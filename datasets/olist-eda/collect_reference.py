"""Систематизация локальных источников для аудита уже подготовленного Olist.

Запуск из корня: .venv/bin/python datasets/olist-eda/collect_reference.py
Исходные CSV и подготовленная таблица не изменяются.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
KEY = ["order_id", "order_item_id"]


def main():
    items = pd.read_csv(DATA / "olist_order_items_dataset.csv")
    orders = pd.read_csv(DATA / "olist_orders_dataset.csv")
    customers = pd.read_csv(DATA / "olist_customers_dataset.csv")
    products = pd.read_csv(DATA / "olist_products_dataset.csv")
    translations = pd.read_csv(DATA / "product_category_name_translation.csv")
    prepared = pd.read_csv(DATA / "olist_prepared_dataset.csv")
    assert not items.duplicated(KEY).any()
    assert not prepared.duplicated(KEY).any()
    assert orders.order_id.is_unique and customers.customer_id.is_unique
    assert products.product_id.is_unique and translations.product_category_name.is_unique
    assert items.order_id.isin(orders.order_id).all()
    assert orders.customer_id.isin(customers.customer_id).all()
    assert items.product_id.isin(products.product_id).all()

    counts = items.groupby("product_id").seller_id.nunique()
    category_map = translations.set_index("product_category_name").product_category_name_english
    products["category"] = products.product_category_name.map(category_map).fillna(
        products.product_category_name
    )
    reference = (
        items[KEY + ["product_id", "price", "freight_value"]]
        .merge(orders[["order_id", "customer_id", "order_status", "order_purchase_timestamp"]],
               on="order_id", validate="many_to_one")
        .merge(customers[["customer_id", "customer_unique_id"]],
               on="customer_id", validate="many_to_one")
        .merge(products[["product_id", "category"]], on="product_id", validate="many_to_one")
        .drop(columns="customer_id")
    )
    reference["source_seller_count"] = reference.product_id.map(counts)
    reference["kept"] = reference.order_status.eq("delivered") & reference.source_seller_count.eq(1)
    selected = reference.loc[reference.kept]
    assert len(reference) == len(items)
    pd.testing.assert_frame_equal(
        selected[KEY].sort_values(KEY).reset_index(drop=True),
        prepared[KEY].sort_values(KEY).reset_index(drop=True),
    )
    comparison = prepared.merge(selected, on=KEY, validate="one_to_one", suffixes=("", "_source"))
    for column in ["product_id", "customer_unique_id", "price", "freight_value", "order_purchase_timestamp"]:
        assert comparison[column].equals(comparison[column + "_source"]), column
    assert comparison.product_category_name.fillna("<NA>").equals(comparison.category.fillna("<NA>"))
    reference.to_csv(OUT / "source_reference.csv.gz", index=False,
                     compression={"method": "gzip", "mtime": 0})

    manifest = {"sources": [], "outputs": [], "environment": {}}
    # Идентичности всех локальных источников и логики исходной подготовки.
    for path in [*sorted(DATA.glob("*.csv")), ROOT / "prepare_data.ipynb"]:
        record = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                  "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}
        manifest["sources"].append(record)
    output = OUT / "source_reference.csv.gz"
    manifest["outputs"].append({"path": str(output.relative_to(ROOT)), "rows": len(reference),
                                "columns": reference.columns.tolist(),
                                "sha256": hashlib.file_digest(output.open("rb"), "sha256").hexdigest()})
    for package in ["pandas", "numpy", "matplotlib", "nbformat", "nbclient", "ipykernel"]:
        manifest["environment"][package] = importlib.metadata.version(package)
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    delivered = reference.loc[reference.order_status.eq("delivered")]
    before = delivered.groupby("order_id").size()
    after = selected.groupby("order_id").size().reindex(before.index, fill_value=0)
    checks = {
        "prepared_rows": len(prepared), "prepared_columns": prepared.shape[1],
        "full_duplicates": int(prepared.duplicated().sum()),
        "key_duplicates": int(prepared.duplicated(KEY).sum()),
        "keys_exactly_match_source_selection": True,
        "source_items": len(reference), "delivered_items": len(delivered),
        "fully_removed_delivered_orders": int(after.eq(0).sum()),
        "partial_retained_orders": int((after.gt(0) & after.lt(before)).sum()),
        "complete_retained_orders": int(after.eq(before).sum()),
        "source_orders_without_items": int((~orders.order_id.isin(items.order_id)).sum()),
        "rows_with_any_missing": int(prepared.isna().any(axis=1).sum()),
    }
    (OUT / "collection_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
