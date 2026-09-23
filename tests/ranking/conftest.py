from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def ranking_snapshot(tmp_path: Path) -> Path:
    directory = tmp_path / "history snapshot"
    directory.mkdir()
    pd.DataFrame({"product_id": [f"p{i:02d}" for i in range(12)]}).to_csv(
        directory / "catalog.csv",
        index=False,
    )
    for split, start, products in (
        ("train", 0, ["p00", "p00", "p01"]),
        ("validation", 3, ["p02", "p02"]),
        ("test", 5, ["p03", "p01"]),
    ):
        rows = [
            {
                "event_id": f"{user}:{start + index}",
                "user_id": user,
                "event_index": start + index,
                "split": split,
                "product_id": product,
                "batch_index": 0,
                "order_id": "source_order",
                "order_item_id": 1,
                "order_purchase_timestamp": "2018-01-01 00:00:00",
            }
            for user in ("001", "NA")
            for index, product in enumerate(products)
        ]
        pd.DataFrame(rows).to_csv(directory / f"{split}.csv", index=False)
    return directory
