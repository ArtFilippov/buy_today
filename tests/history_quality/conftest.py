"""Tiny published snapshot with controlled events across all three splits."""

import hashlib
import json
from pathlib import Path

import fixture_types as ft
import joblib
import numpy as np
import pandas as pd
import pytest

from buy_today.clustering.distances import TimestampDistance
from buy_today.generation import generate_dataset, read_history_dataset
from buy_today.schema import DATE_FORMAT, ROW_KEY


@pytest.fixture
def quality_snapshot(
    working_frame: pd.DataFrame, write_dataset: ft.DatasetWriter, tmp_path: Path,
) -> tuple[Path, Path]:
    reference = working_frame.iloc[:2].copy()
    reference["product_category_name"] = pd.array(["A", "B"], dtype="string")
    reference_path = write_dataset(reference, "reference ' rows.csv")
    distance = tmp_path / "distance.joblib"
    joblib.dump(TimestampDistance(), distance)
    paths = generate_dataset(
        reference_path, distance, tmp_path / "history snapshot", temperature=1,
        n_users=2, split_sizes=(4, 1, 1),
    )
    data = read_history_dataset(paths.output_dir)
    positions = [0] * 5 + [1] + [1] * 5 + [0]
    for column in (*ROW_KEY, "product_id", "order_purchase_timestamp"):
        data.events[column] = reference.iloc[positions][column].reset_index(drop=True)
    for column in ROW_KEY:
        data.anchors[column] = reference[column].reset_index(drop=True)
    data.anchors["source_position"] = np.arange(2)
    for split in ("train", "validation", "test"):
        data.events[data.events["split"].eq(split)].to_csv(
            paths.output_dir / f"{split}.csv", index=False, date_format=DATE_FORMAT,
        )
    data.anchors.to_csv(paths.output_dir / "generator/anchors.csv", index=False)
    for name in data.manifest["files"]:
        checksum = hashlib.sha256((paths.output_dir / name).read_bytes()).hexdigest()
        data.manifest["files"][name] = checksum
    paths.manifest_path.write_text(json.dumps(data.manifest), encoding="utf-8")
    return paths.output_dir, reference_path
