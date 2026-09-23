"""CSV, checksum and atomic-publication operations for history snapshots."""

import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import pandas as pd

from buy_today.generation.histories import (
    ANCHOR_DTYPES, EVENT_DTYPES, SOURCE_DTYPES, check_parameters,
)
from buy_today.generation.records import HistoryDataset, HistoryManifest, SnapshotInputs, SPLITS
from buy_today.schema import DATE_FORMAT

DATETIME_DTYPE = "datetime64[ns]"
STRING_DTYPE = "string"


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def distance_file(index: int) -> str:
    return f"generator/distances/batch_{index:03d}.joblib"


def _data_files(n_batches: int) -> list[str]:
    return [
        *(f"{split}.csv" for split in SPLITS),
        "catalog.csv", "generator/anchors.csv", "generator/sources.csv",
        *(distance_file(index) for index in range(n_batches)),
    ]


def _read_table(path: Path, dtypes: dict[str, str]) -> pd.DataFrame:
    dates = [name for name, dtype in dtypes.items() if dtype == DATETIME_DTYPE]
    frame: pd.DataFrame = pd.read_csv(
        path, dtype={name: dtype for name, dtype in dtypes.items() if name not in dates},
        encoding="utf-8", keep_default_na=False,
    )
    if tuple(frame.columns) != tuple(dtypes):
        raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
    for name in dates:
        frame[name] = pd.to_datetime(frame[name], format=DATE_FORMAT, errors="raise")
    return _check_table(frame.astype(dtypes), path, dtypes)


def _check_table(frame: pd.DataFrame, path: Path, dtypes: dict[str, str]) -> pd.DataFrame:
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"Empty or missing values in {path}")
    for name, dtype in dtypes.items():
        if dtype == STRING_DTYPE and frame[name].eq("").any():
            raise ValueError(f"Empty identifiers in {path}: {name}")
    return frame


def _read_manifest(directory: Path) -> HistoryManifest:
    manifest: HistoryManifest = json.loads(
        (directory / "generator/manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("format_version") != 1 or not manifest.get("batches"):
        raise ValueError("Expected history snapshot format_version=1 with batches")
    for index, record in enumerate(manifest["batches"]):
        if record["batch_index"] != index:
            raise ValueError("Manifest batch indexes must be consecutive, starting at 0")
        check_parameters(
            n_users=record["n_users"], batch_index=index, temperature=record["temperature"],
            random_state=record["random_state"], split_sizes=manifest["split_sizes"],
        )
    files = _data_files(len(manifest["batches"]))
    if set(manifest["files"]) != set(files):
        raise ValueError("Snapshot file list does not match its batches")
    for name in files:
        if digest(directory / name) != manifest["files"][name]:
            raise ValueError(f"Snapshot SHA-256 mismatch: {name}")
    return manifest


def read_snapshot(directory: Path) -> HistoryDataset:
    manifest = _read_manifest(directory)
    parts: list[pd.DataFrame] = []
    for split in SPLITS:
        part = _read_table(directory / f"{split}.csv", EVENT_DTYPES)
        if not part["split"].eq(split).all():
            raise ValueError(f"Unexpected split in {split}.csv")
        parts.append(part)
    events = pd.concat(parts, ignore_index=True).sort_values(
        ["batch_index", "user_id", "event_index"], ignore_index=True,
    )
    return HistoryDataset(
        events, _read_table(directory / "generator/anchors.csv", ANCHOR_DTYPES),
        _read_table(directory / "generator/sources.csv", SOURCE_DTYPES),
        _read_table(directory / "catalog.csv", {"product_id": "string"}), manifest,
    )


def _write_tables(staging: Path, data: HistoryDataset) -> None:
    tables = {
        **{f"{split}.csv": data.events[data.events["split"] == split] for split in SPLITS},
        "catalog.csv": data.catalog,
        "generator/anchors.csv": data.anchors,
        "generator/sources.csv": data.sources,
    }
    for name, table in tables.items():
        table.to_csv(staging / name, index=False, encoding="utf-8", date_format=DATE_FORMAT)


def _copy_distances(staging: Path, inputs: SnapshotInputs, batch_index: int) -> None:
    if inputs.previous_dir is not None:
        for index in range(batch_index):
            name = distance_file(index)
            _ = shutil.copyfile(inputs.previous_dir / name, staging / name)
    _ = shutil.copyfile(inputs.distance_path, staging / distance_file(batch_index))


def publish_snapshot(data: HistoryDataset, inputs: SnapshotInputs) -> None:
    inputs.output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Rename only a fully written snapshot; failures leave the previous step intact.
    with TemporaryDirectory(prefix=".histories-", dir=inputs.output_dir.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        (staging / "generator/distances").mkdir(parents=True)
        _write_tables(staging, data)
        batch_index = len(data.manifest["batches"]) - 1
        _copy_distances(staging, inputs, batch_index)
        data.manifest["files"] = {
            name: digest(staging / name) for name in _data_files(batch_index + 1)
        }
        expected_hash = data.manifest["batches"][-1]["distance"]["sha256"]
        if data.manifest["files"][distance_file(batch_index)] != expected_hash:
            raise ValueError("Distance file changed during generation")
        _ = (staging / "generator/manifest.json").write_text(
            json.dumps(data.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        _ = staging.rename(inputs.output_dir)
