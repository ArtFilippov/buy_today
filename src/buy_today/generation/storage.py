"""Explicit, cumulative history snapshots with immutable previous steps."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from buy_today.auto_eda.checks import check_dataset
from buy_today.generation.histories import (
    ANCHOR_DTYPES, DEFAULT_SPLIT_SIZES, EVENT_DTYPES, SOURCE_DTYPES, SPLITS,
    check_parameters, generate_histories,
)
from buy_today.schema import DATE_FORMAT, ROW_KEY, read_dataset


@dataclass(frozen=True)
class GenerationPaths:
    output_dir: Path
    manifest_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    catalog_path: Path


@dataclass(frozen=True)
class HistoryDataset:
    events: pd.DataFrame
    anchors: pd.DataFrame
    sources: pd.DataFrame
    catalog: pd.DataFrame
    manifest: dict


def _digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _distance_file(index):
    return f"generator/distances/batch_{index:03d}.joblib"


def _data_files(n_batches):
    return [
        *(f"{split}.csv" for split in SPLITS), "catalog.csv",
        "generator/anchors.csv", "generator/sources.csv",
        *(_distance_file(index) for index in range(n_batches)),
    ]


def _read_table(path, dtypes):
    dates = [name for name, dtype in dtypes.items() if dtype == "datetime64[ns]"]
    frame = pd.read_csv(
        path, dtype={name: dtype for name, dtype in dtypes.items() if name not in dates},
        encoding="utf-8", keep_default_na=False,
    )
    if tuple(frame.columns) != tuple(dtypes):
        raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
    for name in dates:
        frame[name] = pd.to_datetime(frame[name], format=DATE_FORMAT, errors="raise")
    frame = frame.astype(dtypes)
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"Empty or missing values in {path}")
    for name, dtype in dtypes.items():
        if dtype == "string" and frame[name].eq("").any():
            raise ValueError(f"Empty identifiers in {path}: {name}")
    return frame


def _source_rows(table, sources):
    keys = ["batch_index", *ROW_KEY]
    positions = pd.MultiIndex.from_frame(sources[keys]).get_indexer(
        pd.MultiIndex.from_frame(table[keys]),
    )
    if (positions < 0).any():
        raise ValueError("History or anchor keys outside their source batch")
    return sources.iloc[positions].reset_index(drop=True)


def _check_snapshot(events, anchors, sources, catalog, manifest):
    batches = manifest["batches"]
    sizes = manifest["split_sizes"]
    if sources.duplicated(list(ROW_KEY)).any():
        raise ValueError("Duplicate source row keys across batches")
    if anchors.user_id.duplicated().any():
        raise ValueError("Duplicate user IDs in anchors")
    if events.event_id.duplicated().any() or events.duplicated(["user_id", "event_index"]).any():
        raise ValueError("Duplicate events or user event positions")
    if not catalog.product_id.is_unique or set(catalog.product_id) != set(sources.product_id):
        raise ValueError("Catalog must contain exactly the unique source products")
    if set(events.user_id) != set(anchors.user_id):
        raise ValueError("Event users must match anchor users")
    if not events.batch_index.eq(events.user_id.map(anchors.set_index("user_id").batch_index)).all():
        raise ValueError("Event batch must match its user's anchor batch")
    if not events.event_index.between(0, sum(sizes) - 1).all():
        raise ValueError("event_index outside the synthetic history")
    expected_splits = np.repeat(SPLITS, sizes)[events.event_index.to_numpy()]
    if not np.array_equal(events.split.to_numpy(), expected_splits):
        raise ValueError("Split does not match synthetic event order")
    counts = events.groupby(["user_id", "split"]).size().unstack(fill_value=0)
    if not counts.reindex(columns=SPLITS, fill_value=0).eq(sizes).all().all():
        raise ValueError("Incorrect per-user split sizes")

    original_events = _source_rows(events, sources)
    for name in ("product_id", "order_purchase_timestamp"):
        if not np.array_equal(events[name].to_numpy(), original_events[name].to_numpy()):
            raise ValueError(f"Event {name} differs from source")
    original_anchors = _source_rows(anchors, sources)
    if not np.array_equal(anchors.source_position, original_anchors.source_position):
        raise ValueError("Anchor position differs from source")
    if set(sources.batch_index) != set(range(len(batches))):
        raise ValueError("Source batch indexes must match manifest")
    if set(anchors.batch_index) != set(range(len(batches))):
        raise ValueError("Anchor batch indexes must match manifest")
    for record in batches:
        index = record["batch_index"]
        source = sources[sources.batch_index == index]
        users = anchors[anchors.batch_index == index]
        if len(source) != record["source"]["rows"] or len(users) != record["n_users"]:
            raise ValueError("Source/user counts differ from manifest")
        if not np.array_equal(np.sort(source.source_position), np.arange(len(source))):
            raise ValueError("Source positions must cover the complete batch")


def read_history_dataset(directory: Path | str) -> HistoryDataset:
    """Read and validate a full generator snapshot, without opening original inputs.

    This generator-side reader includes anchors. Ranking consumers should read
    only their interaction split and catalog, not this generator state.
    """
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "generator/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or not manifest.get("batches"):
        raise ValueError("Expected history snapshot format_version=1 with batches")
    sizes = manifest["split_sizes"]
    for index, record in enumerate(manifest["batches"]):
        if record["batch_index"] != index:
            raise ValueError("Manifest batch indexes must be consecutive, starting at 0")
        check_parameters(
            n_users=record["n_users"], batch_index=index, temperature=record["temperature"],
            random_state=record["random_state"], split_sizes=sizes,
        )
    files = _data_files(len(manifest["batches"]))
    if set(manifest["files"]) != set(files):
        raise ValueError("Snapshot file list does not match its batches")
    for name in files:
        if _digest(directory / name) != manifest["files"][name]:
            raise ValueError(f"Snapshot SHA-256 mismatch: {name}")
    parts = []
    for split in SPLITS:
        part = _read_table(directory / f"{split}.csv", EVENT_DTYPES)
        if not part.split.eq(split).all():
            raise ValueError(f"Unexpected split in {split}.csv")
        parts.append(part)
    events = pd.concat(parts, ignore_index=True).sort_values(
        ["batch_index", "user_id", "event_index"], ignore_index=True,
    )
    anchors = _read_table(directory / "generator/anchors.csv", ANCHOR_DTYPES)
    sources = _read_table(directory / "generator/sources.csv", SOURCE_DTYPES)
    catalog = _read_table(directory / "catalog.csv", {"product_id": "string"})
    _check_snapshot(events, anchors, sources, catalog, manifest)
    return HistoryDataset(events, anchors, sources, catalog, manifest)


def generate_dataset(
    batch_path: Path | str, distance_path: Path | str, output_dir: Path | str, *,
    temperature, previous_dir: Path | str | None = None, n_users=None,
    split_sizes=None, random_state=42,
) -> GenerationPaths:
    """Publish a new cumulative snapshot, sampling only the explicit new batch.

    The output directory must not exist. Previous snapshots are read-only. Split
    sizes default to 70/15/15 initially and are inherited on continuation; an
    explicit change is rejected. n_users defaults to 2000 initially, then 250.
    Temperature is always explicit, in the supplied distance's own units.
    """
    batch_path, distance_path = Path(batch_path).resolve(), Path(distance_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    previous = None
    if previous_dir is not None:
        previous_dir = Path(previous_dir).resolve()
        if previous_dir in output_dir.parents:
            raise ValueError("Output directory must be outside the previous snapshot")
        previous = read_history_dataset(previous_dir)
    batch_index = 0 if previous is None else len(previous.manifest["batches"])
    expected_sizes = DEFAULT_SPLIT_SIZES if previous is None else tuple(previous.manifest["split_sizes"])
    split_sizes = expected_sizes if split_sizes is None else split_sizes
    n_users = (2000 if previous is None else 250) if n_users is None else n_users
    check_parameters(
        n_users=n_users, batch_index=batch_index, temperature=temperature,
        random_state=random_state, split_sizes=split_sizes,
    )
    if previous is not None and tuple(split_sizes) != expected_sizes:
        raise ValueError("split_sizes must remain unchanged when appending histories")
    batch = read_dataset(batch_path)
    check_dataset(batch)
    source_hash = _digest(batch_path)
    if previous is not None:
        keys = pd.MultiIndex.from_frame(batch[list(ROW_KEY)])
        old_keys = pd.MultiIndex.from_frame(previous.sources[list(ROW_KEY)])
        if keys.isin(old_keys).any():
            raise ValueError("New batch overlaps previously received source row keys")
    distance_hash = _digest(distance_path)
    distance = joblib.load(distance_path)
    generated = generate_histories(
        batch, distance=distance, batch_index=batch_index, n_users=n_users,
        temperature=temperature, split_sizes=split_sizes, random_state=random_state,
    )
    sources = batch[[*ROW_KEY, "product_id", "order_purchase_timestamp"]].assign(
        batch_index=batch_index, source_position=np.arange(len(batch)),
    )[list(SOURCE_DTYPES)].astype(SOURCE_DTYPES)
    events, anchors = generated.events, generated.anchors
    records = []
    if previous is not None:
        events = pd.concat([previous.events, events], ignore_index=True)
        anchors = pd.concat([previous.anchors, anchors], ignore_index=True)
        sources = pd.concat([previous.sources, sources], ignore_index=True)
        records = previous.manifest["batches"].copy()
    catalog = pd.DataFrame({"product_id": sorted(sources.product_id.unique())}, dtype="string")
    records.append({
        "batch_index": batch_index, "n_users": int(n_users), "temperature": float(temperature),
        "random_state": int(random_state),
        "source": {"path": str(batch_path), "sha256": source_hash, "rows": len(batch)},
        "distance": {
            "path": str(distance_path), "sha256": distance_hash,
            "snapshot": _distance_file(batch_index),
            "class": f"{type(distance).__module__}.{type(distance).__qualname__}",
            "parameters": repr(distance.get_params(deep=True)) if hasattr(distance, "get_params") else None,
        },
    })
    manifest = {
        "format_version": 1, "split_sizes": [int(size) for size in split_sizes],
        "synthetic_time": "zero-based event_index within user; sampling order",
        "rng": "numpy.default_rng(SeedSequence([random_state, batch_index])); anchors first",
        "numpy_version": np.__version__, "batches": records,
    }
    _check_snapshot(events, anchors, sources, catalog, manifest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after all files have been written; a failed write cannot leave
    # a directory that looks like a complete next step, or alter the prior step.
    with TemporaryDirectory(prefix=".histories-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        (staging / "generator/distances").mkdir(parents=True)
        tables = {
            **{f"{split}.csv": events[events.split == split] for split in SPLITS},
            "catalog.csv": catalog, "generator/anchors.csv": anchors, "generator/sources.csv": sources,
        }
        for name, table in tables.items():
            table.to_csv(staging / name, index=False, encoding="utf-8", date_format=DATE_FORMAT)
        for index in range(batch_index):
            name = _distance_file(index)
            shutil.copyfile(previous_dir / name, staging / name)
        shutil.copyfile(distance_path, staging / _distance_file(batch_index))
        manifest["files"] = {name: _digest(staging / name) for name in _data_files(batch_index + 1)}
        if manifest["files"][_distance_file(batch_index)] != distance_hash:
            raise ValueError("Distance file changed during generation")
        (staging / "generator/manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        staging.rename(output_dir)
    return GenerationPaths(
        output_dir, output_dir / "generator/manifest.json",
        *(output_dir / f"{split}.csv" for split in SPLITS), output_dir / "catalog.csv",
    )
