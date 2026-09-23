"""Explicit, cumulative history snapshots with immutable previous steps."""

from pathlib import Path
from typing import Any, Unpack

import joblib
import numpy as np
import pandas as pd

from buy_today.auto_eda.checks import check_dataset
from buy_today.generation.histories import SOURCE_DTYPES, generate_histories, validate_parameters
from buy_today.generation.persistence import digest, distance_file, publish_snapshot, read_snapshot
from buy_today.generation.records import (
    DEFAULT_SPLIT_SIZES,
    BatchRecord,
    DatasetOptions,
    DatasetSettings,
    DistanceRecord,
    GenerationPaths as GenerationPaths,
    HistoryDataset as HistoryDataset,
    HistoryManifest,
    SamplingParameters,
    SnapshotInputs,
)
from buy_today.generation.validation import check_snapshot
from buy_today.schema import ROW_KEY, read_dataset


def read_history_dataset(directory: Path | str) -> HistoryDataset:
    """Read and validate a full generator snapshot, without opening original inputs.

    This generator-side reader includes anchors. Ranking consumers should read
    only their interaction split and catalog, not this generator state.

    Args:
        directory (Path | str): Published snapshot directory.

    Returns:
        HistoryDataset: Validated events, anchors, sources, catalog and manifest.
    """
    data = read_snapshot(Path(directory).resolve())
    check_snapshot(data)
    return data


def _previous_snapshot(inputs: SnapshotInputs) -> HistoryDataset | None:
    if inputs.output_dir.exists():
        raise ValueError(f"Output directory already exists: {inputs.output_dir}")
    if inputs.previous_dir is None:
        return None
    if inputs.previous_dir in inputs.output_dir.parents:
        raise ValueError("Output directory must be outside the previous snapshot")
    return read_history_dataset(inputs.previous_dir)


def _parameters(
    previous: HistoryDataset | None, temperature: float, settings: DatasetSettings
) -> SamplingParameters:
    expected_sizes = (
        DEFAULT_SPLIT_SIZES if previous is None else tuple(previous.manifest["split_sizes"])
    )
    sizes = settings.split_sizes
    n_users = settings.n_users
    parameters = SamplingParameters(
        n_users=(2000 if previous is None else 250) if n_users is None else n_users,
        batch_index=0 if previous is None else len(previous.manifest["batches"]),
        temperature=temperature, random_state=settings.random_state,
        split_sizes=expected_sizes if sizes is None else sizes,
    )
    validate_parameters(parameters)
    if previous is not None and tuple(parameters.split_sizes) != expected_sizes:
        raise ValueError("split_sizes must remain unchanged when appending histories")
    return parameters


def _read_batch(path: Path, previous: HistoryDataset | None) -> pd.DataFrame:
    batch = read_dataset(path)
    _ = check_dataset(batch)
    if previous is not None:
        keys: pd.MultiIndex = pd.MultiIndex.from_frame(batch[list(ROW_KEY)])
        old_keys = pd.MultiIndex.from_frame(previous.sources[list(ROW_KEY)])
        if keys.isin(old_keys).any():
            raise ValueError("New batch overlaps previously received source row keys")
    return batch


def _distance_record(path: Path, distance: Any, index: int, checksum: str) -> DistanceRecord:
    # The estimator is unpickled dynamically and may optionally expose get_params.
    return {
        "path": str(path), "sha256": checksum, "snapshot": distance_file(index),
        "class": f"{type(distance).__module__}.{type(distance).__qualname__}",
        "parameters": (
            repr(distance.get_params(deep=True)) if hasattr(distance, "get_params") else None
        ),
    }


def _new_histories(
    batch: pd.DataFrame, inputs: SnapshotInputs, parameters: SamplingParameters
) -> HistoryDataset:
    source_hash, distance_hash = digest(inputs.batch_path), digest(inputs.distance_path)
    distance: Any = joblib.load(inputs.distance_path)
    generated = generate_histories(
        batch, distance=distance, batch_index=int(parameters.batch_index),
        n_users=int(parameters.n_users),
        temperature=parameters.temperature, split_sizes=parameters.split_sizes,
        random_state=parameters.random_state,
    )
    sources = batch[[*ROW_KEY, "product_id", "order_purchase_timestamp"]].assign(
        batch_index=int(parameters.batch_index), source_position=np.arange(len(batch)),
    )[list(SOURCE_DTYPES)].astype(SOURCE_DTYPES)
    record: BatchRecord = {
        "batch_index": int(parameters.batch_index), "n_users": int(parameters.n_users),
        "temperature": float(parameters.temperature), "random_state": int(parameters.random_state),
        "source": {"path": str(inputs.batch_path), "sha256": source_hash, "rows": len(batch)},
        "distance": _distance_record(
            inputs.distance_path, distance, int(parameters.batch_index), distance_hash,
        ),
    }
    manifest: HistoryManifest = {
        "format_version": 1, "split_sizes": [int(size) for size in parameters.split_sizes],
        "synthetic_time": "zero-based event_index within user; sampling order",
        "rng": "numpy.default_rng(SeedSequence([random_state, batch_index])); anchors first",
        "numpy_version": np.__version__, "batches": [record], "files": {},
    }
    return HistoryDataset(generated.events, generated.anchors, sources, pd.DataFrame(), manifest)


def _accumulate(current: HistoryDataset, previous: HistoryDataset | None) -> HistoryDataset:
    events, anchors, sources = current.events, current.anchors, current.sources
    if previous is not None:
        events = pd.concat([previous.events, events], ignore_index=True)
        anchors = pd.concat([previous.anchors, anchors], ignore_index=True)
        sources = pd.concat([previous.sources, sources], ignore_index=True)
        current.manifest["batches"] = [*previous.manifest["batches"], *current.manifest["batches"]]
    catalog = pd.DataFrame({"product_id": sorted(sources["product_id"].unique())}, dtype="string")
    return HistoryDataset(events, anchors, sources, catalog, current.manifest)


def _build_snapshot(
    inputs: SnapshotInputs, temperature: float, settings: DatasetSettings,
) -> HistoryDataset:
    previous = _previous_snapshot(inputs)
    parameters = _parameters(previous, temperature, settings)
    batch = _read_batch(inputs.batch_path, previous)
    return _accumulate(_new_histories(batch, inputs, parameters), previous)


def generate_dataset(
    batch_path: Path | str,
    distance_path: Path | str,
    output_dir: Path | str,
    *,
    temperature: float,
    previous_dir: Path | str | None = None,
    **options: Unpack[DatasetOptions],
) -> GenerationPaths:
    """Publish a new cumulative snapshot, sampling only the explicit new batch.

    The output directory must not exist. Previous snapshots are read-only. Split
    sizes default to 70/15/15 initially and are inherited on continuation; an
    explicit change is rejected. n_users defaults to 2000 initially, then 250.
    Temperature is always explicit, in the supplied distance's own units.

    Args:
        batch_path (Path | str): CSV containing the new source batch.
        distance_path (Path | str): Serialized ready distance estimator.
        output_dir (Path | str): New directory to publish atomically.
        temperature (float): Positive finite temperature in distance units.
        previous_dir (Path | str | None, default=None): Prior snapshot to extend, if any.
        **options (Unpack[DatasetOptions]): User count, split sizes and random seed.

    Returns:
        GenerationPaths: Absolute paths to the published snapshot and consumer files.
    """
    settings = DatasetSettings(**options)
    inputs = SnapshotInputs(
        Path(batch_path).resolve(), Path(distance_path).resolve(), Path(output_dir).resolve(),
        None if previous_dir is None else Path(previous_dir).resolve(),
    )
    data = _build_snapshot(inputs, temperature, settings)
    check_snapshot(data)
    publish_snapshot(data, inputs)
    return GenerationPaths(
        inputs.output_dir, inputs.output_dir / "generator/manifest.json",
        inputs.output_dir / "train.csv", inputs.output_dir / "validation.csv",
        inputs.output_dir / "test.csv", inputs.output_dir / "catalog.csv",
    )
