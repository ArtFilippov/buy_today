"""Cross-table invariants for cumulative history snapshots."""

import numpy as np
import pandas as pd

from buy_today.generation.records import BatchRecord, HistoryDataset, SPLITS
from buy_today.schema import ROW_KEY


def _source_rows(table: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    keys = ["batch_index", *ROW_KEY]
    positions = pd.MultiIndex.from_frame(sources[keys]).get_indexer(
        pd.MultiIndex.from_frame(table[keys]),
    )
    if (positions < 0).any():
        raise ValueError("History or anchor keys outside their source batch")
    return sources.iloc[positions].reset_index(drop=True)


def _check_identities(data: HistoryDataset) -> None:
    events, anchors, sources, catalog = data.events, data.anchors, data.sources, data.catalog
    if sources.duplicated(list(ROW_KEY)).any():
        raise ValueError("Duplicate source row keys across batches")
    if anchors["user_id"].duplicated().any():
        raise ValueError("Duplicate user IDs in anchors")
    if events["event_id"].duplicated().any() or events.duplicated(["user_id", "event_index"]).any():
        raise ValueError("Duplicate events or user event positions")
    if not catalog["product_id"].is_unique or (
        set(catalog["product_id"]) != set(sources["product_id"])
    ):
        raise ValueError("Catalog must contain exactly the unique source products")
    if set(events["user_id"]) != set(anchors["user_id"]):
        raise ValueError("Event users must match anchor users")
    if not events["batch_index"].eq(
        events["user_id"].map(anchors.set_index("user_id")["batch_index"])
    ).all():
        raise ValueError("Event batch must match its user's anchor batch")


def _check_splits(events: pd.DataFrame, sizes: list[int]) -> None:
    if not events["event_index"].between(0, sum(sizes) - 1).all():
        raise ValueError("event_index outside the synthetic history")
    expected_splits = np.repeat(SPLITS, sizes)[events["event_index"].to_numpy()]
    if not np.array_equal(events["split"].to_numpy(), expected_splits):
        raise ValueError("Split does not match synthetic event order")
    counts: pd.DataFrame = events.groupby(["user_id", "split"]).size().unstack(fill_value=0)
    if not counts.reindex(columns=SPLITS, fill_value=0).eq(sizes).all().all():
        raise ValueError("Incorrect per-user split sizes")


def _check_provenance(data: HistoryDataset) -> None:
    original_events = _source_rows(data.events, data.sources)
    for name in ("product_id", "order_purchase_timestamp"):
        if not np.array_equal(data.events[name].to_numpy(), original_events[name].to_numpy()):
            raise ValueError(f"Event {name} differs from source")
    original_anchors = _source_rows(data.anchors, data.sources)
    if not np.array_equal(data.anchors["source_position"], original_anchors["source_position"]):
        raise ValueError("Anchor position differs from source")


def _check_batches(
    sources: pd.DataFrame, anchors: pd.DataFrame, batches: list[BatchRecord],
) -> None:
    if set(sources["batch_index"]) != set(range(len(batches))):
        raise ValueError("Source batch indexes must match manifest")
    if set(anchors["batch_index"]) != set(range(len(batches))):
        raise ValueError("Anchor batch indexes must match manifest")
    for record in batches:
        index = record["batch_index"]
        source = sources[sources["batch_index"] == index]
        users = anchors[anchors["batch_index"] == index]
        if len(source) != record["source"]["rows"] or len(users) != record["n_users"]:
            raise ValueError("Source/user counts differ from manifest")
        if not np.array_equal(np.sort(source["source_position"]), np.arange(len(source))):
            raise ValueError("Source positions must cover the complete batch")


def check_snapshot(data: HistoryDataset) -> None:
    _check_identities(data)
    _check_splits(data.events, data.manifest["split_sizes"])
    _check_provenance(data)
    _check_batches(data.sources, data.anchors, data.manifest["batches"])
