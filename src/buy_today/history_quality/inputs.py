"""Validate provenance and restore categories without dropping any observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buy_today.schema import ROW_KEY


CATEGORY = "product_category_name"
ITEM_KEY = "order_item_id"


def _required(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    if not frame.columns.is_unique or not set(columns).issubset(frame.columns):
        raise ValueError(f"{name}: expected unique columns including {columns}")
    if frame.empty or frame[columns].isna().any().any():
        raise ValueError(f"{name}: empty table or missing values")
    for column in columns:
        if column != ITEM_KEY and not frame[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(f"{name}: {column} must contain nonblank strings")
    if ITEM_KEY in columns and not pd.api.types.is_integer_dtype(frame[ITEM_KEY]):
        raise ValueError(f"{name}: order_item_id must contain integers")


def _categories(table: pd.DataFrame, reference: pd.DataFrame, name: str) -> pd.Series[str]:
    positions = pd.MultiIndex.from_frame(reference[list(ROW_KEY)]).get_indexer(
        pd.MultiIndex.from_frame(table[list(ROW_KEY)])
    )
    if (positions < 0).any():
        raise ValueError(f"{name}: source keys missing from reference")
    return reference[CATEGORY].iloc[positions].reset_index(drop=True)


def categorized_events(
    events: pd.DataFrame, anchors: pd.DataFrame, reference: pd.DataFrame,
) -> pd.DataFrame:
    _required(reference, [*ROW_KEY, CATEGORY], "Reference")
    _required(events, ["user_id", *ROW_KEY], "Events")
    _required(anchors, ["user_id", *ROW_KEY], "Anchors")
    if reference.duplicated(list(ROW_KEY)).any():
        raise ValueError("Reference: duplicate source keys")
    if anchors["user_id"].duplicated().any():
        raise ValueError("Anchors: duplicate user IDs")
    if set(events["user_id"]) != set(anchors["user_id"]):
        raise ValueError("Event users must match anchor users")
    anchor_categories = pd.Series(
        _categories(anchors, reference, "Anchors").array,
        index=anchors["user_id"],
    )
    result = events[["user_id"]].reset_index(drop=True).copy()
    result[CATEGORY] = _categories(events, reference, "Events")
    result["anchor_match"] = result[CATEGORY].eq(result["user_id"].map(anchor_categories))
    return result


def check_reference_sources(reference: pd.DataFrame, sources: pd.DataFrame) -> None:
    _required(reference, [*ROW_KEY, CATEGORY], "Reference")
    if reference.duplicated(list(ROW_KEY)).any():
        raise ValueError("Reference: duplicate source keys")
    reference_keys: pd.MultiIndex = pd.MultiIndex.from_frame(reference[list(ROW_KEY)])
    source_keys: pd.MultiIndex = pd.MultiIndex.from_frame(sources[list(ROW_KEY)])
    missing = int((~source_keys.isin(reference_keys)).sum())
    extra = int((~reference_keys.isin(source_keys)).sum())
    if missing or extra:
        raise ValueError(f"Reference source keys differ: missing={missing}, extra={extra}")
    aligned = reference.iloc[reference_keys.get_indexer(source_keys)]
    for column in ("product_id", "order_purchase_timestamp"):
        if not np.array_equal(aligned[column].to_numpy(), sources[column].to_numpy()):
            raise ValueError(f"Reference {column} differs from snapshot sources")
