"""Ranking-only tables: interactions and the complete candidate catalog."""

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import TypeIs

import numpy as np
import pandas as pd

from buy_today.ranking.domain import RankingData


def check_integer(value: object, name: str, minimum: int, maximum: float) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not minimum <= int(value) <= maximum
    ):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _check_identifiers(frame: object, columns: Sequence[str], name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != tuple(columns):
        raise ValueError(f"{name} must be a DataFrame with exactly {tuple(columns)}")
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"{name} must be nonempty and contain no missing values")
    for column in columns:
        if not frame[column].map(_valid_identifier).all():
            raise ValueError(f"{name}.{column} must contain nonempty string identifiers")
    return frame


def _is_ranking_data(data: object) -> TypeIs[RankingData[object]]:
    return isinstance(data, RankingData)


def check_ranking_data(data: object) -> None:
    if not _is_ranking_data(data):
        raise ValueError("X must be RankingData(interactions, catalog)")
    interactions = _check_identifiers(data.interactions, ("user_id", "product_id"), "interactions")
    catalog = _check_identifiers(data.catalog, ("product_id",), "catalog")
    if not catalog.product_id.is_unique:
        raise ValueError("Duplicate product IDs in catalog")
    if not interactions["product_id"].isin(catalog["product_id"]).all():
        raise ValueError("Interaction products outside catalog")


def read_ranking_data(directory: Path | str, *, split: str = "train") -> RankingData[pd.DataFrame]:
    """Read only <split>.csv and catalog.csv, never generator state or other splits.

    Event IDs, positions and split tags are checked before projecting to the two
    interaction columns. Source keys/times and batch/cluster/anchor information
    are not model inputs. Full generator provenance is validated by its reader.

    Args:
        directory (Path | str): History snapshot containing split and catalog CSVs.
        split (str, default="train"): One of train, validation or test.

    Returns:
        RankingData[pd.DataFrame]: Validated user/product interactions and candidate catalog.

    Raises:
        ValueError: Split, event identifiers or positions are invalid.
    """
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation or test")
    directory = Path(directory).resolve()
    columns = ("event_id", "user_id", "event_index", "split", "product_id")
    events = pd.read_csv(
        directory / f"{split}.csv",
        usecols=list(columns),
        dtype={**dict.fromkeys(columns, "string"), "event_index": "int64"},
        encoding="utf-8",
        keep_default_na=False,
    )
    _ = _check_identifiers(
        events[["event_id", "user_id", "product_id"]],
        ("event_id", "user_id", "product_id"),
        "events",
    )
    if not (events["split"] == split).all():
        raise ValueError(f"Unexpected split in {split}.csv")
    if (
        not events.event_id.is_unique
        or pd.DataFrame.duplicated(events, ["user_id", "event_index"]).any()
    ):
        raise ValueError("Duplicate events or user event positions")
    if (events.event_index < 0).any():
        raise ValueError("event_index must be nonnegative")
    catalog = pd.read_csv(
        directory / "catalog.csv",
        dtype="string",
        encoding="utf-8",
        keep_default_na=False,
    )
    data = RankingData(events[["user_id", "product_id"]].copy(), catalog)
    check_ranking_data(data)
    return data
