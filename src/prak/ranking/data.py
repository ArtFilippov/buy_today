"""Ranking-only tables: interactions and the complete candidate catalog."""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankingData:
    """One split of purchases (repetitions retained) and unique candidate IDs."""

    interactions: pd.DataFrame
    catalog: pd.DataFrame


def _check_integer(value, name, minimum, maximum):
    if (
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _check_identifiers(frame, columns, name):
    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != tuple(columns):
        raise ValueError(f"{name} must be a DataFrame with exactly {tuple(columns)}")
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"{name} must be nonempty and contain no missing values")
    for column in columns:
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{name}.{column} must contain nonempty string identifiers")


def check_ranking_data(data):
    if not isinstance(data, RankingData):
        raise ValueError("X must be RankingData(interactions, catalog)")
    _check_identifiers(data.interactions, ("user_id", "product_id"), "interactions")
    _check_identifiers(data.catalog, ("product_id",), "catalog")
    if not data.catalog.product_id.is_unique:
        raise ValueError("Duplicate product IDs in catalog")
    if not data.interactions.product_id.isin(data.catalog.product_id).all():
        raise ValueError("Interaction products outside catalog")


def read_ranking_data(directory: Path | str, *, split="train") -> RankingData:
    """Read only <split>.csv and catalog.csv, never generator state or other splits.

    Event IDs, positions and split tags are checked before projecting to the two
    interaction columns. Source keys/times and batch/cluster/anchor information
    are not model inputs. Full generator provenance is validated by its reader.
    """
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation or test")
    directory = Path(directory).resolve()
    columns = ("event_id", "user_id", "event_index", "split", "product_id")
    events = pd.read_csv(
        directory / f"{split}.csv", usecols=list(columns),
        dtype={name: "int64" if name == "event_index" else "string" for name in columns},
        encoding="utf-8", keep_default_na=False,
    )
    _check_identifiers(events[["event_id", "user_id", "product_id"]],
                       ("event_id", "user_id", "product_id"), "events")
    if not events.split.eq(split).all():
        raise ValueError(f"Unexpected split in {split}.csv")
    if not events.event_id.is_unique or events.duplicated(["user_id", "event_index"]).any():
        raise ValueError("Duplicate events or user event positions")
    if (events.event_index < 0).any():
        raise ValueError("event_index must be nonnegative")
    catalog = pd.read_csv(
        directory / "catalog.csv", dtype="string", encoding="utf-8", keep_default_na=False,
    )
    data = RankingData(events[["user_id", "product_id"]].copy(), catalog)
    check_ranking_data(data)
    return data
