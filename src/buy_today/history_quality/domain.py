"""Serializable results of descriptive history-quality evaluation."""

from typing import TypedDict


class IIDBaseline(TypedDict):
    repeats: int
    random_state: int
    rng: str
    sampling: str
    aggregation: str
    user_top_category_share_mean: float
    user_effective_category_count_mean: float


class HistoryQuality(TypedDict):
    n_users: int
    n_events: int
    n_reference_rows: int
    metrics: dict[str, float]
    iid: IIDBaseline
