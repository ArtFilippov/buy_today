"""Frozen configuration shared by run storage and orchestration."""

from dataclasses import asdict, dataclass
from typing import Any

from buy_today.auto_eda.domain import DriftThresholds
from buy_today.generation.parameters import check_parameters

SVD_MODEL = "svd"


@dataclass(frozen=True)
class _GenerationParameters:
    model: str = SVD_MODEL
    random_state: int = 42
    temperature: float = 86400.0
    initial_users: int = 2000
    additional_users: int = 250
    split_sizes: tuple[int, int, int] = (70, 15, 15)


@dataclass(frozen=True)
class PipelineConfig(_GenerationParameters):
    k: int = 10
    svd_n_components: int = 32
    svd_n_iter: int = 7
    temporal_n_clusters: int | None = None
    max_evaluation_rows: int = 1000
    thresholds: DriftThresholds = DriftThresholds()

    def __post_init__(self) -> None:
        if self.model not in ("random", SVD_MODEL):
            raise ValueError("model must be random or svd")
        for users in (self.initial_users, self.additional_users):
            check_parameters(
                n_users=users,
                batch_index=0,
                temperature=self.temperature,
                random_state=self.random_state,
                split_sizes=self.split_sizes,
            )
        for name in ("k", "svd_n_components", "svd_n_iter", "max_evaluation_rows"):
            _positive_integer(name, getattr(self, name))
        if self.temporal_n_clusters is not None:
            _positive_integer("temporal_n_clusters", self.temporal_n_clusters)
        _check_thresholds(self.thresholds)


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _check_thresholds(value: object) -> None:
    if not isinstance(value, DriftThresholds):
        raise ValueError("thresholds must be DriftThresholds")


def config_dict(config: PipelineConfig) -> dict[str, Any]:
    value = asdict(config)
    value["split_sizes"] = list(config.split_sizes)
    return value


def parse_config(value: dict[str, Any]) -> PipelineConfig:
    value = dict(value)
    value["split_sizes"] = tuple(value["split_sizes"])
    value["thresholds"] = DriftThresholds(**value["thresholds"])
    return PipelineConfig(**value)
