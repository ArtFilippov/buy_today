"""Library-independent contracts for ranking estimators and experiment records."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, TypedDict

SVD = "svd"


class Predictor[Prediction, User = str, Cutoff = int](Protocol):
    def predict(self, user_id: User, k: Cutoff, /) -> Prediction: ...


class Estimator[Data, Prediction](Predictor[Prediction], Protocol):
    def fit(self, data: Data, /, y: object = None) -> Self: ...

    def get_params(self, deep: bool = True) -> dict[str, object]: ...


class Source(TypedDict):
    path: str
    sha256: str
    rows: int


class Dataset(TypedDict):
    path: str
    inputs: dict[str, Source]
    n_users: int
    user_ids_sha256: str


class Timings(TypedDict):
    fit_seconds: float
    evaluation_seconds: float


class DatasetRun(Dataset, Timings):
    """Input identity and timings of one model/dataset pair."""


@dataclass(frozen=True)
class RankingData[Table = Any]:
    interactions: Table
    catalog: Table


@dataclass(frozen=True)
class RankingEvaluation[Table = Any]:
    metrics: dict[str, float]
    per_user: Table


@dataclass(frozen=True)
class TrainingPaths:
    model_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class EvaluationPaths:
    metrics_path: Path
    per_user_path: Path


@dataclass(frozen=True)
class BenchmarkReportPaths:
    html_path: Path
    csv_path: Path


@dataclass(frozen=True)
class EvaluationSource[Data]:
    data: Data
    inputs: dict[str, Source]
    model_path: Path
    model_hash: str
