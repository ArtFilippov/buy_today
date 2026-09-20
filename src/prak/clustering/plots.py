"""t-SNE of the evaluation matrix; labels are only used for point colours."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from sklearn.manifold import TSNE

from prak.clustering.evaluation import EvaluationResult, validate_evaluation_parameters


@dataclass(frozen=True)
class ProjectionResult:
    coordinates: np.ndarray | None
    parameters: dict
    reason: str | None


def project_tsne(result: EvaluationResult, *, random_state=42) -> ProjectionResult:
    """Use exactly the sampled distances, retaining the original matrix intact."""
    validate_evaluation_parameters(1, random_state)
    size = len(result.sample_positions)
    if size < 3:
        return ProjectionResult(
            None, {}, f"t-SNE не построена: требуется хотя бы 3 строки; получено {size}.",
        )
    projection = TSNE(
        n_components=2, metric="precomputed", init="random", random_state=random_state,
        perplexity=min(30.0, float(size - 1)), learning_rate="auto",
    )
    # sklearn may square precomputed distances in place in some implementations.
    coordinates = projection.fit_transform(result.distances.copy())
    return ProjectionResult(coordinates, projection.get_params(), None)


def plot_tsne(result: EvaluationResult, projection: ProjectionResult) -> Figure:
    if projection.coordinates is None:
        raise ValueError(projection.reason)
    labels = np.unique(result.labels)
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axis = plt.subplots(figsize=(10, 8.4), dpi=130)
        colours = plt.get_cmap("tab20")
        markers = ("o", "s", "^", "D", "v", "P", "X")
        for index, label in enumerate(labels):
            points = projection.coordinates[result.labels == label]
            axis.scatter(
                points[:, 0], points[:, 1], s=16, alpha=0.9,
                color=colours(index % 20), marker=markers[(index // 20) % len(markers)],
                edgecolors="#404040", linewidths=0.25, label=str(label),
            )
        axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="Кластеризация · t-SNE расстояний")
        axis.set_aspect("equal", adjustable="box")
        axis.legend(
            title="Кластер", loc="upper left", bbox_to_anchor=(1.02, 1),
            ncols=max(1, (len(labels) + 19) // 20), frameon=False, markerscale=1.5,
        )
        axis.grid(alpha=0.15)
        axis.set_axisbelow(True)
        score = "не определён" if result.silhouette is None else f"{result.silhouette:.6f}"
        figure.text(
            0.08, 0.025,
            f"Точка — позиция заказа; выборка: {len(result.labels):,} "
            f"(новых: {result.new_count:,}, прежних: {result.previous_count:,}).\n"
            f"Силуэт: {score} по исходным расстояниям; геометрия t-SNE — иллюстрация.",
            fontsize=11,
        )
        figure.tight_layout(rect=(0, 0.10, 1, 1))
    return figure
