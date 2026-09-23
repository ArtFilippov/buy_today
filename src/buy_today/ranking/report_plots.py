"""Render benchmark metric matrices and per-dataset comparisons."""

import base64
from html import escape
from io import BytesIO
import textwrap

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.typing import RcKeyType
import numpy as np
from numpy.typing import NDArray

from buy_today.ranking.publication import Metadata
from buy_today.ranking.report_validation import METRICS
from buy_today.plot_style import REPORT_STYLE

_STYLE: dict[RcKeyType, object] = {
    **REPORT_STYLE,
    "axes.titlesize": 13,
    "text.parse_math": False,
}
_DARK_CELL = 0.45
_INSIDE_LABEL = 0.8


def _labels(names: list[str], width: int = 34) -> list[str]:
    return [textwrap.fill(name, width) for name in names]


def _annotate_cells(axes: Axes, values: NDArray[np.float64]) -> None:
    for i, j in np.ndindex(values.shape):
        best = values[i, j] == values[:, j].max()
        _ = axes.text(
            j,
            i,
            f"{values[i, j]:.3f}" + (" *" if best else ""),
            ha="center",
            va="center",
            color="white" if values[i, j] < _DARK_CELL else "black",
            fontweight="bold" if best else "normal",
        )


def heatmap(rows: list[Metadata], metric: str) -> Figure:
    models = list(dict.fromkeys(row["model"] for row in rows))
    datasets = sorted({row["dataset"] for row in rows})
    lookup = {(row["model"], row["dataset"]): row[metric] for row in rows}
    values = np.array(
        [[lookup[model, dataset] for dataset in datasets] for model in models], dtype=np.float64
    )
    labels = _labels(models)
    with mpl.rc_context(_STYLE):
        figure = Figure(
            figsize=(
                max(8, 4 + 1.5 * len(datasets)),
                max(3.5, 2.2 + 0.35 * sum(label.count("\n") + 1 for label in labels)),
            ),
            layout="constrained",
        )
        _canvas = FigureCanvasAgg(figure)
        _heatmap_panel(
            figure,
            values,
            (labels, _labels(datasets, 18)),
            f"{METRICS[metric]} · K = {rows[0]['k']} · больше — лучше",
        )
        return figure


def _heatmap_panel(
    figure: Figure, values: NDArray[np.float64], labels: tuple[list[str], list[str]], title: str
) -> None:
    axes = figure.add_subplot()
    _render_heatmap(axes, values, labels, title)
    _colorbar = figure.colorbar(
        axes.images[0], ax=axes, label="Среднее по пользователям", ticks=np.linspace(0, 1, 6)
    )


def _render_heatmap(
    axes: Axes, values: NDArray[np.float64], labels: tuple[list[str], list[str]], title: str
) -> None:
    _mappable = axes.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    _ = axes.set_yticks(range(len(labels[0])), labels[0])
    _ = axes.set_xticks(range(len(labels[1])), labels[1])
    _ = axes.set(title=title, xlabel="Датасет", ylabel="Конфигурация модели")
    _annotate_cells(axes, values)


def _bar_label(axes: Axes, rectangle: Rectangle, value: float, best: bool) -> None:
    inside = value > _INSIDE_LABEL
    _ = axes.text(
        value - 0.015 if inside else value + 0.015,
        rectangle.get_y() + rectangle.get_height() / 2,
        f"{value:.3f}" + (" *" if best else ""),
        va="center",
        ha="right" if inside else "left",
        color="white" if inside else "#202530",
        fontweight="bold" if best else "normal",
    )


def _metric_panel(axes: Axes, rows: list[Metadata], metric: str, title: str) -> None:
    values: list[float] = [row[metric] for row in rows]
    rectangles = axes.barh(range(len(rows)), values, color="#287a9f", height=0.65)
    _ = axes.set_yticks(range(len(rows)), _labels([row["model"] for row in rows]))
    _ = axes.set_xlim(0, 1)
    _ = axes.set_xticks(np.linspace(0, 1, 6))
    _ = axes.set(title=title, xlabel="Среднее по пользователям · больше — лучше")
    axes.grid(axis="x", alpha=0.2)
    axes.set_axisbelow(True)
    for rectangle, value in zip(rectangles.patches, values, strict=True):
        _bar_label(axes, rectangle, value, value == max(values))


def dataset_plot(rows: list[Metadata]) -> Figure:
    labels = _labels([row["model"] for row in rows])
    with mpl.rc_context(_STYLE):
        figure = Figure(
            figsize=(13, max(3.5, 2 + 0.42 * sum(label.count("\n") + 1 for label in labels))),
            layout="constrained",
        )
        _canvas = FigureCanvasAgg(figure)
        axes = figure.subplots(1, 2, sharey=True)
        for panel, (metric, title), axis in zip("ab", METRICS.items(), axes, strict=True):
            _metric_panel(axis, rows, metric, f"({panel}) {title}")
        axes[0].invert_yaxis()
        _title = figure.suptitle(
            textwrap.fill(rows[0]["dataset"], 90)
            + f" · K = {rows[0]['k']} · пользователей: {rows[0]['n_users']}"
        )
        return figure


def image(figure: Figure, title: str) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    figure.clear()
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    alternate = escape(title, quote=True)
    return f'<figure><img alt="{alternate}" src="data:image/png;base64,{encoded}"></figure>'
