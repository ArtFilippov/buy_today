import hashlib
from html.parser import HTMLParser
import json
import math
import sys
from contextlib import contextmanager
from collections.abc import Generator
from typing import override

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import nbformat
from nbclient.exceptions import CellExecutionError
import numpy as np
import pandas as pd
import pytest

import fixture_types as ft

from buy_today.auto_eda import report_dataset, report_drift
from buy_today.auto_eda.checks import check_dataset
from buy_today.auto_eda.drift import DriftThresholds
from buy_today.auto_eda.notebook import ReportPaths, execute_report, write_metrics
from buy_today.auto_eda.plots import (
    category_shares,
    comparison_category_shares,
    plot_categories_comparison,
    plot_price,
    plot_price_comparison,
    plot_states,
    plot_states_comparison,
)

matplotlib.use("Agg")


@contextmanager
def _figure_axis(figure: Figure) -> Generator[Axes]:
    try:
        assert len(figure.axes) == 1
        yield figure.axes[0]
    finally:
        plt.close(figure)


def _rectangles(axis: Axes) -> list[Rectangle]:
    patches = list(axis.patches)
    assert all(isinstance(patch, Rectangle) for patch in patches)
    return [patch for patch in patches if isinstance(patch, Rectangle)]


def _bar_containers(axis: Axes) -> tuple[BarContainer, BarContainer]:
    first, second = axis.containers
    assert isinstance(first, BarContainer)
    assert isinstance(second, BarContainer)
    return first, second


class HTMLAssets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []
        self.external_assets: list[str] = []

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "img":
            self.images.append(attributes.get("src") or "")
        if tag in {"img", "script", "iframe", "link"}:
            source = attributes.get("src") or attributes.get("href", "")
            if source and not source.startswith("data:"):
                self.external_assets.append(source)


def test_eda_executes_in_project_python_and_exports_standalone_html(
    working_frame: ft.DataFrame,
    write_dataset: ft.DatasetWriter,
    tmp_path: ft.Path,
) -> None:
    dataset = write_dataset(working_frame)
    paths = report_dataset(str(dataset), str(tmp_path / "EDA report"))
    notebook: nbformat.NotebookNode = nbformat.read(paths.notebook_path, as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    outputs = [output for cell in code for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    assert sum("image/png" in output.get("data", {}) for output in outputs) == 3
    text = "\n".join(output.get("text", "") for output in outputs)
    assert sys.executable in text
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() in text
    assert "12 позиций × 35 колонок" in text
    assert "Дрейф по одному датасету не оценивается" in text
    assert "check_dataset(frame)" in "\n".join(cell.source for cell in code)

    assert paths.metrics_path == paths.notebook_path.with_name("metrics.json")
    assert ReportPaths(paths.notebook_path, paths.html_path).metrics_path == paths.metrics_path
    metrics_text = paths.metrics_path.read_text(encoding="utf-8")
    metrics = json.loads(metrics_text)
    assert metrics == {
        "format_version": 1,
        "kind": "eda",
        "input": {"path": str(dataset), "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()},
        "rows": 12,
        "n_columns": 35,
        "n_orders": 6,
        "n_customers": 1,
        "n_products": 1,
        "period_start": "2018-01-01 00:00:00",
        "period_end": "2018-01-01 05:00:00",
        "checks": check_dataset(working_frame).to_dict("records"),
    }
    for key in ("format_version", "rows", "n_columns", "n_orders", "n_customers", "n_products"):
        assert isinstance(metrics[key], int) and not isinstance(metrics[key], bool)
    assert len(metrics["checks"]) == 7
    for check in metrics["checks"]:
        assert set(check) == {"Проверка", "Результат", "Фактически", "Ожидается"}
        assert all(isinstance(value, str) for value in check.values())
        assert check["Результат"] == "OK"
    assert metrics_text == json.dumps(metrics, ensure_ascii=False, allow_nan=False, indent=2) + "\n"

    html = paths.html_path.read_text(encoding="utf-8")
    parser = HTMLAssets()
    parser.feed(html)
    assert len(parser.images) == 3
    assert all(source.startswith("data:image/png;base64,") for source in parser.images)
    assert not parser.external_assets
    for expected in (
        "SHA-256",
        "Размеры и период",
        "2018-01-01 05:00:00",
        "Пропуски",
        "Ожидается",
        "Итог",
    ):
        assert expected in html
    assert set(path.name for path in paths.html_path.parent.iterdir()) == {
        "report.html",
        "report.ipynb",
        "metrics.json",
    }


def test_invalid_data_stops_notebook_before_plots(
    working_frame: ft.DataFrame, write_dataset: ft.DatasetWriter, tmp_path: ft.Path
) -> None:
    working_frame.loc[0, "price"] = 0
    output = tmp_path / "invalid report"
    output.mkdir()
    (output / "metrics.json").write_text('{"kind": "eda"}\n', encoding="utf-8")
    with pytest.raises(CellExecutionError, match="Конечная положительная цена"):
        report_dataset(write_dataset(working_frame), output)
    notebook: nbformat.NotebookNode = nbformat.read(output / "report.ipynb", as_version=4)
    outputs = [
        output for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs
    ]
    assert any(output.output_type == "error" for output in outputs)
    assert not any("image/png" in output.get("data", {}) for output in outputs)
    assert not (output / "report.html").exists()
    assert not (output / "metrics.json").exists()


def test_kernel_technical_error_propagates(tmp_path: ft.Path) -> None:
    (tmp_path / "metrics.json").write_text('{"kind": "eda"}\n', encoding="utf-8")
    with pytest.raises(CellExecutionError, match="rendering failed"):
        execute_report(
            [
                nbformat.v4.new_code_cell(
                    "from pathlib import Path\n"
                    + "assert not Path('metrics.json').exists()\n"
                    + "raise RuntimeError('rendering failed')"
                )
            ],
            tmp_path,
            title="Failure",
        )
    assert not (tmp_path / "report.html").exists()
    assert not (tmp_path / "metrics.json").exists()


def test_report_kernel_uses_requested_thread_limit(
    tmp_path: ft.Path, monkeypatch: ft.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    paths = execute_report(
        [
            nbformat.v4.new_code_cell(
                "import os\nimport numpy as np\nfrom threadpoolctl import threadpool_info\n"
                + "np.ones((4, 4)) @ np.ones((4, 4))\n"
                + "assert os.environ['OMP_NUM_THREADS'] == '1'\n"
                + "assert all(pool['num_threads'] == 1 for pool in threadpool_info())\n"
                + "print('thread policy verified')"
            ),
        ],
        tmp_path,
        title="Threads",
        thread_limit=1,
    )
    assert "thread policy verified" in paths.html_path.read_text(encoding="utf-8")


def test_metrics_serialization_handles_numpy_and_rejects_nonfinite_values(
    tmp_path: ft.Path,
) -> None:
    path = tmp_path / "metrics.json"
    write_metrics(
        {
            "count": np.int64(3),
            "score": np.float32(0.5),
            "drift": np.bool_(False),
            "parameters": {"weights": np.array([1, 2])},
            "reason": "Не определён",
        },
        path,
    )
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == {
        "count": 3,
        "score": 0.5,
        "drift": False,
        "parameters": {"weights": [1, 2]},
        "reason": "Не определён",
    }
    assert "Не определён" in text
    invalid_path = tmp_path / "invalid.json"
    for value in (math.nan, math.inf, -math.inf, np.float32("nan"), np.array([np.inf])):
        with pytest.raises(ValueError, match="Out of range float values"):
            write_metrics({"value": value}, invalid_path)
        assert not invalid_path.exists()


def test_top_ten_ties_are_alphabetical_and_preserve_full_mass(working_frame: ft.DataFrame) -> None:
    shares = category_shares(working_frame["product_category_name"])
    assert list(shares.index) == [f"category_{index:02d}" for index in range(10)] + ["Прочие"]
    assert shares.loc["Прочие"] == pytest.approx(2 / 12)
    assert shares.sum() == pytest.approx(1)


@pytest.mark.parametrize(
    "prices", [[100.0] * 12, [0.01, 1, 2, 3, 5, 10, 20, 50, 100, 500, 1000, 10000]]
)
def test_price_plot_has_one_axis_log_scale_and_unit_mass(
    working_frame: ft.DataFrame, prices: list[float]
) -> None:
    working_frame["price"] = prices
    with _figure_axis(plot_price(working_frame)) as axis:
        assert axis.get_xscale() == "log"
        patches = _rectangles(axis)
        assert sum(patch.get_height() for patch in patches) == pytest.approx(1)
        assert all(patch.get_width() > 0 for patch in patches)


def test_state_shares_count_positions_not_unique_orders(working_frame: ft.DataFrame) -> None:
    with _figure_axis(plot_states(working_frame)) as axis:
        assert [tick.get_text() for tick in axis.get_yticklabels()] == ["MG", "RJ", "SP"]
        np.testing.assert_allclose(
            [patch.get_width() for patch in _rectangles(axis)], [1 / 12, 3 / 12, 8 / 12]
        )
        assert not axis.get_xlim()[0]


def test_deda_executes_comparison_and_exports_standalone_html(
    working_frame: ft.DataFrame,
    new_batch: ft.DataFrame,
    write_dataset: ft.DatasetWriter,
    tmp_path: ft.Path,
) -> None:
    reference = write_dataset(working_frame, "reference ' data.csv")
    batch = write_dataset(new_batch)
    before = [path.read_bytes() for path in (reference, batch)]
    paths = report_drift(
        str(batch),
        str(reference),
        str(tmp_path / "DEDA report"),
        thresholds=DriftThresholds(price=0.2, category=0.3, state=0.4),
    )
    notebook: nbformat.NotebookNode = nbformat.read(paths.notebook_path, as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    outputs = [output for cell in code for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    assert sum("image/png" in output.get("data", {}) for output in outputs) == 3
    text = "\n".join(output.get("text", "") for output in outputs)
    assert sys.executable in text
    for content in before:
        assert hashlib.sha256(content).hexdigest() in text
    assert "drift_detected = False" in text
    assert "Объединение допустимо: 24 позиций" in text
    source = "\n".join(cell.source for cell in code)
    assert "check_combination(reference, batch)" in source
    assert "evaluate_drift(reference, batch, thresholds=thresholds)" in source

    metrics_text = paths.metrics_path.read_text(encoding="utf-8")
    metrics = json.loads(metrics_text)
    assert metrics == {
        "format_version": 1,
        "kind": "deda",
        "inputs": {
            role: {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "rows": 12}
            for role, path, content in zip(("reference", "batch"), (reference, batch), before)
        },
        "drift_detected": False,
        "metrics": [
            {"feature": "price", "measure": "KS D", "value": 0.0, "threshold": 0.2, "drift": False},
            {
                "feature": "product_category_name",
                "measure": "TVD",
                "value": 0.0,
                "threshold": 0.3,
                "drift": False,
            },
            {
                "feature": "customer_state",
                "measure": "TVD",
                "value": 0.0,
                "threshold": 0.4,
                "drift": False,
            },
        ],
        "thresholds": {"price": 0.2, "category": 0.3, "state": 0.4},
    }
    assert isinstance(metrics["format_version"], int)
    assert not isinstance(metrics["format_version"], bool)
    assert all(
        isinstance(value["rows"], int) and not isinstance(value["rows"], bool)
        for value in metrics["inputs"].values()
    )
    assert metrics["drift_detected"] is False
    for metric in metrics["metrics"]:
        assert isinstance(metric["value"], float)
        assert isinstance(metric["threshold"], float)
        assert metric["drift"] is False
        assert metric["drift"] == (metric["value"] >= metric["threshold"])
    assert metrics["drift_detected"] == any(metric["drift"] for metric in metrics["metrics"])
    assert metrics_text == json.dumps(metrics, ensure_ascii=False, allow_nan=False, indent=2) + "\n"

    html = paths.html_path.read_text(encoding="utf-8")
    parser = HTMLAssets()
    parser.feed(html)
    assert len(parser.images) == 3
    assert all(source.startswith("data:image/png;base64,") for source in parser.images)
    assert not parser.external_assets
    for expected in (
        "Эталон",
        "Батч",
        "SHA-256",
        "Размеры и периоды",
        "2018-01-02 05:00:00",
        "Пропуски",
        "в объединении",
        "KS D",
        "TVD",
        "Порог",
        "Отсутствие дрейфа",
        "Итог",
    ):
        assert expected in html
    assert "НЕ ПРОЙДЕНА" not in html
    assert {path.name for path in paths.html_path.parent.iterdir()} == {
        "report.html",
        "report.ipynb",
        "metrics.json",
    }
    assert [path.read_bytes() for path in (reference, batch)] == before


def test_comparison_categories_use_reference_ranking_and_keep_new_mass_separate(
    working_frame: ft.DataFrame,
) -> None:
    reference = working_frame["product_category_name"]
    batch = pd.Series(["category_11"] * 10 + ["brand_new"] * 5 + ["category_01"])
    shares = comparison_category_shares(reference, batch)
    assert list(shares.index) == [f"category_{index:02d}" for index in range(10)] + [
        "Прочие",
        "Новые категории",
    ]
    np.testing.assert_allclose(shares.sum(), [1, 1])
    assert shares.loc["Прочие", "reference"] == pytest.approx(2 / 12)
    assert shares.loc["Прочие", "batch"] == pytest.approx(10 / 16)
    assert not shares.loc["Новые категории", "reference"]
    assert shares.loc["Новые категории", "batch"] == pytest.approx(5 / 16)
    assert not shares.loc["category_00", "batch"]
    # Top ten must be recomputed when the current physical reference changes.
    expanded = pd.concat([reference, batch], ignore_index=True)
    next_shares = comparison_category_shares(expanded, reference)
    assert list(next_shares.index[:2]) == ["category_11", "brand_new"]
    assert "Новые категории" not in next_shares.index


@pytest.mark.parametrize("new", [False, True])
def test_comparison_categories_with_fewer_than_ten_values(new: bool) -> None:
    reference = pd.Series(["a", "b", "a"])
    batch = pd.Series(["c"] if new else ["a"])
    shares = comparison_category_shares(reference, batch)
    assert list(shares.index) == ["a", "b"] + (["Новые категории"] if new else [])
    np.testing.assert_allclose(shares.sum(), [1, 1])


@pytest.mark.parametrize("shifted", [False, True])
def test_comparison_histograms_have_common_bins_and_own_denominators(
    working_frame: ft.DataFrame, shifted: bool
) -> None:
    reference = working_frame.copy()
    batch = working_frame.iloc[:3].copy()
    if shifted:
        reference["price"] = np.geomspace(1, 1000, len(reference))
        batch["price"] = [0.01, 100, 10000]
    with _figure_axis(plot_price_comparison(reference, batch)) as axis:
        assert axis.get_xscale() == "log"
        first, second = _bar_containers(axis)
        assert len(first) == len(second) == 30
        for ref_bar, batch_bar in zip(first.patches, second.patches):
            assert ref_bar.get_x() == batch_bar.get_x()
            assert ref_bar.get_width() == batch_bar.get_width()
            assert ref_bar.get_width() > 0
        assert sum(patch.get_height() for patch in first.patches) == pytest.approx(1)
        assert sum(patch.get_height() for patch in second.patches) == pytest.approx(1)


def test_comparison_state_union_includes_batch_only_and_missing_states(
    working_frame: ft.DataFrame,
) -> None:
    batch = working_frame.iloc[:4].copy()
    batch["customer_state"] = ["AM", "AM", "SP", "SP"]
    with _figure_axis(plot_states_comparison(working_frame, batch)) as axis:
        assert [tick.get_text() for tick in axis.get_yticklabels()] == ["AM", "MG", "RJ", "SP"]
        reference_bars, batch_bars = _bar_containers(axis)
        np.testing.assert_allclose(
            [patch.get_width() for patch in reference_bars.patches], [0, 1 / 12, 3 / 12, 8 / 12]
        )
        np.testing.assert_allclose(
            [patch.get_width() for patch in batch_bars.patches], [0.5, 0, 0, 0.5]
        )
        assert not axis.get_xlim()[0]


def test_comparison_category_bars_preserve_group_shares(
    working_frame: ft.DataFrame, new_batch: ft.DataFrame
) -> None:
    new_batch.loc[:2, "product_category_name"] = "new"
    with _figure_axis(plot_categories_comparison(working_frame, new_batch)) as axis:
        reference_bars, batch_bars = _bar_containers(axis)
        assert len(reference_bars) == len(batch_bars) == 12
        assert sum(patch.get_width() for patch in reference_bars.patches) == pytest.approx(1)
        assert sum(patch.get_width() for patch in batch_bars.patches) == pytest.approx(1)
        assert not reference_bars.patches[-1].get_width()
        assert batch_bars.patches[-1].get_width() == pytest.approx(0.25)
