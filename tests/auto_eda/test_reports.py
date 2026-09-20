import hashlib
from html.parser import HTMLParser
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbclient.exceptions import CellExecutionError
import numpy as np
import pandas as pd
import pytest

from prak.auto_eda import report_dataset, report_drift
from prak.auto_eda.notebook import execute_report
from prak.auto_eda.plots import (
    category_shares, comparison_category_shares, plot_categories_comparison,
    plot_price, plot_price_comparison, plot_states, plot_states_comparison,
)


class HTMLAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.external_assets = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img":
            self.images.append(attrs.get("src", ""))
        if tag in {"img", "script", "iframe", "link"}:
            source = attrs.get("src") or attrs.get("href", "")
            if source and not source.startswith("data:"):
                self.external_assets.append(source)


def test_eda_executes_in_project_python_and_exports_standalone_html(
    working_frame, write_dataset, tmp_path,
):
    dataset = write_dataset(working_frame)
    paths = report_dataset(str(dataset), str(tmp_path / "EDA report"))
    notebook = nbformat.read(paths.notebook_path, as_version=4)
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

    html = paths.html_path.read_text(encoding="utf-8")
    parser = HTMLAssets()
    parser.feed(html)
    assert len(parser.images) == 3
    assert all(source.startswith("data:image/png;base64,") for source in parser.images)
    assert parser.external_assets == []
    for expected in ["SHA-256", "Размеры и период", "2018-01-01 05:00:00", "Пропуски", "Ожидается", "Итог"]:
        assert expected in html
    assert set(path.name for path in paths.html_path.parent.iterdir()) == {"report.html", "report.ipynb"}


def test_invalid_data_stops_notebook_before_plots(working_frame, write_dataset, tmp_path):
    working_frame.loc[0, "price"] = 0
    output = tmp_path / "invalid report"
    with pytest.raises(CellExecutionError, match="Конечная положительная цена"):
        report_dataset(write_dataset(working_frame), output)
    notebook = nbformat.read(output / "report.ipynb", as_version=4)
    outputs = [output for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs]
    assert any(output.output_type == "error" for output in outputs)
    assert not any("image/png" in output.get("data", {}) for output in outputs)
    assert not (output / "report.html").exists()


def test_kernel_technical_error_propagates(tmp_path):
    with pytest.raises(CellExecutionError, match="rendering failed"):
        execute_report(
            [nbformat.v4.new_code_cell("raise RuntimeError('rendering failed')")],
            tmp_path, title="Failure",
        )
    assert not (tmp_path / "report.html").exists()


def test_top_ten_ties_are_alphabetical_and_preserve_full_mass(working_frame):
    shares = category_shares(working_frame["product_category_name"])
    assert list(shares.index) == [f"category_{index:02d}" for index in range(10)] + ["Прочие"]
    assert shares.loc["Прочие"] == pytest.approx(2 / 12)
    assert shares.sum() == pytest.approx(1)


@pytest.mark.parametrize("prices", [[100.0] * 12, [0.01, 1, 2, 3, 5, 10, 20, 50, 100, 500, 1000, 10000]])
def test_price_plot_has_one_axis_log_scale_and_unit_mass(working_frame, prices):
    working_frame["price"] = prices
    figure = plot_price(working_frame)
    try:
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert axis.get_xscale() == "log"
        assert sum(bar.get_height() for bar in axis.patches) == pytest.approx(1)
        assert all(bar.get_width() > 0 for bar in axis.patches)
    finally:
        plt.close(figure)


def test_state_shares_count_positions_not_unique_orders(working_frame):
    figure = plot_states(working_frame)
    try:
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert [tick.get_text() for tick in axis.get_yticklabels()] == ["MG", "RJ", "SP"]
        np.testing.assert_allclose([bar.get_width() for bar in axis.patches], [1 / 12, 3 / 12, 8 / 12])
        assert axis.get_xlim()[0] == 0
    finally:
        plt.close(figure)


def test_deda_executes_comparison_and_exports_standalone_html(
    working_frame, new_batch, write_dataset, tmp_path,
):
    reference = write_dataset(working_frame, "reference ' data.csv")
    batch = write_dataset(new_batch)
    before = [path.read_bytes() for path in (reference, batch)]
    paths = report_drift(str(batch), str(reference), str(tmp_path / "DEDA report"))
    notebook = nbformat.read(paths.notebook_path, as_version=4)
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

    html = paths.html_path.read_text(encoding="utf-8")
    parser = HTMLAssets()
    parser.feed(html)
    assert len(parser.images) == 3
    assert all(source.startswith("data:image/png;base64,") for source in parser.images)
    assert parser.external_assets == []
    for expected in [
        "Эталон", "Батч", "SHA-256", "Размеры и периоды", "2018-01-02 05:00:00",
        "Пропуски", "в объединении", "KS D", "TVD", "Порог", "Отсутствие дрейфа", "Итог",
    ]:
        assert expected in html
    assert "НЕ ПРОЙДЕНА" not in html
    assert {path.name for path in paths.html_path.parent.iterdir()} == {"report.html", "report.ipynb"}
    assert [path.read_bytes() for path in (reference, batch)] == before


def test_comparison_categories_use_reference_ranking_and_keep_new_mass_separate(working_frame):
    reference = working_frame["product_category_name"]
    batch = pd.Series(["category_11"] * 10 + ["brand_new"] * 5 + ["category_01"])
    shares = comparison_category_shares(reference, batch)
    assert list(shares.index) == [f"category_{index:02d}" for index in range(10)] + [
        "Прочие", "Новые категории",
    ]
    np.testing.assert_allclose(shares.sum(), [1, 1])
    assert shares.loc["Прочие", "reference"] == pytest.approx(2 / 12)
    assert shares.loc["Прочие", "batch"] == pytest.approx(10 / 16)
    assert shares.loc["Новые категории", "reference"] == 0
    assert shares.loc["Новые категории", "batch"] == pytest.approx(5 / 16)
    assert shares.loc["category_00", "batch"] == 0
    # Top ten must be recomputed when the current physical reference changes.
    expanded = pd.concat([reference, batch], ignore_index=True)
    next_shares = comparison_category_shares(expanded, reference)
    assert list(next_shares.index[:2]) == ["category_11", "brand_new"]
    assert "Новые категории" not in next_shares.index


@pytest.mark.parametrize("new", [False, True])
def test_comparison_categories_with_fewer_than_ten_values(new):
    reference = pd.Series(["a", "b", "a"])
    batch = pd.Series(["c"] if new else ["a"])
    shares = comparison_category_shares(reference, batch)
    assert list(shares.index) == ["a", "b"] + (["Новые категории"] if new else [])
    np.testing.assert_allclose(shares.sum(), [1, 1])


@pytest.mark.parametrize("shifted", [False, True])
def test_comparison_histograms_have_common_bins_and_own_denominators(working_frame, shifted):
    reference = working_frame.copy()
    batch = working_frame.iloc[:3].copy()
    if shifted:
        reference["price"] = np.geomspace(1, 1000, len(reference))
        batch["price"] = [0.01, 100, 10000]
    figure = plot_price_comparison(reference, batch)
    try:
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert axis.get_xscale() == "log"
        first, second = axis.containers
        assert len(first) == len(second) == 30
        for ref_bar, batch_bar in zip(first, second):
            assert ref_bar.get_x() == batch_bar.get_x()
            assert ref_bar.get_width() == batch_bar.get_width()
            assert ref_bar.get_width() > 0
        assert sum(bar.get_height() for bar in first) == pytest.approx(1)
        assert sum(bar.get_height() for bar in second) == pytest.approx(1)
    finally:
        plt.close(figure)


def test_comparison_state_union_includes_batch_only_and_missing_states(working_frame):
    batch = working_frame.iloc[:4].copy()
    batch["customer_state"] = ["AM", "AM", "SP", "SP"]
    figure = plot_states_comparison(working_frame, batch)
    try:
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert [tick.get_text() for tick in axis.get_yticklabels()] == ["AM", "MG", "RJ", "SP"]
        reference_bars, batch_bars = axis.containers
        np.testing.assert_allclose([bar.get_width() for bar in reference_bars], [0, 1 / 12, 3 / 12, 8 / 12])
        np.testing.assert_allclose([bar.get_width() for bar in batch_bars], [0.5, 0, 0, 0.5])
        assert axis.get_xlim()[0] == 0
    finally:
        plt.close(figure)


def test_comparison_category_bars_preserve_group_shares(working_frame, new_batch):
    new_batch.loc[:2, "product_category_name"] = "new"
    figure = plot_categories_comparison(working_frame, new_batch)
    try:
        assert len(figure.axes) == 1
        reference_bars, batch_bars = figure.axes[0].containers
        assert len(reference_bars) == len(batch_bars) == 12
        assert sum(bar.get_width() for bar in reference_bars) == pytest.approx(1)
        assert sum(bar.get_width() for bar in batch_bars) == pytest.approx(1)
        assert reference_bars[-1].get_width() == 0
        assert batch_bars[-1].get_width() == pytest.approx(0.25)
    finally:
        plt.close(figure)
