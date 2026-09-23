"""Deterministic, position-weighted distributions for EDA and DEDA."""

from __future__ import annotations

import textwrap
from sys import float_info
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from matplotlib.typing import RcKeyType
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from buy_today.plot_style import REPORT_STYLE

_STYLE: dict[RcKeyType, Any] = {
    **REPORT_STYLE,
    "axes.titlesize": 14,
}
_COLOR = "#287a9f"
_BATCH_COLOR = "#c66b27"
_TOP_CATEGORIES = 10


def category_shares(values: pd.Series[Any]) -> pd.Series[float]:
    """Top ten by count, ties alphabetically; the remaining mass is 'Прочие'.

    Args:
        values (pd.Series[Any]): Nonmissing category values for all positions.

    Returns:
        pd.Series[float]: Position-weighted category proportions summing to one.
    """
    counts = values.value_counts().sort_index().sort_values(ascending=False, kind="stable")
    shares = counts.iloc[:_TOP_CATEGORIES] / len(values)
    if len(counts) > _TOP_CATEGORIES:
        shares.loc["Прочие"] = counts.iloc[_TOP_CATEGORIES:].sum() / len(values)
    return shares


def _price_bins(low: float, high: float) -> NDArray[np.float64]:
    if low == high:
        # A single value still needs a visible bin and a nonzero log-axis span.
        low = max(low / 1.1, np.nextafter(0.0, 1.0))
        high = min(high * 1.1, float_info.max)
    return np.geomspace(low, high, 31)


def plot_price(frame: pd.DataFrame) -> Figure:
    """One log-x histogram: bin heights are fractions, not probability density.

    Args:
        frame (pd.DataFrame): Validated dataset containing positive finite prices.

    Returns:
        Figure: Histogram with logarithmic price axis and unit total mass.
    """
    prices = frame["price"].to_numpy()
    bins = _price_bins(float(np.min(prices)), float(np.max(prices)))
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        _ = ax.hist(
            prices,
            bins=np.ndarray.tolist(bins),
            weights=np.full(len(prices), 1 / len(prices)),
            color=_COLOR,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xscale("log")
        _ = ax.set(
            title=f"Распределение цены · {len(frame):,} позиций",
            xlabel="Цена, BRL (логарифмическая шкала)",
            ylabel="Доля позиций в интервале",
        )
        Axis.set_major_formatter(ax.yaxis, PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        return fig


def _plot_shares(shares: pd.Series[float], title: str) -> Figure:
    labels = [textwrap.fill(str(label).replace("_", " "), 32) for label in shares.index]
    with plt.rc_context(_STYLE):
        height = max(4.5, 1.4 + 0.44 * sum(label.count("\n") + 1 for label in labels))
        fig, ax = plt.subplots(figsize=(10, height), layout="constrained")
        bars = ax.barh(range(len(shares)), shares, color=_COLOR)
        _ = ax.set_yticks(range(len(shares)), labels)
        ax.invert_yaxis()
        _ = ax.bar_label(bars, labels=[f"{value:.2%}" for value in shares], padding=5)
        _ = ax.set(title=title, xlabel="Доля позиций заказа")
        _ = ax.set_xlim(0, min(1.15, float(shares.max()) * 1.22))
        Axis.set_major_formatter(ax.xaxis, PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.2)
        return fig


def plot_categories(frame: pd.DataFrame) -> Figure:
    return _plot_shares(
        category_shares(frame["product_category_name"]),
        f"Категории товаров · top-10 · {len(frame):,} позиций",
    )


def plot_states(frame: pd.DataFrame) -> Figure:
    shares = frame["customer_state"].value_counts().sort_index() / len(frame)
    return _plot_shares(shares, f"Штаты покупателей · {len(frame):,} позиций")


def comparison_category_shares(reference: pd.Series[Any], batch: pd.Series[Any]) -> pd.DataFrame:
    """Reference top ten, known remainder and batch-only categories; own denominators.

    Args:
        reference (pd.Series[Any]): Nonmissing reference category values.
        batch (pd.Series[Any]): Nonmissing incoming category values.

    Returns:
        pd.DataFrame: Reference and batch shares with separate denominators.
    """
    counts = reference.value_counts().sort_index().sort_values(ascending=False, kind="stable")
    top = counts.index[:_TOP_CATEGORIES]
    reference_shares = counts.loc[top] / len(reference)
    batch_counts = batch.value_counts()
    batch_shares = pd.Series.reindex(batch_counts, top, fill_value=0) / len(batch)
    if len(counts) > _TOP_CATEGORIES:
        reference_shares.loc["Прочие"] = counts.iloc[_TOP_CATEGORIES:].sum() / len(reference)
        batch_shares.loc["Прочие"] = pd.Series.reindex(
            batch_counts, counts.index[_TOP_CATEGORIES:], fill_value=0
        ).sum() / len(batch)
    if new_count := batch_counts.loc[~pd.Index.isin(batch_counts.index, counts.index)].sum():
        reference_shares.loc["Новые категории"] = 0.0
        batch_shares.loc["Новые категории"] = new_count / len(batch)
    return pd.DataFrame({"reference": reference_shares, "batch": batch_shares})


def plot_price_comparison(reference: pd.DataFrame, batch: pd.DataFrame) -> Figure:
    """Two overlaid histograms with shared bins and separate unit-mass weights.

    Args:
        reference (pd.DataFrame): Validated reference dataset.
        batch (pd.DataFrame): Validated incoming dataset.

    Returns:
        Figure: Overlaid histograms spanning the complete shared price range.
    """
    bins = _price_bins(
        float(min(reference["price"].min(), batch["price"].min())),
        float(max(reference["price"].max(), batch["price"].max())),
    )
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        for frame, label, color, linestyle in (
            (reference, "Эталон", _COLOR, "-"),
            (batch, "Батч", _BATCH_COLOR, "--"),
        ):
            heights, _, _ = ax.hist(
                frame["price"],
                bins=np.ndarray.tolist(bins),
                weights=np.full(len(frame), 1 / len(frame)),
                color=color,
                alpha=0.2,
            )
            _ = ax.stairs(
                heights,
                bins,
                color=color,
                linewidth=1.6,
                linestyle=linestyle,
                label=f"{label} · {len(frame):,} позиций",
            )
        ax.set_xscale("log")
        _ = ax.set(
            title="Сравнение цен · эталон до добавления батча",
            xlabel="Цена, BRL (логарифмическая шкала)",
            ylabel="Доля позиций в интервале",
        )
        Axis.set_major_formatter(ax.yaxis, PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        _ = ax.legend()
        return fig


def _comparison_bars(ax: Axes, shares: pd.DataFrame, sizes: tuple[int, int]) -> None:
    positions = np.arange(len(shares))
    for column, offset, label, color in (
        ("reference", -0.18, f"Эталон · {sizes[0]:,} позиций", _COLOR),
        ("batch", 0.18, f"Батч · {sizes[1]:,} позиций", _BATCH_COLOR),
    ):
        bars = ax.barh(positions + offset, shares[column], height=0.32, color=color, label=label)
        _ = ax.bar_label(bars, labels=[f"{value:.2%}" for value in shares[column]], padding=5)


def _plot_comparison_shares(
    shares: pd.DataFrame,
    title: str,
    reference_size: int,
    batch_size: int,
) -> Figure:
    labels = [textwrap.fill(str(label).replace("_", " "), 32) for label in shares.index]
    with plt.rc_context(_STYLE):
        height = max(5, 2 + 0.65 * sum(label.count("\n") + 1 for label in labels))
        fig, ax = plt.subplots(figsize=(11, height), layout="constrained")
        _comparison_bars(ax, shares, (reference_size, batch_size))
        _ = ax.set_yticks(np.arange(len(shares)), labels)
        ax.invert_yaxis()
        _ = ax.set_title(title, pad=50)
        _ = ax.set_xlabel("Доля позиций заказа в соответствующем входе")
        _ = ax.set_xlim(0, min(1.15, float(shares.to_numpy().max()) * 1.25))
        Axis.set_major_formatter(ax.xaxis, PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.2)
        _ = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncols=2)
        return fig


def plot_categories_comparison(reference: pd.DataFrame, batch: pd.DataFrame) -> Figure:
    return _plot_comparison_shares(
        comparison_category_shares(
            reference["product_category_name"], batch["product_category_name"]
        ),
        "Категории товаров · top-10 текущего эталона",
        len(reference),
        len(batch),
    )


def plot_states_comparison(reference: pd.DataFrame, batch: pd.DataFrame) -> Figure:
    shares = (
        pd.DataFrame(
            {
                "reference": reference["customer_state"].value_counts() / len(reference),
                "batch": batch["customer_state"].value_counts() / len(batch),
            }
        )
        .fillna(0)
        .sort_index()
    )
    return _plot_comparison_shares(
        shares, "Штаты покупателей · по позициям заказа", len(reference), len(batch)
    )
