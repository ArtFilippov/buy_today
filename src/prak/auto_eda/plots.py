"""Deterministic, position-weighted distributions for EDA and DEDA."""

import textwrap

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 120,
}
_COLOR = "#287a9f"
_BATCH_COLOR = "#c66b27"


def category_shares(values: pd.Series) -> pd.Series:
    """Top ten by count, ties alphabetically; the remaining mass is 'Прочие'."""
    counts = values.value_counts().sort_index().sort_values(ascending=False, kind="stable")
    shares = counts.iloc[:10] / len(values)
    if len(counts) > 10:
        shares.loc["Прочие"] = counts.iloc[10:].sum() / len(values)
    return shares


def _price_bins(low: float, high: float) -> np.ndarray:
    if low == high:
        # A single value still needs a visible bin and a nonzero log-axis span.
        low = max(low / 1.1, np.nextafter(0.0, 1.0))
        high = min(high * 1.1, np.finfo(float).max)
    return np.geomspace(low, high, 31)


def plot_price(frame: pd.DataFrame) -> Figure:
    """One log-x histogram: bin heights are fractions, not probability density."""
    prices = frame["price"].to_numpy()
    bins = _price_bins(float(prices.min()), float(prices.max()))
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.hist(
            prices, bins=bins, weights=np.full(len(prices), 1 / len(prices)),
            color=_COLOR, edgecolor="white", linewidth=0.5,
        )
        ax.set_xscale("log")
        ax.set(
            title=f"Распределение цены · {len(frame):,} позиций",
            xlabel="Цена, BRL (логарифмическая шкала)",
            ylabel="Доля позиций в интервале",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        return fig


def _plot_shares(shares: pd.Series, title: str) -> Figure:
    labels = [textwrap.fill(str(label).replace("_", " "), 32) for label in shares.index]
    with plt.rc_context(_STYLE):
        height = max(4.5, 1.4 + 0.44 * sum(label.count("\n") + 1 for label in labels))
        fig, ax = plt.subplots(figsize=(10, height), layout="constrained")
        bars = ax.barh(range(len(shares)), shares, color=_COLOR)
        ax.set_yticks(range(len(shares)), labels)
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[f"{value:.2%}" for value in shares], padding=5)
        ax.set(title=title, xlabel="Доля позиций заказа")
        ax.set_xlim(0, min(1.15, float(shares.max()) * 1.22))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
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


def comparison_category_shares(reference: pd.Series, batch: pd.Series) -> pd.DataFrame:
    """Reference top ten, known remainder and batch-only categories; own denominators."""
    counts = reference.value_counts().sort_index().sort_values(ascending=False, kind="stable")
    top = counts.index[:10]
    reference_shares = counts.loc[top] / len(reference)
    batch_counts = batch.value_counts()
    batch_shares = batch_counts.reindex(top, fill_value=0) / len(batch)
    if len(counts) > 10:
        reference_shares.loc["Прочие"] = counts.iloc[10:].sum() / len(reference)
        batch_shares.loc["Прочие"] = batch_counts.reindex(counts.index[10:], fill_value=0).sum() / len(batch)
    new_count = batch_counts.loc[~batch_counts.index.isin(counts.index)].sum()
    if new_count:
        reference_shares.loc["Новые категории"] = 0.0
        batch_shares.loc["Новые категории"] = new_count / len(batch)
    return pd.DataFrame({"reference": reference_shares, "batch": batch_shares})


def plot_price_comparison(reference: pd.DataFrame, batch: pd.DataFrame) -> Figure:
    """Two overlaid histograms with shared bins and separate unit-mass weights."""
    bins = _price_bins(
        float(min(reference["price"].min(), batch["price"].min())),
        float(max(reference["price"].max(), batch["price"].max())),
    )
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        for frame, label, color, linestyle in (
            (reference, "Эталон", _COLOR, "-"), (batch, "Батч", _BATCH_COLOR, "--"),
        ):
            heights, _, _ = ax.hist(
                frame["price"], bins=bins, weights=np.full(len(frame), 1 / len(frame)),
                color=color, alpha=0.2,
            )
            ax.stairs(
                heights, bins, color=color, linewidth=1.6, linestyle=linestyle,
                label=f"{label} · {len(frame):,} позиций",
            )
        ax.set_xscale("log")
        ax.set(
            title="Сравнение цен · эталон до добавления батча",
            xlabel="Цена, BRL (логарифмическая шкала)",
            ylabel="Доля позиций в интервале",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        return fig


def _plot_comparison_shares(
    shares: pd.DataFrame, title: str, reference_size: int, batch_size: int,
) -> Figure:
    labels = [textwrap.fill(str(label).replace("_", " "), 32) for label in shares.index]
    with plt.rc_context(_STYLE):
        height = max(5, 2 + 0.65 * sum(label.count("\n") + 1 for label in labels))
        fig, ax = plt.subplots(figsize=(11, height), layout="constrained")
        positions = np.arange(len(shares))
        for column, offset, label, color in (
            ("reference", -0.18, f"Эталон · {reference_size:,} позиций", _COLOR),
            ("batch", 0.18, f"Батч · {batch_size:,} позиций", _BATCH_COLOR),
        ):
            bars = ax.barh(positions + offset, shares[column], height=0.32, color=color, label=label)
            ax.bar_label(bars, labels=[f"{value:.2%}" for value in shares[column]], padding=5)
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_title(title, pad=50)
        ax.set_xlabel("Доля позиций заказа в соответствующем входе")
        ax.set_xlim(0, min(1.15, float(shares.to_numpy().max()) * 1.25))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncols=2)
        return fig


def plot_categories_comparison(reference: pd.DataFrame, batch: pd.DataFrame) -> Figure:
    return _plot_comparison_shares(
        comparison_category_shares(reference["product_category_name"], batch["product_category_name"]),
        "Категории товаров · top-10 текущего эталона", len(reference), len(batch),
    )


def plot_states_comparison(reference: pd.DataFrame, batch: pd.DataFrame) -> Figure:
    shares = pd.DataFrame({
        "reference": reference["customer_state"].value_counts() / len(reference),
        "batch": batch["customer_state"].value_counts() / len(batch),
    }).fillna(0).sort_index()
    return _plot_comparison_shares(shares, "Штаты покупателей · по позициям заказа", len(reference), len(batch))
