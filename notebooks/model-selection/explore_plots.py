"""Figures reused as executable code, rather than static images, in the report."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def screening_plot(validation):
    style()
    distances = ["uniform", "timestamp_3d", "timestamp_1d", "timestamp_0.5d",
                 "timestamp_0.125d", "category", "category_price_t0.4",
                 "category_price_t0.2", "category_price_t0.1",
                 "category_price_time_t0.4", "category_price_time_t0.2"]
    labels = ["Равномерно по покупкам", "Время · 3 суток", "Время · 1 сутки",
              "Время · 12 часов", "Время · 3 часа", "Категория",
              "Категория + цена · T=0.4", "Категория + цена · T=0.2",
              "Категория + цена · T=0.1", "Категория + цена + время · T=0.4",
              "Категория + цена + время · T=0.2 *"]
    models = ["random", "popularity", "personal_frequency", "svd_16", "svd_32",
              "svd_64", "svd_128", "user_knn_20", "user_knn_50"]
    table = validation.pivot_table(index="distance", columns="model", values="ndcg_at_k").loc[distances, models]
    fig, ax = plt.subplots(figsize=(14, 8.5))
    image = ax.imshow(table, vmin=0, vmax=.6, cmap="Blues", aspect="auto")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(len(models)), ["Random", "Популяр-\nность", "Личная\nчастота", "SVD16", "SVD32",
                                      "SVD64", "SVD128", "KNN20", "KNN50"])
    ax.tick_params(length=0, pad=9)
    for row in range(len(distances)):
        for column in range(len(models)):
            value = table.iloc[row, column]
            ax.text(column, row, f"{value:.3f}", ha="center", va="center",
                    color="white" if value > .35 else "#18324a", fontsize=10)
    ax.add_patch(Rectangle((-.5, len(distances) - 1.5), len(models), 1,
                           fill=False, edgecolor="#bc6737", linewidth=3))
    ax.get_yticklabels()[-1].set_color("#a34b1f")
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_title("01 · Первичный подбор: ранжировщики на 11 вариантах генератора",
                 loc="left", pad=24, fontsize=14, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=.027, pad=.025, label="Validation NDCG@10 · выше лучше")
    fig.subplots_adjust(left=.31, bottom=.20, right=.92, top=.89)
    fig.text(.03, .055, "Среднее по 3 seed; 1000 пользователей; полный каталог 14 314 товаров; split 70/15/15.\n"
             "Сравнивайте модели внутри строки: разные генераторы задают разные задачи.\n"
             "* Не проходит gate: средняя доля самого частого товара в train >20%.", fontsize=11)
    return fig


def refinement_plot(refinement):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    for index, (ax, temperature) in enumerate(zip(axes, (.1, .15, .2))):
        data = refinement[refinement.distance.eq(f"category_price_t{temperature}")]
        for seed, group in data[data.model.str.startswith("svd_")].groupby("seed"):
            ranks = group.model.str.removeprefix("svd_").astype(int)
            ax.plot(ranks, group.ndcg_at_k, color="#92b4ce", linewidth=1, alpha=.8)
        means = data.groupby("model").ndcg_at_k.mean()
        ax.plot([128, 256, 384], [means[f"svd_{n}"] for n in (128, 256, 384)],
                "o-", color="#125580", linewidth=2.5, label="SVD · среднее")
        ax.axhline(means["personal_frequency"], color="#bc6737", linestyle="--",
                   label="Личная частота · среднее")
        ax.set_title(f"({chr(97 + index)}) Температура {temperature}", loc="left")
        ax.set_xticks([128, 256, 384])
        ax.set_xlabel("Число компонент SVD")
        ax.set_ylim(.32, .53)
        ax.grid(axis="y", alpha=.16)
    axes[0].set_ylabel("Validation NDCG@10 ↑")
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)
    fig.suptitle("02 · SVD256 при T=0.1 — лучшая средняя метрика в уточнённой сетке",
                 x=.07, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=.07, right=.98, bottom=.25, top=.80, wspace=.14)
    fig.text(.07, .065, "(a–c) Тот же первый батч, 1000 пользователей, 3 seed; тонкие линии — отдельные запуски.\n"
             "Подбор только по validation; личная частота — контроль запоминания покупок.\n"
             "Температура меняет датасет, поэтому сравнение моделей — внутри панели.", fontsize=11)
    return fig


def final_plot(metrics):
    style()
    data = metrics[metrics.split.eq("test") & metrics.distance.eq("category_price_t0.1")]
    models = ["svd_256", "personal_frequency", "svd_32", "popularity", "random"]
    labels = ["SVD256", "Личная частота", "SVD32", "Популярность", "Random"]
    colors = ["#125580", "#bc6737", "#7994a4", "#7994a4", "#7994a4"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, metric, title in zip(axes, ("ndcg_at_k", "recall_at_k"), ("(a) NDCG@10 →", "(b) Recall@10 →")):
        stats = data.groupby("model")[metric].agg(["mean", "min", "max"]).loc[models]
        y = np.arange(len(models))
        ax.barh(y, stats["mean"], color=colors, height=.52)
        ax.errorbar(stats["mean"], y,
                    xerr=np.vstack([stats["mean"] - stats["min"], stats["max"] - stats["mean"]]),
                    fmt="none", ecolor="#102b3c", capsize=4, linewidth=1.5)
        for position, value in enumerate(stats["mean"]):
            ax.text(stats["max"].iloc[position] + .012, position, f"{value:.4f}", va="center")
        ax.set_xlim(0, .60)
        ax.set_yticks(y, labels)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("03 · Независимый test: +5.0% NDCG относительно личной частоты",
                 x=.10, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=.16, right=.98, bottom=.27, top=.80, wspace=.25)
    fig.text(.04, .06, "(a–b) Одинаковые датасеты: категория + log-цена, T=0.1; 2000 пользователей × 3 новых seed.\n"
             "Столбцы — среднее, усы — min–max по seed (не доверительный интервал); каталог 14 314 товаров.\n"
             "Повторные покупки разрешены. Результат относится к синтетическим однокатегорийным интересам.", fontsize=11)
    return fig


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    output = here.parents[1] / "visualizations/model-selection"
    output.mkdir(parents=True, exist_ok=True)
    for filename, function, source in (
        ("01-screening", screening_plot, "validation.csv"),
        ("02-refinement", refinement_plot, "refinement.csv"),
        ("03-final-test", final_plot, "final_metrics.csv"),
    ):
        figure = function(pd.read_csv(here / source))
        figure.savefig(output / f"{filename}.png", dpi=160)
        plt.close(figure)
