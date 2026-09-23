"""Assemble and execute the report from inspected plotting code and prepared CSVs."""

import inspect
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from explore_plots import style, screening_plot, refinement_plot, final_plot

HERE = Path(__file__).resolve().parent


def main():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md("**Ответ:** CategoryPriceDistance, температура **0.1**, SVDRanker(**256** компонент, "
           "7 итераций, seed 42). На трёх новых реализациях по 2000 пользователей test "
           "NDCG@10 ≈ **0.494**, Recall@10 ≈ **0.372**. Это модель узких однокатегорийных "
           "интересов с повторными покупками; качество на реальных людях не измерено."),
        md("**Подготовка:** исходный полный `data/olist-stream/batches/batch_000.csv`, "
           "41 387 позиций, 14 314 товаров. Существующая очистка Olist описана в "
           "`../../datasets/model-selection/dataset_evidence.md`. Из корня: "
           "`uv run --locked python notebooks/model-selection/prepare_data.py`, затем "
           "`refine.py` и `finalize.py` тем же способом. Дорогая генерация и обучение "
           "выполняются отдельно; этот notebook читает только подготовленные CSV."),
        code("from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\n"
             "import matplotlib.pyplot as plt\nfrom matplotlib.patches import Rectangle\n"
             "from IPython.display import display\n"
             "HERE = Path.cwd()\n"
             "required = ['validation.csv', 'quality.csv', 'refinement.csv', 'final_metrics.csv', "
             "'final_quality.csv', 'test_per_user.csv', 'novelty.csv', 'selection.json']\n"
             "missing = [name for name in required if not (HERE / name).exists()]\n"
             "if missing:\n    raise FileNotFoundError(f'{missing}: run prepare_data.py, refine.py, finalize.py first')\n"
             "validation = pd.read_csv(HERE / 'validation.csv')\n"
             "quality = pd.read_csv(HERE / 'quality.csv')\n"
             "refinement = pd.read_csv(HERE / 'refinement.csv')\n"
             "metrics = pd.read_csv(HERE / 'final_metrics.csv')\n"
             "final_quality = pd.read_csv(HERE / 'final_quality.csv')\n"
             "per_user = pd.read_csv(HERE / 'test_per_user.csv')\n"
             "novelty = pd.read_csv(HERE / 'novelty.csv')\n"
             "selection = json.loads((HERE / 'selection.json').read_text())\n"
             "assert len(validation) == 297 and len(refinement) == 36 and len(metrics) == 33\n"
             "assert not validation.duplicated(['distance', 'seed', 'model']).any()\n"
             "assert not refinement.duplicated(['distance', 'seed', 'model']).any()\n"
             "assert not metrics.duplicated(['distance', 'seed', 'model', 'split']).any()\n"
             "for frame in (validation, refinement, metrics, quality, final_quality):\n"
             "    assert not frame.isna().any().any()\n"
             "    assert np.isfinite(frame.select_dtypes('number')).all().all()\n"
             + inspect.getsource(style)),
        md("**Протокол:** 70/15/15 iid событий на пользователя; train-only user/item counts; "
           "полный каталог и бинарная relevance по уникальным held-out товарам. "
           "Метрики — macro-average по пользователям. Повторы разрешены между split. "
           "Одинаковые anchors между генераторами при одном seed, но разные задачи оценки."),
        code("quality['passes'] = ((quality.unique_train_median >= 30) & "
             "(quality.unique_train_p10 >= 10) & (quality.mean_top_product_share <= .2) & "
             "(quality.category_tvd <= .1))\n"
             "display(quality.groupby('distance').agg("
             "median_unique=('unique_train_median', 'mean'), p10_unique=('unique_train_p10', 'mean'), "
             "top_product_share=('mean_top_product_share', 'mean'), "
             "catalog_coverage=('train_catalog_coverage', 'mean'), "
             "category_tvd=('category_tvd', 'mean'), all_seeds_pass=('passes', 'all')).round(4))"),
        code("# Figure 01: screening\n" + inspect.getsource(screening_plot) + "\nscreening_plot(validation);\nplt.show()"),
        md("**Следующая проверка:** SVD128 на T=.1 уступил личной частоте. Поэтому до открытия "
           "test увеличили размерность и добавили T=.15. Пороги разнообразия были заданы до "
           "первичного screening; они определяют учебную пригодность, не реалистичность."),
        code("# Figure 02: refinement\n" + inspect.getsource(refinement_plot) + "\nrefinement_plot(refinement);\nplt.show()"),
        code("chosen = refinement[refinement.distance.eq('category_price_t0.1')]\n"
             "paired_validation = chosen.pivot(index='seed', columns='model', values='ndcg_at_k')\n"
             "display(paired_validation.assign(svd256_minus_frequency="
             "paired_validation.svd_256 - paired_validation.personal_frequency).round(5))\n"
             "assert ((paired_validation.svd_256 - paired_validation.personal_frequency) > 0).all()\n"
             "display(selection)"),
        md("**Фиксация:** параметры сохранены в `selection.json` до финальной оценки. "
           "Новые seeds — 73/211/907, по 2000 пользователей, каждый с собственными train/test. "
           "Модель заново обучается только на train; параметры по test не меняются."),
        code("# Figure 03: final-test\n" + inspect.getsource(final_plot) + "\nfinal_plot(metrics);\nplt.show()"),
        code("test = metrics[metrics.split.eq('test')]\n"
             "display(test.groupby(['distance', 'model'])[['ndcg_at_k', 'recall_at_k']].agg(['mean', 'min', 'max']).round(5))\n"
             "display(final_quality.round(4))\n"
             "assert (final_quality.unique_train_median >= 30).all()\n"
             "assert (final_quality.unique_train_p10 >= 10).all()\n"
             "assert (final_quality.mean_top_product_share <= .2).all()\n"
             "assert (final_quality.category_tvd <= .1).all()"),
        md("**Контроль выигрыша:** парный bootstrap по пользователям отдельно внутри каждого "
           "seed, затем равновесное среднее по seeds. Это условный 95% интервал для данных "
           "трёх фиксированных обученных моделей, не интервал обобщения на Olist или на все seeds."),
        code("paired = per_user.pivot(index=['seed', 'user_id'], columns='model', values='ndcg_at_k')\n"
             "delta = paired.svd_256 - paired.personal_frequency\n"
             "rng = np.random.default_rng(20260923)\n"
             "resampled = np.zeros(2000)\n"
             "for seed, values in delta.groupby(level='seed'):\n"
             "    values = values.to_numpy()\n"
             "    resampled += values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1) / 3\n"
             "display(pd.Series({'delta_ndcg': delta.mean(), 'ci_low': np.quantile(resampled, .025), "
             "'ci_high': np.quantile(resampled, .975)}).round(5))\n"
             "display(delta.groupby(level='seed').mean().rename('paired_delta_ndcg').round(5))"),
        md("**Повторение или новые товары?** Дополнительный Recall@10 считается только по "
           "held-out товарам, которых пользователь не покупал в train. Пользователи без "
           "таких товаров исключены только из этой дополнительной метрики."),
        code("display(novelty.groupby('model').agg("
             "mean_relevant=('n_relevant', 'mean'), mean_novel_relevant=('n_novel_relevant', 'mean'), "
             "mean_novel_hits=('novel_hits', 'mean'), mean_seen_hits=('seen_hits', 'mean'), "
             "novel_recall=('novel_recall', 'mean'), users_with_novel=('novel_recall', 'count')).round(5))"),
        md("**Итог:** связка даёт стабильный выигрыш над личной частотой и проходит выбранные "
           "пороги разнообразия. Однако около 99.9% покупок в категории якоря; большинство "
           "попаданий — повторы. Novel recall ≈6.0%. Для многокатегорийных корзин, поздних "
           "батчей и реальных пользователей нужна отдельная проверка. Разница с текущим "
           "pipeline (время 1d + SVD32) смешивает изменение данных и модели."),
        md("**Артефакты:** `../../models/model-selection/distance.joblib`; "
           "`../../models/model-selection/seed_73/ranking/`; "
           "`../../data/model-selection/seed_73/`. Команды использования — "
           "`../../docs/model-selection.md`. Все три рисунка приведены в порядке исследования; "
           "контекст — `../../visualizations/model-selection/data_context.md`."),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    })
    nbformat.validate(notebook)
    NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.validate(notebook)
    figures = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.startswith("# Figure")]
    assert len(figures) == 3
    assert all(any("image/png" in output.get("data", {}) for output in cell.outputs) for cell in figures)
    nbformat.write(notebook, HERE / "report.ipynb")
    html, _ = HTMLExporter().from_notebook_node(notebook)
    (HERE / "report.html").write_text(html, encoding="utf-8")
    print("Saved executed report.ipynb and report.html; all 3 figures have inline output.")


if __name__ == "__main__":
    main()
