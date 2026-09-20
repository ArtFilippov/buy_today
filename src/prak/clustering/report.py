"""Key alignment and executable reports for already saved clustering snapshots."""

from pathlib import Path
from textwrap import dedent

import nbformat
import numpy as np
import pandas as pd

from prak.auto_eda.notebook import ReportPaths, execute_report
from prak.clustering.evaluation import validate_evaluation_parameters
from prak.clustering.training import check_labels
from prak.schema import CSV_DTYPES, ROW_KEY


def read_labels(path: Path | str) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype={
        **{column: CSV_DTYPES[column] for column in ROW_KEY}, "cluster_id": "int64",
    })
    check_labels(labels)
    return labels


def align_assignments(
    frame: pd.DataFrame, labels: pd.DataFrame, new_batch: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Restore label order and last-batch membership by the composite row key."""
    check_labels(labels)
    indexes = []
    for name, data in (("dataset", frame), ("labels", labels), ("new_batch", new_batch)):
        keys = data[list(ROW_KEY)]
        if keys.isna().any().any() or keys.duplicated().any():
            raise ValueError(f"Missing or duplicate row keys in {name}")
        indexes.append(pd.MultiIndex.from_frame(keys))
    frame_keys, label_keys, new_keys = indexes
    missing = len(frame_keys.difference(label_keys))
    extra = len(label_keys.difference(frame_keys))
    if missing or extra:
        raise ValueError(f"labels keys do not match dataset: missing={missing}, extra={extra}")
    outside = len(new_keys.difference(frame_keys))
    if outside:
        raise ValueError(f"new_batch keys outside dataset: {outside}")
    positions = label_keys.get_indexer(frame_keys)
    return labels["cluster_id"].to_numpy()[positions], frame_keys.isin(new_keys)


def report_clustering(
    dataset_path: Path | str,
    new_batch_path: Path | str,
    model_dir: Path | str,
    *,
    max_evaluation_rows=1000,
    random_state=42,
) -> ReportPaths:
    """Execute loading, validation, sampling, silhouette and t-SNE in a notebook."""
    validate_evaluation_parameters(max_evaluation_rows, random_state)
    dataset_path = Path(dataset_path).resolve()
    new_batch_path = Path(new_batch_path).resolve()
    model_dir = Path(model_dir).resolve()
    markdown = nbformat.v4.new_markdown_cell

    def code(source):
        return nbformat.v4.new_code_cell(dedent(source).strip())

    cells = [
        markdown("""# Оценка кластеризации Olist

## Входные данные и сохранённые артефакты
Единица наблюдения — **позиция заказа**, ключ — `(order_id, order_item_id)`.
Оцениваются готовые метки всего накопленного датасета. Новые строки — позиции
явно указанного последнего батча. Обучение выполняется отдельной командой `cluster`.
"""),
        code(f"""
            from pathlib import Path
            import hashlib
            import sys
            import joblib
            import pandas as pd
            import matplotlib.pyplot as plt
            from IPython.display import HTML, display
            from prak.schema import read_dataset
            from prak.auto_eda.checks import check_dataset
            from prak.auto_eda.eda import dataset_summary
            from prak.clustering.report import read_labels, align_assignments
            from prak.clustering.evaluation import evaluate_clustering
            from prak.clustering.plots import project_tsne, plot_tsne

            %matplotlib inline
            %config InlineBackend.figure_format = 'png'
            %config InlineBackend.print_figure_kwargs = {{'bbox_inches': None}}

            dataset_path = Path({str(dataset_path)!r})
            new_batch_path = Path({str(new_batch_path)!r})
            model_dir = Path({str(model_dir)!r})
            frame = read_dataset(dataset_path)
            new_batch = read_dataset(new_batch_path)
            assignments = read_labels(model_dir / 'labels.csv')
            distance = joblib.load(model_dir / 'distance.joblib')
            model = joblib.load(model_dir / 'model.joblib')
            for role, path in [('Датасет', dataset_path), ('Последний батч', new_batch_path),
                               ('Метки', model_dir / 'labels.csv')]:
                with path.open('rb') as source:
                    digest = hashlib.file_digest(source, 'sha256').hexdigest()
                print(f'{{role}}: {{path}}')
                print(f'SHA-256: {{digest}}')
            print(f'Модель: {{model_dir / "model.joblib"}}')
            print(f'Расстояние: {{model_dir / "distance.joblib"}}')
            print(f'Python: {{sys.executable}}')

            def show_table(table):
                display(HTML(table.to_html(index=False, escape=True, border=0)))
        """),
        markdown("## Размеры и период\nНакопленный датасет; даты без часового пояса, как в CSV."),
        code("show_table(dataset_summary(frame))\nprint(f'Последний батч: {len(new_batch):,} позиций.')"),
        markdown("## Параметры модели и расстояния\nМодель загружается только для показа параметров."),
        code("""
            print(f'Модель: {type(model).__name__}; параметры: {model.get_params(deep=True)}')
            print(f'Расстояние: {type(distance).__name__}; параметры: {distance.get_params(deep=True)}')
        """),
        markdown("""## Обязательные проверки
Строгая схема и типы, непустые CSV, пропуски, уникальность ключей,
хронологический порядок и положительная конечная цена проверяются для обоих входов.
Метки восстанавливаются по ключам, независимо от порядка строк `labels.csv`.
"""),
        code("""
            print('Накопленный датасет')
            show_table(check_dataset(frame))
            print('Последний батч')
            show_table(check_dataset(new_batch))
            labels, new_rows = align_assignments(frame, assignments, new_batch)
            print('OK: целочисленные метки; ключи меток точно совпадают с датасетом.')
            print('OK: все ключи последнего батча включены в датасет и имеют метки.')
        """),
        markdown("""## Выборка и силуэт
Случайная выборка без возвращения, независимо от меток. Квота — половина лимита
для новых строк (при нечётном лимите округление вверх) и половина для прежних;
при нехватке одной части добирается другая. Если строк меньше лимита, берутся все.
Матрица расстояний строится **после выборки**, один раз для силуэта и t-SNE.
Силуэт: от −1 до 1, больше — лучше разделение в выбранном расстоянии.
"""),
        code(f"""
            max_evaluation_rows = {int(max_evaluation_rows)}
            random_state = {int(random_state)}
            result = evaluate_clustering(
                frame, labels, distance=distance, new_rows=new_rows,
                max_evaluation_rows=max_evaluation_rows, random_state=random_state,
            )
            print(f'Лимит: {{max_evaluation_rows}}; random_state: {{random_state}}')
            print(f'Выбрано строк: {{len(result.sample_positions)}}; '
                  f'новых: {{result.new_count}}; прежних: {{result.previous_count}}')
            print(f'OK: расстояния конечные, неотрицательные, симметричные, '
                  f'диагональ нулевая; матрица {{result.distances.shape}}.')
            if result.silhouette is None:
                print(result.silhouette_reason)
            else:
                print(f'Силуэт: {{result.silhouette:.6f}}')
        """),
        markdown("""## t-SNE · проекция расстояний
Точка — позиция заказа, цвет — сохранённый номер кластера. t-SNE использует ту же
выборку и исходную матрицу расстояний; метки не участвуют в построении проекции.
Глобальные расстояния, площади и разрывы на рисунке не измеряют качество кластеров;
силуэт рассчитан по исходной матрице, а не по двумерным координатам.
При числе строк менее 3 проекция пропускается с причиной.
"""),
        code("""
            projection = project_tsne(result, random_state=random_state)
            print(f'Параметры t-SNE: {projection.parameters}')
            if projection.coordinates is None:
                print(projection.reason)
            else:
                figure = plot_tsne(result, projection)
                plt.show()
                plt.close(figure)
        """),
        markdown("""## Итог
Оценка описывает готовые метки в выбранном расстоянии на квотной выборке.
Доли новых и прежних позиций в ней могут отличаться от долей во всём датасете.
Состояние модели, расстояния и файл меток при оценке не изменяются.
"""),
    ]
    return execute_report(cells, model_dir, title="Оценка кластеризации Olist")
