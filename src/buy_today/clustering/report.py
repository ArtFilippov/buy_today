"""Key alignment and executable reports for already saved clustering snapshots."""

from pathlib import Path
import json
from typing import Any, Unpack, cast

import numpy as np
from numpy.random import RandomState
import pandas as pd

from buy_today.auto_eda.cells import code, markdown, setup_code
from buy_today.auto_eda.notebook import ReportPaths, execute_report
from buy_today.clustering.contracts import BooleanArray, IntegerArray, Parameterized
from buy_today.clustering.parameters import ReportOptions, ReportParameters
from buy_today.clustering.labels import check_labels
from buy_today.schema import CSV_DTYPES, ROW_KEY


def parameter_metadata(estimator: Parameterized) -> dict[str, Any]:
    """Keep ordinary parameters typed and render non-JSON parameter objects.

    These are descriptive metadata, not an estimator serialization format;
    executable state remains in joblib. Numeric evaluation results stay strict.

    Args:
        estimator (Parameterized): Estimator exposing sklearn parameter metadata.

    Returns:
        dict[str, Any]: JSON-compatible descriptions of the estimator parameters.
    """

    def describe(value: object) -> object:
        if isinstance(value, np.generic):
            return cast("np.generic[object]", value).item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, RandomState):
            result: object = {"class": "RandomState", "state": value.get_state()}
        elif hasattr(value, "get_params"):
            parameters = cast(Parameterized, value).get_params(deep=False)
            result = {"class": type(value).__name__, "parameters": parameters}
        else:
            result = repr(value)
        return result

    return json.loads(
        json.dumps(estimator.get_params(deep=True), default=describe), parse_constant=str
    )


def read_labels(path: Path | str) -> pd.DataFrame:
    labels = pd.read_csv(
        path,
        dtype={
            **{column: CSV_DTYPES[column] for column in ROW_KEY},
            "cluster_id": "int64",
        },
    )
    check_labels(labels)
    return labels


def _row_index(frame: pd.DataFrame, name: str) -> pd.MultiIndex:
    keys = frame[list(ROW_KEY)]
    if keys.isna().any().any() or keys.duplicated().any():
        raise ValueError(f"Missing or duplicate row keys in {name}")
    return pd.MultiIndex.from_frame(keys)


def align_assignments(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    new_batch: pd.DataFrame,
) -> tuple[IntegerArray, BooleanArray]:
    """Restore label order and last-batch membership by the composite row key.

    Args:
        frame (pd.DataFrame): Evaluated dataset in desired row order.
        labels (pd.DataFrame): Saved keyed assignments in arbitrary order.
        new_batch (pd.DataFrame): Rows belonging to the most recent batch.

    Returns:
        tuple[IntegerArray, BooleanArray]: Labels and new-row membership in dataset order.

    Raises:
        ValueError: Assignment keys differ from the dataset or batch keys are outside it.
    """
    check_labels(labels)
    frame_keys = _row_index(frame, "dataset")
    label_keys = _row_index(labels, "labels")
    new_keys = _row_index(new_batch, "new_batch")
    missing = len(frame_keys.difference(label_keys))
    extra = len(label_keys.difference(frame_keys))
    if missing or extra:
        raise ValueError(f"labels keys do not match dataset: missing={missing}, extra={extra}")
    if outside := len(new_keys.difference(frame_keys)):
        raise ValueError(f"new_batch keys outside dataset: {outside}")
    positions = pd.MultiIndex.get_indexer(label_keys, frame_keys)
    return labels["cluster_id"].to_numpy()[positions], pd.MultiIndex.isin(frame_keys, new_keys)


def report_clustering(
    dataset_path: Path | str,
    new_batch_path: Path | str,
    model_dir: Path | str,
    **options: Unpack[ReportOptions],
) -> ReportPaths:
    """Execute loading, validation, sampling, silhouette and t-SNE in a notebook.

    Args:
        dataset_path (Path | str): Accumulated dataset CSV.
        new_batch_path (Path | str): Most recent batch CSV.
        model_dir (Path | str): Directory containing the saved clustering snapshot.
        **options (Unpack[ReportOptions]): Maximum sample size via
            ``max_evaluation_rows`` (default 1000), sampling and projection seed via
            ``random_state`` (default 42), and optional native kernel thread limit via
            ``thread_limit`` (default None).

    Returns:
        ReportPaths: Paths of the executed notebook and standalone HTML report.
    """
    parameters = ReportParameters(**options)
    dataset_path = Path(dataset_path).resolve()
    new_batch_path = Path(new_batch_path).resolve()
    model_dir = Path(model_dir).resolve()
    cells = [
        markdown("""# Оценка кластеризации Olist

## Входные данные и сохранённые артефакты
Единица наблюдения — **позиция заказа**, ключ — `(order_id, order_item_id)`.
Оцениваются готовые метки всего накопленного датасета. Новые строки — позиции
явно указанного последнего батча. Обучение выполняется отдельной командой `cluster`.
"""),
        setup_code(f"""
            import joblib
            from buy_today.auto_eda.checks import check_dataset
            from buy_today.auto_eda.eda import dataset_summary
            from buy_today.auto_eda.notebook import write_metrics
            from buy_today.clustering.report import read_labels, align_assignments, parameter_metadata
            from buy_today.clustering.evaluation import evaluate_clustering
            from buy_today.clustering.plots import project_tsne, plot_tsne

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
            inputs = {{}}
            for key, role, path in [
                ('dataset', 'Датасет', dataset_path),
                ('new_batch', 'Последний батч', new_batch_path),
                ('labels', 'Метки', model_dir / 'labels.csv'),
            ]:
                with path.open('rb') as source:
                    digest = hashlib.file_digest(source, 'sha256').hexdigest()
                inputs[key] = {{'path': str(path), 'sha256': digest}}
                print(f'{{role}}: {{path}}')
                print(f'SHA-256: {{digest}}')
            print(f'Модель: {{model_dir / "model.joblib"}}')
            print(f'Расстояние: {{model_dir / "distance.joblib"}}')
            print(f'Python: {{sys.executable}}')

            def show_table(table):
                display(HTML(table.to_html(index=False, escape=True, border=0)))
        """),
        markdown("## Размеры и период\nНакопленный датасет; даты без часового пояса, как в CSV."),
        code("""
            show_table(dataset_summary(frame))
            print(f'Последний батч: {len(new_batch):,} позиций.')
        """),
        markdown(
            "## Параметры модели и расстояния\nМодель загружается только для показа параметров."
        ),
        code("""
            model_info = {'class': type(model).__name__, 'parameters': parameter_metadata(model)}
            distance_info = {'class': type(distance).__name__, 'parameters': parameter_metadata(distance)}
            print(f"Модель: {model_info['class']}; параметры: {model_info['parameters']}")
            print(f"Расстояние: {distance_info['class']}; параметры: {distance_info['parameters']}")
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
            max_evaluation_rows = {int(parameters.max_evaluation_rows)}
            random_state = {int(parameters.random_state)}
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
        code("""
            write_metrics({
                'format_version': 1,
                'kind': 'clustering',
                'inputs': inputs,
                'silhouette': result.silhouette,
                'silhouette_reason': result.silhouette_reason,
                'sample_rows': len(result.sample_positions),
                'new_count': result.new_count,
                'previous_count': result.previous_count,
                'max_evaluation_rows': max_evaluation_rows,
                'random_state': random_state,
                'model': model_info,
                'distance': distance_info,
            })
        """),
    ]
    return execute_report(
        cells, model_dir, title="Оценка кластеризации Olist", thread_limit=parameters.thread_limit
    )
