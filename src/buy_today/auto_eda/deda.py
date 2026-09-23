"""Comparison report contents; validation and drift calculations run in the notebook."""

from pathlib import Path
import pandas as pd

from buy_today.auto_eda.cells import DATASET_SCOPE, code, markdown, setup_code
from buy_today.auto_eda.drift import DEFAULT_THRESHOLDS, DriftResult, DriftThresholds
from buy_today.auto_eda.notebook import ReportPaths, execute_report


def drift_checks(result: DriftResult[pd.DataFrame]) -> pd.DataFrame:
    """Absence-of-drift checks are informative and never raise for detected drift.

    Args:
        result (DriftResult[pd.DataFrame]): Computed drift decisions.

    Returns:
        pd.DataFrame: Informative check rows, including failed absence checks.
    """
    return pd.DataFrame(
        [
            {
                "Проверка": f"Отсутствие дрейфа: {row['Признак']}",
                "Результат": "НЕ ПРОЙДЕНА" if row["Дрейф"] else "OK",
                "Фактически": f"{row['Мера']} = {row['Значение']:.6g}",
                "Ожидается": f"{row['Мера']} < {row['Порог']:.6g}",
            }
            for row in result.metrics.to_dict("records")
        ]
    )


def drift_conclusion(
    reference: pd.DataFrame, batch: pd.DataFrame, result: DriftResult[pd.DataFrame]
) -> str:
    changed = result.metrics.loc[result.metrics["Дрейф"], "Признак"].tolist()
    return (
        f"Эталон до добавления: {len(reference):,} позиций; батч: {len(batch):,} позиций.\n"
        + "Все обязательные проверки пройдены; пересечение ключей отсутствует. "
        + f"Объединение допустимо: {len(reference) + len(batch):,} позиций.\n"
        + f"drift_detected = {result.drift_detected}. "
        + (f"Дрейф обнаружен: {', '.join(changed)}.\n" if changed else "Дрейф не обнаружен.\n")
        + "Дрейф — информационный результат, он не препятствует обновлению эталона. "
        + "Это инженерные пороги расстояний распределений, не p-value и не оценка причинности. "
        + "Сам отчёт не изменяет входные CSV."
    )


def report_drift(
    batch_path: Path | str,
    reference_path: Path | str,
    output_dir: Path | str,
    *,
    thresholds: DriftThresholds = DEFAULT_THRESHOLDS,
) -> ReportPaths:
    """Report two existing CSVs; success permits an update even if drift is detected.

    Args:
        batch_path (Path | str): Incoming positions CSV.
        reference_path (Path | str): Current accumulated positions CSV.
        output_dir (Path | str): Directory for executed report artifacts.
        thresholds (DriftThresholds, default=DEFAULT_THRESHOLDS): Inclusive detection limits.

    Returns:
        ReportPaths: Paths of the executed notebook and HTML report.
    """
    batch_path = Path(batch_path).resolve()
    reference_path = Path(reference_path).resolve()
    cells = [
        markdown(
            """# DEDA · эталон и новый батч Olist

## Входные данные
Сравниваются текущий физический **эталон до добавления** и один новый **батч**.
Единица наблюдения — позиция заказа; ключ — `(order_id, order_item_id)`.
Оба CSV анализируются целиком, без семплирования, импутации и удаления дубликатов.
"""
            + DATASET_SCOPE
        ),
        setup_code(f"""
            from buy_today.auto_eda.checks import check_dataset, check_combination
            from buy_today.auto_eda.drift import DriftThresholds, evaluate_drift
            from buy_today.auto_eda.eda import dataset_summary
            from buy_today.auto_eda.deda import drift_checks, drift_conclusion
            from buy_today.auto_eda.notebook import write_metrics
            from buy_today.auto_eda.plots import (
                plot_price_comparison, plot_categories_comparison, plot_states_comparison,
            )

            %matplotlib inline
            %config InlineBackend.figure_format = 'png'
            %config InlineBackend.print_figure_kwargs = {{'bbox_inches': None}}

            reference_path = Path({str(reference_path)!r})
            batch_path = Path({str(batch_path)!r})
            reference = read_dataset(reference_path)
            batch = read_dataset(batch_path)
            thresholds = DriftThresholds(
                price={float(thresholds.price)!r}, category={float(thresholds.category)!r},
                state={float(thresholds.state)!r},
            )
            inputs = {{}}
            for key, role, path, frame in [
                ('reference', 'Эталон', reference_path, reference),
                ('batch', 'Батч', batch_path, batch),
            ]:
                with path.open('rb') as source:
                    digest = hashlib.file_digest(source, 'sha256').hexdigest()
                inputs[key] = {{'path': str(path), 'sha256': digest, 'rows': len(frame)}}
                print(f'{{role}} CSV: {{path}}')
                print(f'SHA-256: {{digest}}')
            print(f'Python: {{sys.executable}}')

            def show_table(table):
                display(HTML(table.to_html(index=False, escape=True, border=0)))
        """),
        markdown("## Размеры и периоды\nДаты покупки без часового пояса, как в исходных CSV."),
        code("""
            summary = dataset_summary(reference).rename(columns={'Значение': 'Эталон'})
            summary['Батч'] = dataset_summary(batch)['Значение']
            show_table(summary)
        """),
        markdown("""## Обязательные проверки
Каждый вход проверяется отдельно; затем проверяется уникальность ключей
предполагаемого объединения. Любое нарушение немедленно останавливает отчёт.
Хронология — неубывание времени покупки внутри каждого входа;
цепочка «оплата → перевозчик → доставка» не проверяется.
"""),
        code("""
            for role, frame in [('Эталон', reference), ('Батч', batch)]:
                print(role)
                show_table(check_dataset(frame))
            show_table(check_combination(reference, batch))
        """),
        markdown("""## Метрики и проверки отсутствия дрейфа
Цена: **KS D** — максимум абсолютного разрыва эмпирических CDF.
Категории и штаты: **TVD = 0.5 × Σ |p_batch − p_reference|** по объединению
всех значений, с нулевой долей для отсутствующих. Top-10 используется только на графике.
Каждое распределение нормируется на число позиций своего входа.
Дрейф признака: **значение ≥ порог**; общий `drift_detected` — логическое ИЛИ.
Пороги инженерные, не калиброванные уровни статистической значимости.
Непройденная проверка отсутствия дрейфа не останавливает отчёт или обновление.
"""),
        code("""
            drift = evaluate_drift(reference, batch, thresholds=thresholds)
            show_table(drift.metrics)
            show_table(drift_checks(drift))
            print(f'drift_detected = {drift.drift_detected}')
        """),
        markdown("""## Три сравнительных распределения
### 1. Цена (`price`)
Две наложенные гистограммы: 30 общих интервалов с равными шагами в логарифме
цены по полному диапазону обоих входов. Высота — доля позиций своего входа,
не плотность на BRL. Семплирования и обрезки хвостов нет.
"""),
        code("figure = plot_price_comparison(reference, batch)\nplt.show()\nplt.close(figure)"),
        markdown("""### 2. Категории (`product_category_name`)
Top-10 по текущему эталону, равные частоты разрешаются по алфавиту.
Остальные известные эталону категории — «Прочие»; отсутствующие в эталоне
категории батча — «Новые категории». Эти группы показываются при их наличии.
Знаменатель — все позиции соответствующего входа.
"""),
        code(
            "figure = plot_categories_comparison(reference, batch)\nplt.show()\nplt.close(figure)"
        ),
        markdown("""### 3. Штаты покупателей (`customer_state`)
Объединение штатов обоих входов, алфавитный порядок. Доли по позициям заказа,
не по уникальным заказам или покупателям; нормировки на население нет.
"""),
        code("figure = plot_states_comparison(reference, batch)\nplt.show()\nplt.close(figure)"),
        markdown("## Итог"),
        code("print(drift_conclusion(reference, batch, drift))"),
        code("""
            write_metrics({
                'format_version': 1,
                'kind': 'deda',
                'inputs': inputs,
                'drift_detected': drift.drift_detected,
                'metrics': drift.metrics.rename(columns={
                    'Признак': 'feature', 'Мера': 'measure', 'Значение': 'value',
                    'Порог': 'threshold', 'Дрейф': 'drift',
                }).to_dict('records'),
                'thresholds': {
                    'price': thresholds.price,
                    'category': thresholds.category,
                    'state': thresholds.state,
                },
            })
        """),
    ]
    return execute_report(cells, output_dir, title="DEDA · эталон и новый батч Olist")
