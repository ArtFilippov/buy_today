"""Contents of the single-dataset report; calculations run in its notebook."""

from pathlib import Path
import pandas as pd

from buy_today.auto_eda.cells import DATASET_SCOPE, code, markdown, setup_code
from buy_today.auto_eda.notebook import ReportPaths, execute_report


def dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    times = frame["order_purchase_timestamp"]
    return pd.DataFrame(
        {
            "Показатель": [
                "Позиции заказа",
                "Колонки",
                "Уникальные заказы",
                "Уникальные покупатели",
                "Уникальные товары",
                "Начало периода покупок",
                "Конец периода покупок",
            ],
            "Значение": [
                str(len(frame)),
                str(len(frame.columns)),
                str(frame["order_id"].nunique()),
                str(frame["customer_unique_id"].nunique()),
                str(frame["product_id"].nunique()),
                str(times.min()),
                str(times.max()),
            ],
        }
    )


def dataset_conclusion(frame: pd.DataFrame) -> str:
    """Data-dependent, descriptive conclusion (no drift decision for one input).

    Args:
        frame (pd.DataFrame): Validated dataset to summarize.

    Returns:
        str: Dataset size, price quantiles and category and state coverage.
    """
    categories = frame["product_category_name"].value_counts()
    top = pd.Series.sort_index(categories).sort_values(ascending=False, kind="stable")
    return (
        f"Все обязательные проверки пройдены. Датасет: {len(frame):,} позиций × "
        f"{len(frame.columns)} колонок.\n"
        f"Медианная цена: {frame['price'].median():.2f} BRL; "
        f"90-й процентиль: {frame['price'].quantile(0.9):.2f} BRL.\n"
        f"Категорий: {len(categories)}; ведущая — {top.index[0]} "
        f"({top.iloc[0] / len(frame):.2%} позиций). "
        f"Штатов покупателей: {frame['customer_state'].nunique()}.\n"
        "Доли рассчитаны по позициям заказа, включая повторные покупки товара с разными ключами. "
        "Дрейф по одному датасету не оценивается."
    )


def report_dataset(dataset_path: Path | str, output_dir: Path | str) -> ReportPaths:
    """Create notebook, HTML and metrics; raise on invalid data or execution failure.

    Args:
        dataset_path (Path | str): Input dataset CSV.
        output_dir (Path | str): Directory for executed report artifacts.

    Returns:
        ReportPaths: Paths of the executed notebook and HTML report.
    """
    dataset_path = Path(dataset_path).resolve()
    cells = [
        markdown(
            """# EDA · один датасет Olist

## Входные данные
Единица наблюдения — **позиция заказа**, ключ — `(order_id, order_item_id)`.
Анализируется весь переданный CSV без семплирования и импутации.
"""
            + DATASET_SCOPE
        ),
        setup_code(f"""
            from buy_today.auto_eda.checks import check_dataset
            from buy_today.auto_eda.eda import dataset_summary, dataset_conclusion
            from buy_today.auto_eda.notebook import write_metrics
            from buy_today.auto_eda.plots import plot_price, plot_categories, plot_states

            %matplotlib inline
            %config InlineBackend.figure_format = 'png'
            %config InlineBackend.print_figure_kwargs = {{'bbox_inches': None}}

            dataset_path = Path({str(dataset_path)!r})
            frame = read_dataset(dataset_path)
            with dataset_path.open('rb') as source:
                digest = hashlib.file_digest(source, 'sha256').hexdigest()
            print(f'CSV: {{dataset_path}}')
            print(f'SHA-256: {{digest}}')
            print(f'Python: {{sys.executable}}')

            def show_table(table):
                display(HTML(table.to_html(index=False, escape=True, border=0)))
        """),
        markdown("## Размеры и период\nДаты покупки без часового пояса, как в исходном CSV."),
        code("summary = dataset_summary(frame)\nshow_table(summary)"),
        markdown("""## Обязательные проверки
Нарушение условия немедленно завершает исполнение. Хронология — неубывание
времени покупки; цепочка «оплата → перевозчик → доставка» не проверяется.
"""),
        code("checks = check_dataset(frame)\nshow_table(checks)"),
        markdown("""## Описательная статистика
Числовые признаки: число наблюдений, среднее, стандартное отклонение,
минимум, квартили и максимум. `order_item_id` — номер позиции, а не количество товара.
"""),
        code("""
            show_table(
                frame.select_dtypes(include='number').describe().T
                .rename_axis('Признак').reset_index()
            )
        """),
        markdown(
            "Строковые признаки: число наблюдений, число уникальных значений, мода и её частота."
        ),
        code(
            "show_table(frame.describe(include=['string']).T.rename_axis('Признак').reset_index())"
        ),
        markdown("""## Три распределения
### 1. Цена (`price`)
30 интервалов с равными шагами в логарифме цены; высота столбца — доля всех
позиций в интервале, не плотность на BRL. Включён весь диапазон положительных цен.
"""),
        code("figure = plot_price(frame)\nplt.show()\nplt.close(figure)"),
        markdown("""### 2. Категории (`product_category_name`)
Top-10 по числу позиций этого датасета; равенство частот разрешается по алфавиту.
Остальные категории объединены в «Прочие». Знаменатель — все позиции.
"""),
        code("figure = plot_categories(frame)\nplt.show()\nplt.close(figure)"),
        markdown("""### 3. Штаты покупателей (`customer_state`)
Штаты упорядочены по алфавиту. Доли по позициям заказа, не по уникальным
заказам или покупателям; нормировки на население нет.
"""),
        code("figure = plot_states(frame)\nplt.show()\nplt.close(figure)"),
        markdown("## Итог"),
        code("print(dataset_conclusion(frame))"),
        code("""
            rows, n_columns, n_orders, n_customers, n_products, period_start, period_end = summary['Значение']
            write_metrics({
                'format_version': 1,
                'kind': 'eda',
                'input': {'path': str(dataset_path), 'sha256': digest},
                'rows': int(rows),
                'n_columns': int(n_columns),
                'n_orders': int(n_orders),
                'n_customers': int(n_customers),
                'n_products': int(n_products),
                'period_start': period_start,
                'period_end': period_end,
                'checks': checks.to_dict('records'),
            })
        """),
    ]
    return execute_report(cells, output_dir, title="EDA · один датасет Olist")
