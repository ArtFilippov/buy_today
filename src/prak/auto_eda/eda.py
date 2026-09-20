"""Contents of the single-dataset report; calculations run in its notebook."""

from pathlib import Path
from textwrap import dedent

import nbformat
import pandas as pd

from prak.auto_eda.notebook import ReportPaths, execute_report


def dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    times = frame["order_purchase_timestamp"]
    return pd.DataFrame({
        "Показатель": [
            "Позиции заказа", "Колонки", "Уникальные заказы", "Уникальные покупатели",
            "Уникальные товары", "Начало периода покупок", "Конец периода покупок",
        ],
        "Значение": [
            str(len(frame)), str(len(frame.columns)), str(frame["order_id"].nunique()),
            str(frame["customer_unique_id"].nunique()), str(frame["product_id"].nunique()),
            str(times.min()), str(times.max()),
        ],
    })


def dataset_conclusion(frame: pd.DataFrame) -> str:
    """Data-dependent, descriptive conclusion (no drift decision for one input)."""
    categories = frame["product_category_name"].value_counts()
    top = categories.sort_index().sort_values(ascending=False, kind="stable")
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
    """Create report.ipynb and report.html; raise on invalid data or execution failure."""
    dataset_path = Path(dataset_path).resolve()
    markdown = nbformat.v4.new_markdown_cell

    def code(source: str) -> nbformat.NotebookNode:
        return nbformat.v4.new_code_cell(dedent(source).strip())

    cells = [
        markdown("""# EDA · один датасет Olist

## Входные данные
Единица наблюдения — **позиция заказа**, ключ — `(order_id, order_item_id)`.
Анализируется весь переданный CSV без семплирования и импутации.
Подготовленный Olist — ретроспективная выборка доставленных покупок товаров
с единственным продавцом, после удаления строк с любым пропуском.
Распределения описывают сохранённые покупки, а не весь рынок или каталог.
"""),
        code(f"""
            from pathlib import Path
            import hashlib
            import sys
            import pandas as pd
            import matplotlib.pyplot as plt
            from IPython.display import HTML, display
            from prak.schema import read_dataset
            from prak.auto_eda.checks import check_dataset
            from prak.auto_eda.eda import dataset_summary, dataset_conclusion
            from prak.auto_eda.plots import plot_price, plot_categories, plot_states

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
        code("show_table(dataset_summary(frame))"),
        markdown("""## Обязательные проверки
Нарушение условия немедленно завершает исполнение. Хронология — неубывание
времени покупки; цепочка «оплата → перевозчик → доставка» не проверяется.
"""),
        code("checks = check_dataset(frame)\nshow_table(checks)"),
        markdown("""## Описательная статистика
Числовые признаки: число наблюдений, среднее, стандартное отклонение,
минимум, квартили и максимум. `order_item_id` — номер позиции, а не количество товара.
"""),
        code("show_table(frame.select_dtypes(include='number').describe().T.rename_axis('Признак').reset_index())"),
        markdown("Строковые признаки: число наблюдений, число уникальных значений, мода и её частота."),
        code("show_table(frame.describe(include=['string']).T.rename_axis('Признак').reset_index())"),
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
    ]
    return execute_report(cells, output_dir, title="EDA · один датасет Olist")
