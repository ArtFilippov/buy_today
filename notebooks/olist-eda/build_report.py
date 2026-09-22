"""Собрать исследование и выполнить notebook в новом ядре текущего Python.

Сначала: MPLBACKEND=Agg .venv/bin/python notebooks/olist-eda/explore.py
Затем:   .venv/bin/python notebooks/olist-eda/build_report.py
"""

import hashlib
import json
from pathlib import Path
import re
import sys

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

OPENING = """- **Вопрос.** Подходит ли собранный Olist для выбранного рекомендательного baseline, и какие свойства данных нужно учесть?
- **Короткий ответ.** Для исследования признаков и воспроизводимых синтетических экспериментов — да. Реальные пользовательские истории слишком коротки для широкого персонального holdout; нужна отдельная обработка cold start. Исходный отбор остаётся ретроспективным.
- **Объект.** 96 068 позиций × 35 столбцов; 84 130 заказов, 81 541 пользователь (`customer_unique_id`), 30 991 товар. Реальные покупки 15.09.2016–29.08.2018; синтетические события здесь не создаются.
- **Порядок.** Все 13 исследовательских графиков сохранены в порядке исследования, включая проверки чувствительности и результаты без существенных изменений.
"""

INPUTS = """- **Входы.** `../../dataset/olist_prepared_dataset.csv` и `../../datasets/olist-eda/source_reference.csv.gz`; SHA-256 проверяется по `input_manifest.json`. [Происхождение и команды](../../datasets/olist-eda/dataset_evidence.md), [контекст](../../visualizations/olist-eda/data_context.md), [описание полей](../../docs/data.md#схема-и-все-35-признаков).
- **Reference.** По всем исходным позициям присоединены заказ, пользователь и категория; сохранены число продавцов товара и признак отбора. В notebook читается готовый результат. При его отсутствии выполнить **из корня репозитория**: `.venv/bin/python datasets/olist-eda/collect_reference.py`.
- **Правила.** Пропуски не импутируются; заказные метрики дедуплицируются по `order_id`, товарные — по `product_id`. Почтовые префиксы читаются строками. Выборок и случайных seed нет: используется полный доступный набор.
- **Параметры.** Ниже доступны `TOP_N`, `CUT_DATES`, `HOLDOUT_DAYS`. Запускать сверху вниз в окружении проекта (`uv sync --frozen`); сборка данных внутри notebook не вызывается.
"""

TRANSITIONS = {
    "quality": "- **Проверка смысла.** Уникальность ключа не исключает ошибочных дат и координат. Ниже отдельные диагностические флаги; пересекающиеся счётчики нельзя суммировать.",
    "P01": "- **Уровень товара.** Пропуски категории затрагивают 584 разных товара. Подсчёт по позициям дополнительно взвешивает проблему популярностью этих товаров.",
    "P02": "- **Проверка отбора.** Чтобы отделить свойства источника от решений сборки, используем reference до фильтров. Его ключи `kept=True` в точности совпали с итоговым CSV.",
    "P03": "- **Временная опора.** Перед выбором границ обучения проверяем наблюдаемое покрытие. Отсутствие строк в месяце означает отсутствие покупок в выгрузке, но не доказывает нулевую активность платформы.",
    "P04": "- **Пространство расстояний.** Проверим масштабы перед стандартным масштабированием: редкие крупные значения могут сильно влиять на евклидовы расстояния. Наблюдаемая цена меняется внутри 4 646 товаров.",
    "P05": "- **Состав покупок.** Часть различий цен может объясняться категориями; распределение проданных позиций не является прайс-листом всего каталога.",
    "P06": "- **Географический контекст.** Город без штата может объединять разные места. Здесь используются устойчивее интерпретируемые адресные коды штатов, без нормировки на население.",
    "P07": "- **История пользователя.** `customer_id` относится к записи заказа; объединять реальные истории нужно через `customer_unique_id`. Ни размер корзины, ни повторные единицы товара не заменяют отдельные заказы.",
    "P08": "- **История товара.** Плотность наблюдаемой матрицы user–item — 0,003442%: 86 973 уникальные пары. В разных заказах повторяются только 327 пар. Время присутствия товара и экспозиции неизвестны.",
    "P09": "- **Практическое следствие.** Большое общее число пользователей не означает большой набор персональных тестовых историй. Проверяем три независимых временных окна; это аудит покрытия до моделирования.",
    "P10": "- **Зависимость признаков.** Сопоставляем вес каждой проданной позиции с весом каждого товара. Спирмен отражает монотонность, а не причинность; парные `n` не являются числом независимых наблюдений.",
    "P11": "- **Проверка объяснения.** Общая связь расстояния и доставки может быть связана с массой. Проверим группы массы, а затем хвосты, повторные единицы и одну категорию.",
    "P12": "- **Отдельная ретроспективная задача.** Стоимость доставки и её срок — разные величины. Опоздание ниже определяется календарной датой позже обещанной: доставка в обещанный день не считается поздней. Исключение по согласованности требует также полноты всех пяти дат.",
    "P13": "- **Возвращение к качеству.** Проверим, не скрывает ли общая небольшая доля пропусков концентрацию по времени и регионам. Complete-case сравнение служит проверкой устойчивости выводов, а не правилом удаления строк.",
    "diagnostics": "- **Уточнение аномалий.** Пропуски DF распределены по 58 почтовым префиксам. Ни адресный штат, ни геосправочник автоматически не объявляются истиной. В конце проверяем повторы при одинаковом 90-дневном окне наблюдения.",
}

CONCLUSIONS = """- **Оценка рекомендаций.** Синтетические истории оправданы как учебный эксперимент, но их метрики не доказывают качество на реальных пользователях. Для реальных данных нужны явно определённые cold-start-правила и отдельный отчёт о покрытии; товары без обучающей истории сохраняются в ground truth.
- **Отбор.** Фильтр по единственному продавцу во всей истории и конечному статусу использует последующую информацию. Даже хронологическое разбиение текущего CSV остаётся условным на ретроспективном отборе. Суммы частичных корзин не следует называть полным чеком.
- **Предобработка.** Идентификаторы — ключи, индексы — категории; константный `order_status` не различает строки. Полное one-hot кодирование городов и префиксов создаёт десятки тысяч координат и дублирует географические сигналы. Проверить вклад групп в расстояние; обучать imputer/scaler/encoder только на train.
- **Числа.** Сравнить baseline со способами уменьшения влияния хвостов, например `log1p` для неотрицательных величин. Учесть взаимосвязь массы и объёма. Низкая общая корреляция числа фото не доказывает бесполезность признака. Нули массы и аномальные координаты требуют явного правила, а не общего удаления строк.
- **Доступность во времени.** Фактические подтверждение, перевозка и доставка недоступны при оформлении покупки. Производные сроки допустимы для этого EDA; в рекомендательной модели их применение зависит от явно заданного момента прогноза. Исторические версии карточек отсутствуют.
- **Границы вывода.** Неизвестны дата среза статусов, истинная доступность каталога и причины пропусков; возможны цензурирование последних когорт и остаточное смешение факторов. Неоднородность геопропусков требует контроля по регионам. Подгрупповые проверки устойчивости не заменяют причинного исследования.
"""


def main():
    source = (HERE / "explore.py").read_text()
    findings_path = HERE / "_work/findings.json"
    if not findings_path.exists():
        raise FileNotFoundError("Сначала выполните explore.py и визуально проверьте рисунки.")
    parts = re.split(r"^# %% (\w+)\s*$", source, flags=re.MULTILINE)
    sections = dict(zip(parts[1::2], parts[2::2]))
    notes = (HERE / "_work/analysis_notes.md").read_text()
    expected = re.findall(r"^\| (P\d{2}) \|", notes, flags=re.MULTILINE)
    actual = [label for label in sections if re.fullmatch(r"P\d{2}", label)]
    assert expected == actual and len(actual) == 13, (expected, actual)
    for plot_id in actual:
        assert (ROOT / f"visualizations/olist-eda/{plot_id}.png").exists()

    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
    nb.metadata["analysis"] = {"name": "olist-eda", "plot_order": actual,
                               "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
    nb.cells = [nbformat.v4.new_markdown_cell(OPENING, id="question-answer"),
                nbformat.v4.new_markdown_cell(INPUTS, id="inputs")]
    inline = """%matplotlib inline
%config InlineBackend.figure_format = 'png'
%config InlineBackend.print_figure_kwargs = {'bbox_inches': None}
"""
    for label, content in sections.items():
        if label == "save_findings":
            continue
        if label in TRANSITIONS:
            nb.cells.append(nbformat.v4.new_markdown_cell(TRANSITIONS[label], id=f"text-{label.lower()}"))
        content = content.strip()
        if label == "setup":
            content = inline + content.replace('EXPORT_FIGURES = "__file__" in globals()', 'EXPORT_FIGURES = False')
        cell = nbformat.v4.new_code_cell(f"# {label}\n{content}", id=f"code-{label.lower()}")
        if label in actual:
            cell.metadata["plot_id"] = label
        nb.cells.append(cell)
    nb.cells.append(nbformat.v4.new_markdown_cell(CONCLUSIONS, id="conclusions"))
    # Assertions here run in the notebook too: a numerical check of narrative facts.
    nb.cells.append(nbformat.v4.new_code_cell(
        "assert plot_ids == " + repr(actual) + "\n"
        "assert findings['rows'] == 96068 and findings['columns'] == 35\n"
        "assert findings['orders'] == 84130 and findings['products'] == 30991\n"
        "assert findings['repeat_users'] == 2318 and findings['partial_orders'] == 328\n"
        "assert findings['late_orders'] == 5545\n"
        "assert findings['df_missing_geo_prefixes'] == 58\n"
        "summary = pd.Series({\n"
        "    'Пользователи с одним заказом, %': findings['single_order_share'] * 100,\n"
        "    'Товары в одной позиции, %': findings['singleton_product_share'] * 100,\n"
        "    'Плотность user–item, %': findings['matrix_density'] * 100,\n"
        "    'Медиана срока доставки, дней': findings['delivery_median_days'],\n"
        "    'Поздние доставки, %': findings['late_share'] * 100,\n"
        "    'One-hot: сумма наблюдаемых категорий без ID': findings['categorical_width_observed'],\n"
        "})\n"
        "display(summary.round(4))\n"
        "print('Проверены расчёты и последовательность всех 13 рисунков.')",
        id="reconciliation"))
    nbformat.validate(nb)
    client = NotebookClient(nb, timeout=180, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    # Не полагаемся на системную kernelspec: запускаем именно окружение команды.
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client.execute()
    nbformat.validate(nb)
    rendered = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        assert cell.execution_count is not None
        assert not any(output.output_type == "error" for output in cell.outputs)
        if "plot_id" in cell.metadata:
            images = [o for o in cell.outputs if "image/png" in o.get("data", {})]
            assert len(images) == 1, (cell.metadata.plot_id, len(images))
            rendered.append(cell.metadata.plot_id)
    assert rendered == expected
    destination = HERE / "report.ipynb"
    nbformat.write(nb, destination)
    validation = {
        "python": sys.version, "executable": str(Path(sys.executable).relative_to(ROOT)),
        "notebook": str(destination.relative_to(ROOT)),
        "fresh_kernel_execution": "passed", "nbformat_validation": "passed",
        "code_cells_executed": sum(c.cell_type == "code" for c in nb.cells),
        "plot_order": rendered, "all_plots_have_code_and_png": True,
        "source_sha256": nb.metadata.analysis.source_sha256,
        "notebook_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    (HERE / "_work/validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
