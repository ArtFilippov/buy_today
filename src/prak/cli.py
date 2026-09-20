"""Command-line entry point for the local Olist pipeline."""

import argparse
from math import isfinite
from pathlib import Path

from prak.auto_eda import DriftThresholds, report_dataset, report_drift
from prak.preparation import prepare_data
from prak.update import initialize_reference, update_reference


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ожидается положительное целое число") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("ожидается положительное целое число")
    return number


def _drift_threshold(value: str) -> float:
    message = "ожидается конечное число в [0, 1]"
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(message) from exc
    if not isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError(message)
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prak", description="Локальный конвейер Olist")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="собрать, очистить и разбить Olist на батчи",
        description="Подготовить рабочий датасет Olist и хронологический поток батчей.",
    )
    prepare.add_argument(
        "--raw-dir", type=Path, required=True, help="каталог исходных CSV Olist"
    )
    prepare.add_argument(
        "--output-dir", type=Path, required=True, help="каталог результатов подготовки"
    )
    prepare.add_argument(
        "--batch-size", type=_positive_int, default=5000,
        help="размер батчей после первого (по умолчанию: 5000)",
    )
    prepare.add_argument(
        "--min-category-count", type=_positive_int, default=1000,
        help="минимум позиций категории во всём очищенном датасете (по умолчанию: 1000)",
    )
    eda = commands.add_parser(
        "eda", help="проверить один датасет и создать EDA-отчёт",
        description="Исполнить EDA в отдельном ядре Python и сохранить notebook и HTML.",
    )
    eda.add_argument("--dataset", type=Path, required=True, help="подготовленный CSV Olist")
    eda.add_argument(
        "--output-dir", type=Path, required=True, help="каталог report.ipynb и report.html",
    )
    init = commands.add_parser(
        "init", help="создать или перезаписать эталон из батча и выполнить EDA",
        description="Проверить батч, записать эталон (с перезаписью) и создать EDA-отчёт.",
    )
    init.add_argument("--batch", type=Path, required=True, help="CSV первого батча")
    init.add_argument("--reference", type=Path, required=True, help="путь сохраняемого эталона CSV")
    init.add_argument(
        "--output-dir", type=Path, required=True, help="каталог шага; отчёт сохраняется в eda/",
    )
    defaults = DriftThresholds()
    for name, description in (
        ("deda", "Сравнить эталон и новый батч, сохранить DEDA notebook и HTML."),
        ("update", "Выполнить DEDA, добавить батч в существующий эталон и выполнить EDA."),
    ):
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--batch", type=Path, required=True, help="CSV нового батча")
        command.add_argument("--reference", type=Path, required=True, help="существующий эталон CSV")
        command.add_argument(
            "--output-dir", type=Path, required=True,
            help="каталог report.ipynb и report.html" if name == "deda" else
                 "каталог шага; отчёты сохраняются в deda/ и eda/",
        )
        for feature in ("price", "category", "state"):
            default = getattr(defaults, feature)
            command.add_argument(
                f"--{feature}-threshold", type=_drift_threshold, default=default,
                help=f"порог дрейфа {feature}, конечное число в [0, 1] (по умолчанию: {default:.2f})",
            )
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_data(
                args.raw_dir, args.output_dir, args.batch_size,
                min_category_count=args.min_category_count,
            )
        elif args.command == "eda":
            report = report_dataset(args.dataset, args.output_dir)
        elif args.command == "init":
            report = initialize_reference(args.batch, args.reference, args.output_dir)
        else:
            thresholds = DriftThresholds(
                price=args.price_threshold, category=args.category_threshold,
                state=args.state_threshold,
            )
            if args.command == "deda":
                report = report_drift(
                    args.batch, args.reference, args.output_dir, thresholds=thresholds,
                )
            else:
                reports = update_reference(
                    args.batch, args.reference, args.output_dir, thresholds=thresholds,
                )
    except Exception as exc:
        # CLI boundary: data, kernel, rendering and filesystem errors all exit 1.
        stage = "подготовки" if args.command == "prepare" else args.command
        parser.exit(1, f"Ошибка {stage}: {exc}\n")

    if args.command == "update":
        print(f"Эталон: {args.reference.resolve()}")
        for name, report in (("DEDA", reports.deda), ("EDA", reports.eda)):
            print(f"{name} Notebook: {report.notebook_path}")
            print(f"{name} HTML: {report.html_path}")
        return

    if args.command != "prepare":
        if args.command == "init":
            print(f"Эталон: {args.reference.resolve()}")
        print(f"Notebook: {report.notebook_path}")
        print(f"HTML: {report.html_path}")
        return

    print(f"Строк до очистки: {result.rows_before_cleaning:,}")
    print(f"Строк после очистки: {result.rows_after_cleaning:,}")
    print(f"Удалено неполных строк: {result.rows_before_cleaning - result.rows_after_cleaning:,}")
    print(f"Удалено строк редких категорий: {result.rows_after_cleaning - result.rows_after_category_filtering:,}")
    print(f"Категорий: {result.categories_before_filtering} → {result.categories_after_filtering}")
    print(f"Строк после фильтрации категорий: {result.rows_after_category_filtering:,}")
    print(f"Батчей: {len(result.batch_sizes)}; размеры: {list(result.batch_sizes)}")
    print(f"Отброшенный из потока хвост: {result.dropped_tail_rows:,}")
    print(f"Рабочий датасет: {result.working_dataset_path}")
    print(f"Манифест: {result.manifest_path}")
