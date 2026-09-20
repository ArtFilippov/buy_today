"""Command-line entry point for the local Olist pipeline."""

import argparse
from pathlib import Path

from prak.preparation import prepare_data


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ожидается положительное целое число") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("ожидается положительное целое число")
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
    args = parser.parse_args(argv)
    try:
        result = prepare_data(args.raw_dir, args.output_dir, args.batch_size)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Ошибка подготовки: {exc}\n")

    print(f"Строк до очистки: {result.rows_before_cleaning:,}")
    print(f"Строк после очистки: {result.rows_after_cleaning:,}")
    print(f"Удалено неполных строк: {result.rows_before_cleaning - result.rows_after_cleaning:,}")
    print(f"Батчей: {len(result.batch_sizes)}; размеры: {list(result.batch_sizes)}")
    print(f"Отброшенный из потока хвост: {result.dropped_tail_rows:,}")
    print(f"Рабочий датасет: {result.working_dataset_path}")
    print(f"Манифест: {result.manifest_path}")
