"""Parser construction for the local Olist command-line interface."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any, override

from buy_today.auto_eda.domain import DriftThresholds
from buy_today.cli_values import (
    Command,
    benchmark_model,
    drift_threshold,
    positive_float,
    positive_int,
    random_state,
)


type ParserFactory = Callable[[str, str, str | None], argparse.ArgumentParser]


class _SingleReference(argparse.Action):
    @override
    def __call__(
        self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
        values: Any, option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} принимает один CSV и не может повторяться")
        setattr(namespace, self.dest, values)


def command_factory(parser: argparse.ArgumentParser) -> ParserFactory:
    commands = parser.add_subparsers(dest="command", required=True)

    def add_command(name: str, help_text: str, description: str | None) -> argparse.ArgumentParser:
        return commands.add_parser(name, help=help_text, description=description)

    return add_command


def _run(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--run-dir", type=Path, required=True, help="новый или существующий каталог прогона"
    )
    _ = parser.add_argument(
        "--data-dir", type=Path, help="подготовленный поток; обязателен для нового прогона"
    )
    _ = parser.add_argument("--all", action="store_true", help="обработать все оставшиеся батчи")
    _ = parser.add_argument(
        "--model", choices=["random", "svd"], help="один ранжировщик; сначала по умолчанию svd"
    )
    _ = parser.add_argument("--random-state", type=random_state)
    _ = parser.add_argument(
        "--temperature",
        type=positive_float,
        help="температура TimestampDistance, по умолчанию 86400 секунд",
    )
    for name, help_text in (
        ("initial-users", "новые пользователи первого батча; по умолчанию 2000"),
        ("additional-users", "новые пользователи каждого следующего батча; по умолчанию 250"),
        ("k", "K оценки; по умолчанию 10"),
        ("svd-n-components", "компоненты SVD; по умолчанию 32"),
        ("svd-n-iter", "итерации SVD; по умолчанию 7"),
        ("temporal-n-clusters", "число кластеров временной модели; по умолчанию выбирает модель"),
        ("max-evaluation-rows", "лимит оценки кластеризации; по умолчанию 1000"),
    ):
        _ = parser.add_argument(f"--{name}", type=positive_int, help=help_text)
    _ = parser.add_argument(
        "--split-sizes", type=positive_int, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST")
    )
    for feature in ("price", "category", "state"):
        _ = parser.add_argument(f"--{feature}-threshold", type=drift_threshold)


def _inference(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--model-dir", type=Path, required=True)
    _ = parser.add_argument("--user-id", required=True)
    _ = parser.add_argument("--k", type=positive_int, default=10)
    _ = parser.add_argument(
        "--output", type=Path, required=True, help="CSV: user_id, rank, product_id"
    )


def _summary(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--run-dir", type=Path, required=True)


def _prepare(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--raw-dir", type=Path, required=True, help="каталог исходных CSV Olist"
    )
    _ = parser.add_argument(
        "--output-dir", type=Path, required=True, help="каталог результатов подготовки"
    )
    _ = parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=5000,
        help="размер батчей после первого (по умолчанию: 5000)",
    )
    _ = parser.add_argument(
        "--min-category-count",
        type=positive_int,
        default=1000,
        help="минимум позиций категории во всём очищенном датасете (по умолчанию: 1000)",
    )


def _eda(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--dataset", type=Path, required=True, help="подготовленный CSV Olist")
    _ = parser.add_argument(
        "--output-dir", type=Path, required=True, help="каталог report.ipynb и report.html"
    )


def _init(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--batch", type=Path, required=True, help="CSV первого батча")
    _ = parser.add_argument(
        "--reference", type=Path, required=True, help="путь сохраняемого эталона CSV"
    )
    _ = parser.add_argument(
        "--output-dir", type=Path, required=True, help="каталог шага; отчёт сохраняется в eda/"
    )


def _drift(parser: argparse.ArgumentParser, output_help: str) -> None:
    _ = parser.add_argument("--batch", type=Path, required=True, help="CSV нового батча")
    _ = parser.add_argument("--reference", type=Path, required=True, help="существующий эталон CSV")
    _ = parser.add_argument("--output-dir", type=Path, required=True, help=output_help)
    defaults = DriftThresholds()
    for feature in ("price", "category", "state"):
        default = getattr(defaults, feature)
        _ = parser.add_argument(
            f"--{feature}-threshold",
            type=drift_threshold,
            default=default,
            help=f"порог дрейфа {feature}, конечное число в [0, 1] (по умолчанию: {default:.2f})",
        )


def _deda(parser: argparse.ArgumentParser) -> None:
    _drift(parser, "каталог report.ipynb и report.html")


def _update(parser: argparse.ArgumentParser) -> None:
    _drift(parser, "каталог шага; отчёты сохраняются в deda/ и eda/")


def _cluster(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--batch", type=Path, required=True, help="CSV; для temporal — накопленные данные"
    )
    _ = parser.add_argument(
        "--output-dir", type=Path, required=True, help="каталог модели данного шага"
    )
    _ = parser.add_argument("--model", choices=["temporal"], default="temporal")
    _ = parser.add_argument("--distance", choices=["timestamp"], default="timestamp")
    _ = parser.add_argument(
        "--temporal-n-clusters",
        type=positive_int,
        help="число групп temporal; по умолчанию определяется реализацией модели",
    )


def _generate(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--batch", type=Path, required=True, help="CSV только нового батча")
    _ = parser.add_argument(
        "--distance", type=Path, required=True, help="готовое расстояние в joblib"
    )
    _ = parser.add_argument(
        "--output-dir", type=Path, required=True, help="новый каталог накопленного снимка"
    )
    _ = parser.add_argument("--previous-dir", type=Path, help="предыдущий снимок историй")
    _ = parser.add_argument(
        "--temperature",
        type=positive_float,
        required=True,
        help="температура в единицах расстояния",
    )
    _ = parser.add_argument(
        "--n-users", type=positive_int, help="новые пользователи: по умолчанию 2000, затем 250"
    )
    _ = parser.add_argument(
        "--split-sizes",
        type=positive_int,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        help="покупки на пользователя: сначала 70 15 15; затем наследуются из предыдущего снимка",
    )
    _ = parser.add_argument("--random-state", type=random_state, default=42)


def _history_quality(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--dataset-dir", type=Path, required=True, help="готовый снимок с generator/"
    )
    _ = parser.add_argument(
        "--reference", type=Path, action=_SingleReference, required=True,
        help="один CSV всех исходных батчей снимка",
    )
    _ = parser.add_argument(
        "--output-dir", type=Path, required=True, help="новый каталог отчёта вне снимка"
    )
    _ = parser.add_argument("--iid-repeats", type=positive_int, default=100)
    _ = parser.add_argument("--random-state", type=random_state, default=42)


def _rank(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--dataset-dir", type=Path, required=True, help="снимок модельных историй"
    )
    _ = parser.add_argument("--output-dir", type=Path, required=True, help="новый каталог модели")
    _ = parser.add_argument("--model", choices=["random", "svd"], default="random")
    _ = parser.add_argument("--random-state", type=random_state, default=42)
    _ = parser.add_argument(
        "--svd-n-components",
        type=positive_int,
        help="число компонент SVD; по умолчанию определяется моделью (32)",
    )
    _ = parser.add_argument(
        "--svd-n-iter",
        type=positive_int,
        help="число итераций SVD; по умолчанию определяется моделью (7)",
    )


def _evaluate_ranking(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--dataset-dir", type=Path, required=True, help="снимок модельных историй"
    )
    _ = parser.add_argument(
        "--model-dir", type=Path, required=True, help="каталог готового ранжировщика"
    )
    _ = parser.add_argument("--output-dir", type=Path, required=True, help="новый каталог оценки")
    _ = parser.add_argument("--split", choices=["validation", "test"], required=True)
    _ = parser.add_argument("--k", type=positive_int, default=10)


def _benchmark(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--output", type=Path, required=True, help="общий корень результатов")
    _ = parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        required=True,
        help="готовая папка с train.csv, test.csv и catalog.csv; можно повторять",
    )
    _ = parser.add_argument(
        "--model",
        type=benchmark_model,
        action="append",
        required=True,
        help='спецификация, например "svd --n-components 16 --n-iter 7"; можно повторять',
    )
    _ = parser.add_argument("--k", type=positive_int, default=10)
    _ = parser.add_argument(
        "--no-save-model", action="store_true", help="сохранить только метрики и метаданные"
    )


def _benchmark_report(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("root", type=Path, help="один общий корень результатов benchmark")
    _ = parser.add_argument(
        "--output", type=Path, required=True, help="путь HTML; CSV записывается рядом"
    )


def _evaluate(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--dataset", type=Path, required=True, help="CSV всех накопленных данных"
    )
    _ = parser.add_argument(
        "--new-batch", type=Path, required=True, help="CSV последнего добавленного батча"
    )
    _ = parser.add_argument(
        "--model-dir", type=Path, required=True, help="каталог модели, расстояния и меток"
    )
    _ = parser.add_argument("--max-evaluation-rows", type=positive_int, default=1000)
    _ = parser.add_argument("--random-state", type=random_state, default=42)


_COMMANDS: tuple[
    tuple[Command, str, str | None, Callable[[argparse.ArgumentParser], None]], ...
] = (
    (
        Command.RUN,
        "выполнить полный цикл для следующего батча или остатка потока",
        "Один ранжировщик на прогон; параметры фиксируются при первом запуске.",
        _run,
    ),
    (
        Command.INFERENCE,
        "сохранить рекомендации выбранной модели для пользователя",
        None,
        _inference,
    ),
    (Command.SUMMARY, "сводка завершённых шагов одного прогона: HTML, JSON, CSV", None, _summary),
    (
        Command.PREPARE,
        "собрать, очистить и разбить Olist на батчи",
        "Подготовить рабочий датасет Olist и хронологический поток батчей.",
        _prepare,
    ),
    (
        Command.EDA,
        "проверить один датасет и создать EDA-отчёт",
        "Исполнить EDA в отдельном ядре Python и сохранить notebook и HTML.",
        _eda,
    ),
    (
        Command.INIT,
        "создать или перезаписать эталон из батча и выполнить EDA",
        "Проверить батч, записать эталон (с перезаписью) и создать EDA-отчёт.",
        _init,
    ),
    (
        Command.DEDA,
        "Сравнить эталон и новый батч, сохранить DEDA notebook и HTML.",
        "Сравнить эталон и новый батч, сохранить DEDA notebook и HTML.",
        _deda,
    ),
    (
        Command.UPDATE,
        "Выполнить DEDA, добавить батч в существующий эталон и выполнить EDA.",
        "Выполнить DEDA, добавить батч в существующий эталон и выполнить EDA.",
        _update,
    ),
    (
        Command.CLUSTER,
        "обучить кластеризацию и сохранить модель, расстояние и метки",
        "Временной бейзлайн: равные последовательные группы по времени покупки.",
        _cluster,
    ),
    (
        Command.GENERATE,
        "создать и накопить модельные пользовательские истории",
        "Новые истории из нового батча и готового расстояния; прежние истории сохраняются.",
        _generate,
    ),
    (
        Command.HISTORY_QUALITY,
        "оценить частоты и концентрацию категорий синтетических историй",
        "Общий отчёт по train + validation + test с IID baseline; метрики без порогов допуска.",
        _history_quality,
    ),
    (
        Command.RANK,
        "обучить ранжировщик на train и сохранить модель",
        "Random или SVD на накопленном train и полном каталоге товаров.",
        _rank,
    ),
    (
        Command.EVALUATE_RANKING,
        "оценить сохранённый ранжировщик: Recall@K и NDCG@K",
        "Независимая оценка выбранного split без обучения; JSON и CSV по пользователям.",
        _evaluate_ranking,
    ),
    (
        Command.BENCHMARK,
        "сравнить конфигурации ранжировщиков на готовых датасетах",
        "Одно свежее обучение на train и оценка на test каждой пары модель × датасет.",
        _benchmark,
    ),
    (
        Command.BENCHMARK_REPORT,
        "HTML/CSV сравнения последних завершённых benchmark runs",
        "Общий рейтинг по максимуму NDCG@K по датасетам; без обучения и чтения датасетов.",
        _benchmark_report,
    ),
    (
        Command.EVALUATE,
        "оценить сохранённые метки: силуэт и отчёт с t-SNE",
        "Независимая оценка без обучения: выполненный notebook и самодостаточный HTML.",
        _evaluate,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="buy_today", description="Локальный конвейер Olist")
    _configure_commands(command_factory(parser))
    return parser


def _configure_commands(add_parser: ParserFactory) -> None:
    for name, help_text, description, configure in _COMMANDS:
        command = add_parser(name.value, help_text, description)
        configure(command)
