"""Command-line entry point for the local Olist pipeline."""

import argparse
from dataclasses import replace
from functools import partial
from math import isfinite
from pathlib import Path

from prak.auto_eda import DriftThresholds, report_dataset, report_drift
from prak.clustering.distances import TimestampDistance
from prak.clustering.models.temporal import train_temporal
from prak.clustering.report import report_clustering
from prak.clustering.training import train_clustering
from prak.generation import generate_dataset
from prak.preparation import prepare_data
from prak.pipeline import PipelineConfig, read_pipeline_config, run_pipeline
from prak.ranking import RandomRanker, SVDRanker, export_recommendations, report_ranking, train_ranker
from prak.summary import report_summary
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


def _random_state(value: str) -> int:
    message = "ожидается целое число в [0, 2**32 - 1]"
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(message) from exc
    if not 0 <= number <= 2**32 - 1:
        raise argparse.ArgumentTypeError(message)
    return number


def _positive_float(value: str) -> float:
    message = "ожидается конечное положительное число"
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(message) from exc
    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(message)
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prak", description="Локальный конвейер Olist")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="выполнить полный цикл для следующего батча или остатка потока",
        description="Один ранжировщик на прогон; параметры фиксируются при первом запуске.",
    )
    run.add_argument("--run-dir", type=Path, required=True, help="новый или существующий каталог прогона")
    run.add_argument("--data-dir", type=Path, help="подготовленный поток; обязателен для нового прогона")
    run.add_argument("--all", action="store_true", help="обработать все оставшиеся батчи")
    run.add_argument("--model", choices=["random", "svd"], help="один ранжировщик; сначала по умолчанию svd")
    run.add_argument("--random-state", type=_random_state)
    run.add_argument("--temperature", type=_positive_float, help="температура TimestampDistance, по умолчанию 86400 секунд")
    for name, help_text in (
        ("initial-users", "новые пользователи первого батча; по умолчанию 2000"),
        ("additional-users", "новые пользователи каждого следующего батча; по умолчанию 250"),
        ("k", "K оценки; по умолчанию 10"),
        ("svd-n-components", "компоненты SVD; по умолчанию 32"),
        ("svd-n-iter", "итерации SVD; по умолчанию 7"),
        ("temporal-n-clusters", "число кластеров временной модели; по умолчанию выбирает модель"),
        ("max-evaluation-rows", "лимит оценки кластеризации; по умолчанию 1000"),
    ):
        run.add_argument(f"--{name}", type=_positive_int, help=help_text)
    run.add_argument("--split-sizes", type=_positive_int, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST"))
    for feature in ("price", "category", "state"):
        run.add_argument(f"--{feature}-threshold", type=_drift_threshold)
    inference = commands.add_parser("inference", help="сохранить рекомендации выбранной модели для пользователя")
    inference.add_argument("--model-dir", type=Path, required=True)
    inference.add_argument("--user-id", required=True)
    inference.add_argument("--k", type=_positive_int, default=10)
    inference.add_argument("--output", type=Path, required=True, help="CSV: user_id, rank, product_id")
    summary = commands.add_parser("summary", help="сводка завершённых шагов одного прогона: HTML, JSON, CSV")
    summary.add_argument("--run-dir", type=Path, required=True)
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
    cluster = commands.add_parser(
        "cluster", help="обучить кластеризацию и сохранить модель, расстояние и метки",
        description="Временной бейзлайн: равные последовательные группы по времени покупки.",
    )
    cluster.add_argument("--batch", type=Path, required=True, help="CSV; для temporal — накопленные данные")
    cluster.add_argument("--output-dir", type=Path, required=True, help="каталог модели данного шага")
    cluster.add_argument("--model", choices=["temporal"], default="temporal")
    cluster.add_argument("--distance", choices=["timestamp"], default="timestamp")
    cluster.add_argument(
        "--temporal-n-clusters", type=_positive_int,
        help="число групп temporal; по умолчанию определяется реализацией модели",
    )
    generate = commands.add_parser(
        "generate", help="создать и накопить модельные пользовательские истории",
        description="Новые истории из нового батча и готового расстояния; прежние истории сохраняются.",
    )
    generate.add_argument("--batch", type=Path, required=True, help="CSV только нового батча")
    generate.add_argument("--distance", type=Path, required=True, help="готовое расстояние в joblib")
    generate.add_argument("--output-dir", type=Path, required=True, help="новый каталог накопленного снимка")
    generate.add_argument("--previous-dir", type=Path, help="предыдущий снимок историй")
    generate.add_argument("--temperature", type=_positive_float, required=True, help="температура в единицах расстояния")
    generate.add_argument("--n-users", type=_positive_int, help="новые пользователи: по умолчанию 2000, затем 250")
    generate.add_argument(
        "--split-sizes", type=_positive_int, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST"),
        help="покупки на пользователя: сначала 70 15 15; затем наследуются из предыдущего снимка",
    )
    generate.add_argument("--random-state", type=_random_state, default=42)
    rank = commands.add_parser(
        "rank", help="обучить ранжировщик на train и сохранить модель",
        description="Random или SVD на накопленном train и полном каталоге товаров.",
    )
    rank.add_argument("--dataset-dir", type=Path, required=True, help="снимок модельных историй")
    rank.add_argument("--output-dir", type=Path, required=True, help="новый каталог модели")
    rank.add_argument("--model", choices=["random", "svd"], default="random")
    rank.add_argument("--random-state", type=_random_state, default=42)
    rank.add_argument(
        "--svd-n-components", type=_positive_int,
        help="число компонент SVD; по умолчанию определяется моделью (32)",
    )
    rank.add_argument(
        "--svd-n-iter", type=_positive_int,
        help="число итераций SVD; по умолчанию определяется моделью (7)",
    )
    ranking_evaluate = commands.add_parser(
        "evaluate-ranking", help="оценить сохранённый ранжировщик: Recall@K и NDCG@K",
        description="Независимая оценка выбранного split без обучения; JSON и CSV по пользователям.",
    )
    ranking_evaluate.add_argument("--dataset-dir", type=Path, required=True, help="снимок модельных историй")
    ranking_evaluate.add_argument("--model-dir", type=Path, required=True, help="каталог готового ранжировщика")
    ranking_evaluate.add_argument("--output-dir", type=Path, required=True, help="новый каталог оценки")
    ranking_evaluate.add_argument("--split", choices=["validation", "test"], required=True)
    ranking_evaluate.add_argument("--k", type=_positive_int, default=10)
    evaluate = commands.add_parser(
        "evaluate", help="оценить сохранённые метки: силуэт и отчёт с t-SNE",
        description="Независимая оценка без обучения: выполненный notebook и самодостаточный HTML.",
    )
    evaluate.add_argument("--dataset", type=Path, required=True, help="CSV всех накопленных данных")
    evaluate.add_argument("--new-batch", type=Path, required=True, help="CSV последнего добавленного батча")
    evaluate.add_argument("--model-dir", type=Path, required=True, help="каталог модели, расстояния и меток")
    evaluate.add_argument("--max-evaluation-rows", type=_positive_int, default=1000)
    evaluate.add_argument("--random-state", type=_random_state, default=42)
    args = parser.parse_args(argv)
    # These commands expose execution failures as exceptions, including in the
    # Python entry point. argparse still handles command-line syntax errors.
    if args.command == "run":
        options = {
            name: getattr(args, name) for name in PipelineConfig.__dataclass_fields__
            if name != "thresholds" and getattr(args, name) is not None
        }
        thresholds = {
            name: getattr(args, f"{name}_threshold") for name in ("price", "category", "state")
            if getattr(args, f"{name}_threshold") is not None
        }
        config = None
        if options or thresholds:
            base = read_pipeline_config(args.run_dir) if args.run_dir.exists() else PipelineConfig()
            if options.get("model", base.model) != "svd" and (
                args.svd_n_components is not None or args.svd_n_iter is not None
            ):
                parser.error("--svd-n-components и --svd-n-iter требуют --model svd")
            if "split_sizes" in options:
                options["split_sizes"] = tuple(options["split_sizes"])
            if thresholds:
                options["thresholds"] = replace(base.thresholds, **thresholds)
            config = replace(base, **options)
        results = run_pipeline(args.run_dir, data_dir=args.data_dir, config=config, all_batches=args.all)
        for result in results:
            print(f"Шаг {result.step_index:03d}: {result.manifest_path}")
        if not results:
            print("Поток завершён: новых батчей нет.")
        return
    if args.command == "inference":
        output = export_recommendations(args.model_dir, args.user_id, args.output, k=args.k)
        print(f"Рекомендации: {output}")
        return
    if args.command == "summary":
        paths = report_summary(args.run_dir)
        print(f"HTML: {paths.html_path}")
        print(f"JSON: {paths.json_path}")
        print(f"CSV: {paths.csv_path}")
        return
    if args.command == "rank" and args.model != "svd" and (
        args.svd_n_components is not None or args.svd_n_iter is not None
    ):
        parser.error("--svd-n-components и --svd-n-iter требуют --model svd")
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
        elif args.command == "cluster":
            distances = {"timestamp": TimestampDistance}
            temporal_options = {}
            if args.temporal_n_clusters is not None:
                temporal_options["n_clusters"] = args.temporal_n_clusters
            artifacts = train_clustering(
                args.batch, args.output_dir,
                strategy=partial(
                    train_temporal, distance=distances[args.distance](), **temporal_options,
                ),
            )
        elif args.command == "evaluate":
            report = report_clustering(
                args.dataset, args.new_batch, args.model_dir,
                max_evaluation_rows=args.max_evaluation_rows, random_state=args.random_state,
            )
        elif args.command == "generate":
            histories = generate_dataset(
                args.batch, args.distance, args.output_dir, temperature=args.temperature,
                previous_dir=args.previous_dir, n_users=args.n_users,
                split_sizes=args.split_sizes, random_state=args.random_state,
            )
        elif args.command == "rank":
            if args.model == "svd":
                svd_options = {}
                if args.svd_n_components is not None:
                    svd_options["n_components"] = args.svd_n_components
                if args.svd_n_iter is not None:
                    svd_options["n_iter"] = args.svd_n_iter
                ranker = SVDRanker(random_state=args.random_state, **svd_options)
            else:
                ranker = RandomRanker(random_state=args.random_state)
            ranking_model = train_ranker(
                args.dataset_dir, args.output_dir, ranker=ranker,
            )
        elif args.command == "evaluate-ranking":
            ranking_report = report_ranking(
                args.dataset_dir, args.model_dir, args.output_dir, split=args.split, k=args.k,
            )
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

    if args.command == "rank":
        print(f"Модель: {ranking_model.model_path}")
        print(f"Манифест: {ranking_model.manifest_path}")
        return

    if args.command == "evaluate-ranking":
        print(f"Метрики: {ranking_report.metrics_path}")
        print(f"По пользователям: {ranking_report.per_user_path}")
        return

    if args.command == "generate":
        print(f"Истории: {histories.output_dir}")
        print(f"Train: {histories.train_path}")
        print(f"Validation: {histories.validation_path}")
        print(f"Test: {histories.test_path}")
        print(f"Каталог: {histories.catalog_path}")
        print(f"Манифест генератора: {histories.manifest_path}")
        return

    if args.command == "cluster":
        print(f"Модель: {artifacts.model_path}")
        print(f"Расстояние: {artifacts.distance_path}")
        print(f"Метки: {artifacts.labels_path}")
        return

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
