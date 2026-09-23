"""Command-line entry point for the local Olist pipeline."""

import argparse
from collections.abc import Callable
from dataclasses import replace
from functools import partial
import os
from pathlib import Path
from types import TracebackType
from typing import Self

from buy_today.auto_eda import DriftThresholds, report_dataset, report_drift
from buy_today.cli_arguments import build_parser
from buy_today.cli_values import (
    Command,
    SVD,
    SVD_OPTIONS_ERROR,
    pipeline_options,
    threshold_options,
    svd_options_requested,
)
from buy_today.clustering.distances import TimestampDistance
from buy_today.clustering.models.temporal import train_temporal
from buy_today.clustering.report import report_clustering
from buy_today.clustering.training import train_clustering
from buy_today.generation import generate_dataset
from buy_today.preparation import prepare_data
from buy_today.pipeline import PipelineConfig, read_pipeline_config, run_pipeline
from buy_today.ranking import (
    RandomRanker,
    SVDRanker,
    export_recommendations,
    report_ranking,
    train_ranker,
    parse_model_spec,
    report_ranking_benchmark,
    run_ranking_benchmark,
)
from buy_today.summary import report_summary
from buy_today.update import initialize_reference, update_reference


class _CommandErrors:
    def __init__(self, parser: argparse.ArgumentParser, command: str) -> None:
        super().__init__()
        self.parser = parser
        self.command = command

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # The legacy command boundary translates every execution Exception to exit 1,
        # while SystemExit/KeyboardInterrupt retain their own semantics.
        if self.command in {Command.INFERENCE, Command.SUMMARY}:
            return
        if isinstance(exc_value, Exception):
            stage = "подготовки" if self.command == Command.PREPARE else self.command
            self.parser.exit(1, f"Ошибка {stage}: {exc_value}\n")


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    options = pipeline_options(args)
    thresholds = threshold_options(args)
    config = None
    if options or thresholds:
        run_dir: Path = args.run_dir
        base = read_pipeline_config(run_dir) if run_dir.exists() else PipelineConfig()
        if options.get("model", base.model) != SVD and svd_options_requested(args):
            parser.error(SVD_OPTIONS_ERROR)
        if thresholds:
            options["thresholds"] = replace(base.thresholds, **thresholds)
        config = replace(base, **options)
    results = run_pipeline(
        args.run_dir, data_dir=args.data_dir, config=config, all_batches=args.all
    )
    for result in results:
        print(f"Шаг {result.step_index:03d}: {result.manifest_path}")
    if not results:
        print("Поток завершён: новых батчей нет.")


def _inference(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        output = export_recommendations(args.model_dir, args.user_id, args.output, k=args.k)
    return [f"Рекомендации: {output}"]


def _summary(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        paths = report_summary(args.run_dir)
    return [f"HTML: {paths.html_path}", f"JSON: {paths.json_path}", f"CSV: {paths.csv_path}"]


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command == Command.RANK and args.model != SVD and svd_options_requested(args):
        parser.error(SVD_OPTIONS_ERROR)
    if args.command == Command.BENCHMARK:
        names = [parse_model_spec(spec).name for spec in args.model]
        if len(set(names)) != len(names):
            parser.error("повторяются канонические имена моделей")
        names = [Path(os.path.abspath(directory)).name for directory in args.dataset]
        if len(set(names)) != len(names):
            parser.error("повторяются basename датасетов")


def _prepare(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        result = prepare_data(
            args.raw_dir,
            args.output_dir,
            args.batch_size,
            min_category_count=args.min_category_count,
        )
    incomplete = result.rows_before_cleaning - result.rows_after_cleaning
    rare = result.rows_after_cleaning - result.rows_after_category_filtering
    return [
        f"Строк до очистки: {result.rows_before_cleaning:,}",
        f"Строк после очистки: {result.rows_after_cleaning:,}",
        f"Удалено неполных строк: {incomplete:,}",
        f"Удалено строк редких категорий: {rare:,}",
        f"Категорий: {result.categories_before_filtering} → {result.categories_after_filtering}",
        f"Строк после фильтрации категорий: {result.rows_after_category_filtering:,}",
        f"Батчей: {len(result.batch_sizes)}; размеры: {list(result.batch_sizes)}",
        f"Отброшенный из потока хвост: {result.dropped_tail_rows:,}",
        f"Рабочий датасет: {result.working_dataset_path}",
        f"Манифест: {result.manifest_path}",
    ]


def _eda(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        report = report_dataset(args.dataset, args.output_dir)
    return [f"Notebook: {report.notebook_path}", f"HTML: {report.html_path}"]


def _init(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        report = initialize_reference(args.batch, args.reference, args.output_dir)
    reference: Path = args.reference
    return [
        f"Эталон: {reference.resolve()}",
        f"Notebook: {report.notebook_path}",
        f"HTML: {report.html_path}",
    ]


def _cluster(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        distances = {"timestamp": TimestampDistance}
        temporal_options: dict[str, int] = {}
        if args.temporal_n_clusters is not None:
            temporal_options["n_clusters"] = args.temporal_n_clusters
        artifacts = train_clustering(
            args.batch,
            args.output_dir,
            strategy=partial(
                train_temporal, distance=distances[args.distance](), **temporal_options
            ),
        )
    return [
        f"Модель: {artifacts.model_path}",
        f"Расстояние: {artifacts.distance_path}",
        f"Метки: {artifacts.labels_path}",
    ]


def _evaluate(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        report = report_clustering(
            args.dataset,
            args.new_batch,
            args.model_dir,
            max_evaluation_rows=args.max_evaluation_rows,
            random_state=args.random_state,
        )
    return [f"Notebook: {report.notebook_path}", f"HTML: {report.html_path}"]


def _generate(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        histories = generate_dataset(
            args.batch,
            args.distance,
            args.output_dir,
            temperature=args.temperature,
            previous_dir=args.previous_dir,
            n_users=args.n_users,
            split_sizes=args.split_sizes,
            random_state=args.random_state,
        )
    return [
        f"Истории: {histories.output_dir}",
        f"Train: {histories.train_path}",
        f"Validation: {histories.validation_path}",
        f"Test: {histories.test_path}",
        f"Каталог: {histories.catalog_path}",
        f"Манифест генератора: {histories.manifest_path}",
    ]


def _rank(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        if args.model == SVD:
            svd_options: dict[str, int] = {}
            if args.svd_n_components is not None:
                svd_options["n_components"] = args.svd_n_components
            if args.svd_n_iter is not None:
                svd_options["n_iter"] = args.svd_n_iter
            ranker = SVDRanker(random_state=args.random_state, **svd_options)
        else:
            ranker = RandomRanker(random_state=args.random_state)
        model = train_ranker(args.dataset_dir, args.output_dir, ranker=ranker)
    return [f"Модель: {model.model_path}", f"Манифест: {model.manifest_path}"]


def _evaluate_ranking(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        report = report_ranking(
            args.dataset_dir, args.model_dir, args.output_dir, split=args.split, k=args.k
        )
    return [f"Метрики: {report.metrics_path}", f"По пользователям: {report.per_user_path}"]


def _benchmark(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        runs = run_ranking_benchmark(
            args.dataset,
            args.output,
            models=args.model,
            k=args.k,
            save_model=not args.no_save_model,
        )
    return [f"Run: {path}" for path in runs]


def _benchmark_report(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        comparison = report_ranking_benchmark(args.root, args.output)
    return [f"HTML: {comparison.html_path}", f"CSV: {comparison.csv_path}"]


def _deda(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        report = report_drift(
            args.batch,
            args.reference,
            args.output_dir,
            thresholds=DriftThresholds(**threshold_options(args)),
        )
    return [f"Notebook: {report.notebook_path}", f"HTML: {report.html_path}"]


def _update(args: argparse.Namespace, errors: _CommandErrors) -> list[str]:
    with errors:
        reports = update_reference(
            args.batch,
            args.reference,
            args.output_dir,
            thresholds=DriftThresholds(**threshold_options(args)),
        )
    reference: Path = args.reference
    lines = [f"Эталон: {reference.resolve()}"]
    for name, report in (("DEDA", reports.deda), ("EDA", reports.eda)):
        lines.extend(
            [f"{name} Notebook: {report.notebook_path}", f"{name} HTML: {report.html_path}"]
        )
    return lines


_HANDLERS: dict[str, Callable[[argparse.Namespace, _CommandErrors], list[str]]] = {
    Command.INFERENCE: _inference,
    Command.SUMMARY: _summary,
    Command.PREPARE: _prepare,
    Command.EDA: _eda,
    Command.INIT: _init,
    Command.CLUSTER: _cluster,
    Command.EVALUATE: _evaluate,
    Command.GENERATE: _generate,
    Command.RANK: _rank,
    Command.EVALUATE_RANKING: _evaluate_ranking,
    Command.BENCHMARK: _benchmark,
    Command.BENCHMARK_REPORT: _benchmark_report,
    Command.DEDA: _deda,
    Command.UPDATE: _update,
}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == Command.RUN:
        _run(args, parser)
        return
    _validate(args, parser)
    handler = _HANDLERS[args.command]
    lines = handler(args, _CommandErrors(parser, args.command))
    for line in lines:
        print(line)
