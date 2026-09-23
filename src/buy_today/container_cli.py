"""Three-command container interface with a persistent /workspace bind mount."""

import argparse
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
import sys
from typing import Never, override
from uuid import uuid4

from buy_today.auto_eda import DriftThresholds
from buy_today.bundled_data import OLIST_FILES
from buy_today.cli_arguments import ParserFactory, command_factory
from buy_today.cli_values import (
    Command,
    SVD,
    SVD_OPTIONS_ERROR,
    drift_threshold,
    positive_float,
    positive_int,
    random_state,
    pipeline_options,
    threshold_options,
    svd_options_requested,
)
from buy_today.pipeline import PipelineConfig, StepResult, run_next_batch
from buy_today.preparation import prepare_data
from buy_today.progress import stage
from buy_today.ranking import export_recommendations
from buy_today.summary import SummaryPaths, report_summary


WORKSPACE = Path("/workspace")
BUNDLED_DATA = Path("/opt/buy_today/olist")
# python -m executes this module as __main__; keep events in the configured tree.
_logger = logging.getLogger("buy_today.container_cli")
_VERBOSE = "--verbose"
_HELP_FLAGS = {"--help", "-h"}
_COMMAND_NAMES = {Command.INIT, Command.UPDATE, Command.INFERENCE}


def _invocation_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]


def _directory(path: Path) -> Path:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"Expected a real workspace directory, not a file or symlink: {path}")
    return path


def _log_handlers(path: Path, command: str, verbose: bool) -> list[logging.Handler]:
    file_handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    formatter = logging.Formatter(
        f"%(asctime)s %(levelname)s command={command} %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    for handler in (file_handler, console):
        handler.setFormatter(formatter)
    return [file_handler, console]


@contextmanager
def _log_events(workspace: Path, path: Path) -> Generator[None]:
    try:
        yield from _started_log(workspace, path)
    except BaseException:
        _logger.exception("event=command_failed")
        raise
    _logger.info("event=command_succeeded")


def _started_log(workspace: Path, path: Path) -> Generator[None]:
    print(f"Лог: {path}", flush=True)
    _logger.info("event=command_start workspace=%s", workspace)
    yield


@contextmanager
def _command_log(workspace: Path, command: str, invocation: str, verbose: bool) -> Generator[None]:
    directory = _directory(workspace / "logs")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{invocation}_{command}.log"
    handlers = _log_handlers(path, command, verbose)
    with _installed_handlers(logging.getLogger("buy_today"), handlers):
        with _log_events(workspace, path):
            yield


@contextmanager
def _installed_handlers(logger: logging.Logger, handlers: list[logging.Handler]) -> Generator[None]:
    previous = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = handlers
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield
    finally:
        logger.handlers, previous_level, logger.propagate = previous
        logger.setLevel(previous_level)
        for handler in handlers:
            handler.close()


class _ArgumentParser(argparse.ArgumentParser):
    @override
    def error(self, message: str) -> Never:
        _logger.error("Invalid arguments: %s", message)
        super().error(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="buy_today-container",
        description="Olist в Docker: init, update, inference; результаты в /workspace.",
    )
    _command_arguments(command_factory(parser))
    return parser


def _command_arguments(add_parser: ParserFactory) -> None:
    init = add_parser("init", "сбросить эксперимент и обработать первый батч", None)
    update = add_parser("update", "обработать следующий батч и обновить сводку", None)
    inference = add_parser("inference", "рекомендации явно выбранной сохранённой модели", None)
    for command in (init, update, inference):
        _ = command.add_argument(
            "--verbose", action="store_true", help="показывать текущие стадии и время выполнения"
        )
    _init_arguments(init)
    _inference_arguments(inference)


def _init_arguments(init: argparse.ArgumentParser) -> None:
    _ = init.add_argument("--batch-size", type=positive_int, default=5000)
    _ = init.add_argument("--min-category-count", type=positive_int, default=1000)
    _ = init.add_argument("--model", choices=["random", "svd"], default="svd")
    _ = init.add_argument("--random-state", type=random_state)
    _ = init.add_argument("--temperature", type=positive_float)
    for name in (
        "initial-users",
        "additional-users",
        "k",
        "svd-n-components",
        "svd-n-iter",
        "temporal-n-clusters",
        "max-evaluation-rows",
    ):
        _ = init.add_argument(f"--{name}", type=positive_int)
    _ = init.add_argument(
        "--split-sizes", type=positive_int, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST")
    )
    for feature in ("price", "category", "state"):
        _ = init.add_argument(f"--{feature}-threshold", type=drift_threshold)


def _inference_arguments(inference: argparse.ArgumentParser) -> None:
    _ = inference.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="каталог модели; относительный путь от /workspace",
    )
    _ = inference.add_argument("--user-id", required=True, help="известный модельный user_id")
    _ = inference.add_argument("--k", type=positive_int, default=10)
    _ = inference.add_argument(
        "--output",
        type=Path,
        help="CSV внутри /workspace/recommendations; относительный путь от этого каталога",
    )


def _settings(args: argparse.Namespace) -> PipelineConfig:
    if args.model != SVD and svd_options_requested(args):
        raise ValueError(SVD_OPTIONS_ERROR)
    options = pipeline_options(args)
    thresholds = threshold_options(args)
    settings = PipelineConfig(**options, thresholds=DriftThresholds(**thresholds))
    if settings.model == SVD and settings.svd_n_components > settings.initial_users:
        raise ValueError("--svd-n-components must not exceed --initial-users")
    return settings


def _validate_export(workspace: Path) -> None:
    # Validate every source and destination before resetting an existing run.
    for filename in OLIST_FILES:
        source = BUNDLED_DATA / filename
        if not source.is_file() or source.is_symlink() or not source.stat().st_size:
            raise ValueError(f"Missing or empty bundled Olist CSV: {source}")
    for name in ("dataset", "data", "run", "recommendations"):
        _ = _directory(workspace / name)
    for filename in OLIST_FILES:
        target = workspace / "dataset" / filename
        if target.exists() and not target.is_file() and not target.is_symlink():
            raise ValueError(
                f"Expected a CSV destination, not a directory or special file: {target}"
            )


def _reset_and_export(workspace: Path) -> None:
    _validate_export(workspace)
    with stage("reset", workspace=workspace):
        for name in ("data", "run", "recommendations"):
            directory = workspace / name
            if directory.exists():
                shutil.rmtree(directory)
    with stage("export_olist", source=BUNDLED_DATA, destination=workspace / "dataset"):
        raw = workspace / "dataset"
        raw.mkdir(exist_ok=True)
        for filename in OLIST_FILES:
            target = raw / filename
            # Replace the directory entry instead of following an existing alias.
            target.unlink(missing_ok=True)
            _ = shutil.copyfile(BUNDLED_DATA / filename, target)


def _summary_report(run_dir: Path) -> SummaryPaths:
    with stage("summary", run_dir=run_dir):
        return report_summary(run_dir)


def _summarize(run_dir: Path, completed: StepResult | None) -> SummaryPaths:
    try:
        result = _summary_report(run_dir)
    except Exception:
        if completed is not None:
            _logger.error(
                "Батч %03d завершён и сохранён; ошибка при сборке сводки. %s",
                completed.step_index,
                "Следующий update обработает следующий батч.",
            )
        raise
    return result


def _inference(args: argparse.Namespace, workspace: Path, invocation: str) -> None:
    directory = _directory(workspace / "recommendations")
    destination: Path = args.output or Path(f"{invocation}.csv")
    output = (directory / destination).resolve()
    _validate_output(directory, output)
    model = (workspace / args.model_dir).resolve()
    with stage("inference", model_dir=model, user_id=args.user_id, k=args.k):
        result = export_recommendations(model, args.user_id, output, k=args.k)
    print(f"Рекомендации: {result}", flush=True)


def _validate_output(directory: Path, output: Path) -> None:
    if directory not in output.parents:
        raise ValueError("--output must be inside /workspace/recommendations")
    if output.exists() and output.stat().st_nlink > 1:
        raise ValueError("--output must not overwrite a file alias")


def _initialize(args: argparse.Namespace, workspace: Path, run_dir: Path) -> StepResult | None:
    config = _settings(args)
    _reset_and_export(workspace)
    with stage("preparation"):
        prepared = prepare_data(
            workspace / "dataset",
            workspace / "data",
            args.batch_size,
            min_category_count=args.min_category_count,
        )
        _logger.info(
            "Prepared batches=%s rows=%d",
            prepared.batch_sizes,
            prepared.rows_after_category_filtering,
        )
    return run_next_batch(run_dir, data_dir=workspace / "data", config=config)


def _advance(args: argparse.Namespace, workspace: Path) -> None:
    run_dir = _directory(workspace / "run")
    # No implicit initialization or source export during update.
    completed = (
        _initialize(args, workspace, run_dir)
        if args.command == Command.INIT
        else run_next_batch(run_dir)
    )
    summary = _summarize(run_dir, completed)
    if completed is None:
        print("Поток завершён: новых батчей нет. Сводка обновлена.", flush=True)
    else:
        model_dir = completed.output_dir / "ranking"
        print(f"Шаг {completed.step_index:03d} завершён. Модель: {model_dir}", flush=True)
    print(f"Сводка: {summary.html_path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if _HELP_FLAGS.intersection(argv):
        _ = parser.parse_args(argv)
        return
    workspace = WORKSPACE.resolve()
    invocation = _invocation_id()
    command = argv[0] if argv and argv[0] in _COMMAND_NAMES else "cli"
    with _command_log(workspace, command, invocation, _VERBOSE in argv):
        args = parser.parse_args(argv)
        if args.command == Command.INFERENCE:
            _inference(args, workspace, invocation)
        else:
            _advance(args, workspace)


if __name__ == "__main__":
    main()
