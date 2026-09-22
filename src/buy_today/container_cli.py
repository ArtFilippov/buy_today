"""Three-command container interface with a persistent /workspace bind mount."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
import sys
from uuid import uuid4

from buy_today.auto_eda import DriftThresholds
from buy_today.bundled_data import OLIST_FILES
from buy_today.cli import _drift_threshold, _positive_float, _positive_int, _random_state
from buy_today.pipeline import PipelineConfig, run_next_batch
from buy_today.preparation import prepare_data
from buy_today.progress import stage
from buy_today.ranking import export_recommendations
from buy_today.summary import report_summary


WORKSPACE = Path("/workspace")
BUNDLED_DATA = Path("/opt/buy_today/olist")
# python -m executes this module as __main__; keep events in the configured tree.
_logger = logging.getLogger("buy_today.container_cli")


def _invocation_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]


def _directory(path):
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"Expected a real workspace directory, not a file or symlink: {path}")
    return path


@contextmanager
def _command_log(workspace, command, invocation, verbose):
    directory = _directory(workspace / "logs")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{invocation}_{command}.log"
    file_handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    formatter = logging.Formatter(
        f"%(asctime)s %(levelname)s command={command} %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    for handler in (file_handler, console):
        handler.setFormatter(formatter)
    logger = logging.getLogger("buy_today")
    previous = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = [file_handler, console]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        print(f"Лог: {path}", flush=True)
        _logger.info("event=command_start workspace=%s", workspace)
        yield
    except BaseException:
        _logger.exception("event=command_failed")
        raise
    else:
        _logger.info("event=command_succeeded")
    finally:
        logger.handlers, previous_level, logger.propagate = previous
        logger.setLevel(previous_level)
        file_handler.close()
        console.close()


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        _logger.error("Invalid arguments: %s", message)
        super().error(message)


def _parser():
    parser = _ArgumentParser(
        prog="buy_today-container", description="Olist в Docker: init, update, inference; результаты в /workspace.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="сбросить эксперимент и обработать первый батч")
    update = commands.add_parser("update", help="обработать следующий батч и обновить сводку")
    inference = commands.add_parser("inference", help="рекомендации явно выбранной сохранённой модели")
    for command in (init, update, inference):
        command.add_argument("--verbose", action="store_true", help="показывать текущие стадии и время выполнения")
    init.add_argument("--batch-size", type=_positive_int, default=5000)
    init.add_argument("--min-category-count", type=_positive_int, default=1000)
    init.add_argument("--model", choices=["random", "svd"], default="svd")
    init.add_argument("--random-state", type=_random_state)
    init.add_argument("--temperature", type=_positive_float)
    for name in (
        "initial-users", "additional-users", "k", "svd-n-components", "svd-n-iter",
        "temporal-n-clusters", "max-evaluation-rows",
    ):
        init.add_argument(f"--{name}", type=_positive_int)
    init.add_argument("--split-sizes", type=_positive_int, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST"))
    for feature in ("price", "category", "state"):
        init.add_argument(f"--{feature}-threshold", type=_drift_threshold)
    inference.add_argument("--model-dir", type=Path, required=True, help="каталог модели; относительный путь от /workspace")
    inference.add_argument("--user-id", required=True, help="известный модельный user_id")
    inference.add_argument("--k", type=_positive_int, default=10)
    inference.add_argument(
        "--output", type=Path, help="CSV внутри /workspace/recommendations; относительный путь от этого каталога",
    )
    return parser


def _settings(args):
    if args.model != "svd" and (args.svd_n_components is not None or args.svd_n_iter is not None):
        raise ValueError("--svd-n-components и --svd-n-iter требуют --model svd")
    options = {
        name: getattr(args, name) for name in PipelineConfig.__dataclass_fields__
        if name != "thresholds" and getattr(args, name) is not None
    }
    if "split_sizes" in options:
        options["split_sizes"] = tuple(options["split_sizes"])
    thresholds = {
        name: getattr(args, f"{name}_threshold") for name in ("price", "category", "state")
        if getattr(args, f"{name}_threshold") is not None
    }
    settings = PipelineConfig(**options, thresholds=DriftThresholds(**thresholds))
    if settings.model == "svd" and settings.svd_n_components > settings.initial_users:
        raise ValueError("--svd-n-components must not exceed --initial-users")
    return settings


def _reset_and_export(workspace):
    # Validate every source and destination before resetting an existing run.
    for filename in OLIST_FILES:
        source = BUNDLED_DATA / filename
        if not source.is_file() or source.is_symlink() or source.stat().st_size == 0:
            raise ValueError(f"Missing or empty bundled Olist CSV: {source}")
    for name in ("dataset", "data", "run", "recommendations"):
        _directory(workspace / name)
    for filename in OLIST_FILES:
        target = workspace / "dataset" / filename
        if target.exists() and not target.is_file() and not target.is_symlink():
            raise ValueError(f"Expected a CSV destination, not a directory or special file: {target}")
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
            shutil.copyfile(BUNDLED_DATA / filename, target)


def _summarize(run_dir, completed):
    try:
        with stage("summary", run_dir=run_dir):
            result = report_summary(run_dir)
    except Exception:
        if completed is not None:
            _logger.error(
                "Батч %03d завершён и сохранён; ошибка при сборке сводки. "
                "Следующий update обработает следующий батч.", completed.step_index,
            )
        raise
    return result


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if "--help" in argv or "-h" in argv:
        parser.parse_args(argv)
        return
    workspace = WORKSPACE.resolve()
    invocation = _invocation_id()
    command = argv[0] if argv and argv[0] in ("init", "update", "inference") else "cli"
    with _command_log(workspace, command, invocation, "--verbose" in argv):
        args = parser.parse_args(argv)
        if args.command == "inference":
            directory = _directory(workspace / "recommendations")
            output = args.output or Path(f"{invocation}.csv")
            output = (directory / output).resolve()
            if directory not in output.parents:
                raise ValueError("--output must be inside /workspace/recommendations")
            if output.exists() and output.stat().st_nlink > 1:
                raise ValueError("--output must not overwrite a file alias")
            model = (workspace / args.model_dir).resolve()
            with stage("inference", model_dir=model, user_id=args.user_id, k=args.k):
                result = export_recommendations(model, args.user_id, output, k=args.k)
            print(f"Рекомендации: {result}", flush=True)
            return

        run_dir = _directory(workspace / "run")
        if args.command == "init":
            config = _settings(args)
            _reset_and_export(workspace)
            with stage("preparation"):
                prepared = prepare_data(
                    workspace / "dataset", workspace / "data", args.batch_size,
                    min_category_count=args.min_category_count,
                )
                _logger.info("Prepared batches=%s rows=%d", prepared.batch_sizes, prepared.rows_after_category_filtering)
            completed = run_next_batch(run_dir, data_dir=workspace / "data", config=config)
        else:
            # No implicit initialization or source export during update.
            completed = run_next_batch(run_dir)
        summary = _summarize(run_dir, completed)
        if completed is None:
            print("Поток завершён: новых батчей нет. Сводка обновлена.", flush=True)
        else:
            print(f"Шаг {completed.step_index:03d} завершён. Модель: {completed.output_dir / 'ranking'}", flush=True)
        print(f"Сводка: {summary.html_path}", flush=True)


if __name__ == "__main__":
    main()
