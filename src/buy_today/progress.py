"""Library progress events; applications configure handlers on ``buy_today``."""

from collections.abc import Generator
from contextlib import contextmanager, suppress
import logging
from time import perf_counter


# Suppress lastResort for unconfigured callers, while allowing root/caplog
# handlers to receive records through normal propagation.
logging.getLogger("buy_today").addHandler(logging.NullHandler())
_logger = logging.getLogger(__name__)


@contextmanager
def stage(name: str, **fields: object) -> Generator[None]:
    """Log INFO start/done or ERROR failed with elapsed seconds and traceback.

    Records expose ``stage``, ``event``, ``stage_fields`` and, on exit,
    ``elapsed_seconds``. Fields also appear in the message, so an ordinary
    logging formatter is sufficient. No levels or output handlers are set.
    A failed stage re-raises the original exception, even if error logging fails.

    Args:
        name (str): Stable name identifying the operation in progress records.
        **fields (object): Context values included in the message and record extras.

    Yields:
        None: Control to the operation being timed.

    Raises:
        BaseException: The original exception raised by the operation.
    """
    context = "".join(f" {key}={value!r}" for key, value in fields.items())
    extra: dict[str, object] = {"stage": name, "stage_fields": fields}
    _logger.info("stage=%s event=start%s", name, context, extra=extra | {"event": "start"})
    started = perf_counter()
    try:
        yield
    except BaseException:
        # Failure reporting is best-effort: preserve the operation's exception.
        with suppress(Exception):
            _log_failed(name, context, extra, started)
        raise
    elapsed = perf_counter() - started
    _logger.info(
        "stage=%s event=done elapsed_seconds=%.3f%s",
        name,
        elapsed,
        context,
        extra=extra | {"event": "done", "elapsed_seconds": elapsed},
    )


def _log_failed(name: str, context: str, extra: dict[str, object], started: float) -> None:
    elapsed = perf_counter() - started
    _logger.error(
        "stage=%s event=failed elapsed_seconds=%.3f%s",
        name,
        elapsed,
        context,
        extra=extra | {"event": "failed", "elapsed_seconds": elapsed},
        exc_info=True,
    )
