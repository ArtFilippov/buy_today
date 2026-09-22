"""Library progress events; applications configure handlers on ``buy_today``."""

from collections.abc import Iterator
from contextlib import contextmanager
import logging
from time import perf_counter


# Suppress lastResort for unconfigured callers, while allowing root/caplog
# handlers to receive records through normal propagation.
logging.getLogger("buy_today").addHandler(logging.NullHandler())
_logger = logging.getLogger(__name__)


@contextmanager
def stage(name: str, **fields) -> Iterator[None]:
    """Log INFO start/done or ERROR failed with elapsed seconds and traceback.

    Records expose ``stage``, ``event``, ``stage_fields`` and, on exit,
    ``elapsed_seconds``. Fields also appear in the message, so an ordinary
    logging formatter is sufficient. No levels or output handlers are set.
    A failed stage re-raises the original exception, even if error logging fails.
    """
    context = "".join(f" {key}={value!r}" for key, value in fields.items())
    extra = {"stage": name, "stage_fields": fields}
    _logger.info("stage=%s event=start%s", name, context, extra=extra | {"event": "start"})
    started = perf_counter()
    try:
        yield
    except BaseException:
        try:
            elapsed = perf_counter() - started
            _logger.error(
                "stage=%s event=failed elapsed_seconds=%.3f%s", name, elapsed, context,
                extra=extra | {"event": "failed", "elapsed_seconds": elapsed}, exc_info=True,
            )
        except Exception:
            # A broken log destination must not replace the operation's error.
            pass
        raise
    else:
        elapsed = perf_counter() - started
        _logger.info(
            "stage=%s event=done elapsed_seconds=%.3f%s", name, elapsed, context,
            extra=extra | {"event": "done", "elapsed_seconds": elapsed},
        )
