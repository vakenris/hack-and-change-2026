"""Request budgets and diagnostics shared by the assistant and the batch runner."""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .contracts import TechnicalError

deadline: ContextVar[float | None] = ContextVar("deadline", default=None)
timings: ContextVar[dict[str, int] | None] = ContextVar("timings", default=None)


def remaining() -> float:
    end = deadline.get()
    if end is None:
        return float("inf")
    seconds = end - time.monotonic()
    if seconds <= 0:
        raise TechnicalError("Истёк бюджет времени обработки", code="deadline")
    return seconds


@contextmanager
def stage(name: str) -> Iterator[None]:
    start = time.monotonic()
    try:
        remaining()
        yield
    except TechnicalError as error:
        if error.stage is None:
            error.stage = name
        raise
    except Exception as error:
        raise TechnicalError(f"{name}: {type(error).__name__}", code="internal_error") from error
    finally:
        metrics = timings.get()
        if metrics is not None:
            metrics[name] = round((time.monotonic() - start) * 1000)
