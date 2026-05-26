"""Helpers for benchmark CLI error handling."""

from __future__ import annotations

import logging
from typing import Callable, NoReturn, TypeVar

T = TypeVar("T")


def log_and_exit(
    logger: logging.Logger,
    message: str,
    *,
    exit_code: int,
) -> NoReturn:
    """Log a user-facing benchmark error and terminate."""
    logger.error(message)
    raise SystemExit(exit_code)


def require_or_exit(
    condition: bool,
    logger: logging.Logger,
    message: str,
    *,
    exit_code: int,
) -> None:
    """Exit with a logged error when a benchmark precondition is not met."""
    if not condition:
        log_and_exit(logger, message, exit_code=exit_code)


def run_or_exit(
    callback: Callable[[], T],
    logger: logging.Logger,
    *,
    exit_code: int,
    exception_types: tuple[type[Exception], ...] = (ValueError,),
) -> T:
    """Run a benchmark CLI step and convert expected failures into clean exits."""
    try:
        return callback()
    except exception_types as error:
        log_and_exit(logger, str(error), exit_code=exit_code)
