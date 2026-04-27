"""Error mapping helpers.

The OpenAI Agents SDK requires sandbox provider implementations to raise its
canonical error types from ``agents.sandbox.errors`` (keyword-arg-only, with
``cause=`` carrying the original exception). This module centralizes the
mapping so individual call-sites stay thin.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from agents.sandbox.errors import (
    ExecTransportError,
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceStartError,
    WorkspaceStopError,
    WorkspaceWriteTypeError,
)

# BoxLite-core exception names that mean "thing not found" regardless of context.
_NOT_FOUND_HINTS = ("notfound", "not_found", "no such", "missing")


def _looks_like_not_found(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return any(hint in msg or hint in name for hint in _NOT_FOUND_HINTS)


@contextmanager
def map_read_errors(path: Path) -> Iterator[None]:
    """Translate BoxLite read-side failures into the SDK contract."""
    try:
        yield
    except FileNotFoundError as e:
        raise WorkspaceReadNotFoundError(path=path) from e
    except OSError as e:
        raise WorkspaceArchiveReadError(path=path, cause=e) from e
    except Exception as e:
        if _looks_like_not_found(e):
            raise WorkspaceReadNotFoundError(path=path) from e
        raise WorkspaceArchiveReadError(path=path, cause=e) from e


@contextmanager
def map_write_errors(path: Path, *, context: str = "write") -> Iterator[None]:
    """Translate BoxLite write-side failures into the SDK contract."""
    try:
        yield
    except TypeError as e:
        raise WorkspaceWriteTypeError(path=path, actual_type=type(e).__name__) from e
    except Exception as e:
        raise WorkspaceArchiveWriteError(path=path, context=context, cause=e) from e


@contextmanager
def map_exec_transport_errors(command: Sequence[object]) -> Iterator[None]:
    """Wrap arbitrary BoxLite/FFI failures during exec setup as
    ``ExecTransportError``. Process-level failures (non-zero exit, timeout) are
    handled separately at call-sites and must not flow through here.
    """
    try:
        yield
    except Exception as e:
        raise ExecTransportError(command=tuple(command), cause=e) from e


@contextmanager
def map_start_errors(path: Path) -> Iterator[None]:
    try:
        yield
    except WorkspaceStartError:
        raise
    except Exception as e:
        raise WorkspaceStartError(path=path) from e


@contextmanager
def map_stop_errors(path: Path) -> Iterator[None]:
    try:
        yield
    except WorkspaceStopError:
        raise
    except Exception as e:
        raise WorkspaceStopError(path=path) from e
