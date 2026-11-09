from __future__ import annotations

import os
import re
import select
import threading
import warnings
from contextlib import contextmanager
from typing import Callable, TypeVar


SKLEARN_PARALLEL_DELAYED_WARNING = (
    r".*(?:"
    r"`sklearn\.utils\.parallel\.Parallel` needs to be used in conjunction with "
    r"`sklearn\.utils\.parallel\.delayed` instead of `joblib\.delayed`"
    r"|"
    r"`sklearn\.utils\.parallel\.delayed` should be used with "
    r"`sklearn\.utils\.parallel\.Parallel` to make it possible to propagate the "
    r"scikit-learn configuration of the current thread to the joblib workers"
    r").*"
)

SKLEARN_PARALLEL_DELAYED_PYTHONWARNINGS = (
    "ignore:`sklearn.utils.parallel.Parallel` needs to be used in conjunction with "
    "`sklearn.utils.parallel.delayed` instead of `joblib.delayed`:UserWarning",
    "ignore:`sklearn.utils.parallel.delayed` should be used with "
    "`sklearn.utils.parallel.Parallel` to make it possible to propagate the "
    "scikit-learn configuration of the current thread to the joblib workers:UserWarning",
)

_APPLICATION_STDERR_FILTER_ACTIVE = "BIOMARKERML_STDERR_FILTER_ACTIVE"
_SKLEARN_PARALLEL_WARNING_HEADER = re.compile(r"sklearn/utils/parallel\.py:\d+: UserWarning:")
_TARGET_WARNING_BODIES = (
    "`sklearn.utils.parallel.Parallel` needs to be used in conjunction with "
    "`sklearn.utils.parallel.delayed` instead of `joblib.delayed`",
    "`sklearn.utils.parallel.delayed` should be used with "
    "`sklearn.utils.parallel.Parallel` to make it possible to propagate the "
    "scikit-learn configuration of the current thread to the joblib workers",
)
T = TypeVar("T")


def install_runtime_warning_filters() -> None:
    """Install only the sklearn nested-Parallel warning filters for this process tree."""
    warnings.filterwarnings(
        "ignore",
        message=SKLEARN_PARALLEL_DELAYED_WARNING,
        category=UserWarning,
    )
    entries = [entry for entry in os.environ.get("PYTHONWARNINGS", "").split(",") if entry]
    for rule in SKLEARN_PARALLEL_DELAYED_PYTHONWARNINGS:
        if rule not in entries:
            entries.append(rule)
    os.environ["PYTHONWARNINGS"] = ",".join(entries)


def _is_target_parallel_warning(record: list[str]) -> bool:
    normalized = " ".join("".join(record).split())
    return any(body in normalized for body in _TARGET_WARNING_BODIES)


def _relay_filtered_stderr(read_fd: int, stop_fd: int, output_fd: int) -> None:
    """Forward filtered stderr until EOF or an explicit final-drain signal.

    ``read_fd`` belongs to this relay. ``stop_fd`` belongs to the caller and is
    deliberately left open until the caller has joined this thread.
    """
    pending = ""
    record: list[str] | None = None

    def emit(text: str) -> None:
        payload = memoryview(text.encode(errors="replace"))
        while payload:
            written = os.write(output_fd, payload)
            payload = payload[written:]

    def finish_record() -> None:
        nonlocal record
        if record is not None:
            if not _is_target_parallel_warning(record):
                emit("".join(record))
            record = None

    def consume(chunk: bytes) -> None:
        nonlocal pending, record
        pending += chunk.decode(errors="replace")
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            line += "\n"
            if record is None:
                if _SKLEARN_PARALLEL_WARNING_HEADER.search(line):
                    record = [line]
                else:
                    emit(line)
            else:
                record.append(line)
                if "warnings.warn(" in line:
                    finish_record()
                elif _SKLEARN_PARALLEL_WARNING_HEADER.search(line):
                    previous = record[:-1]
                    record = [line]
                    if not _is_target_parallel_warning(previous):
                        emit("".join(previous))

    def finish_pending() -> None:
        nonlocal pending
        if pending:
            if record is None:
                emit(pending)
            else:
                record.append(pending)
            pending = ""
        finish_record()

    try:
        while True:
            ready, _, _ = select.select((read_fd, stop_fd), (), ())
            if stop_fd in ready:
                os.read(stop_fd, 1)
                os.set_blocking(read_fd, False)
                while True:
                    try:
                        chunk = os.read(read_fd, 65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    consume(chunk)
                finish_pending()
                return
            chunk = os.read(read_fd, 65536)
            if not chunk:
                finish_pending()
                return
            consume(chunk)
    finally:
        os.close(read_fd)


@contextmanager
def filter_runtime_stderr_records():
    """Relay inherited stderr while removing only the two sklearn warning records."""
    if os.environ.get(_APPLICATION_STDERR_FILTER_ACTIVE) == "1":
        yield
        return

    original_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    stop_read_fd, stop_write_fd = os.pipe()
    previous = os.environ.get(_APPLICATION_STDERR_FILTER_ACTIVE)
    os.environ[_APPLICATION_STDERR_FILTER_ACTIVE] = "1"
    os.dup2(write_fd, 2)
    os.close(write_fd)
    relay = threading.Thread(
        target=_relay_filtered_stderr,
        args=(read_fd, stop_read_fd, original_fd),
        name="_relay_filtered_stderr",
    )
    relay.start()
    try:
        yield
    finally:
        # Restoring fd 2 closes the parent pipe writer. Children may still
        # hold inherited writers, so final draining cannot depend on pipe EOF.
        os.dup2(original_fd, 2)
        os.write(stop_write_fd, b"1")
        os.close(stop_write_fd)
        relay.join()
        os.close(stop_read_fd)
        os.close(original_fd)
        if previous is None:
            os.environ.pop(_APPLICATION_STDERR_FILTER_ACTIVE, None)
        else:
            os.environ[_APPLICATION_STDERR_FILTER_ACTIVE] = previous


def run_with_runtime_output_filter(main_function: Callable[[], T]) -> T:
    """Initialize runtime filtering before application work or worker creation."""
    install_runtime_warning_filters()
    with filter_runtime_stderr_records():
        return main_function()
