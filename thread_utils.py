"""Safe stopping of background QThreads (fixed infrastructure).

Why this file exists: a QThread must never be destroyed while its `run()` is still
executing - Qt then aborts the whole program with
"QThread: Destroyed while thread is still running". That used to happen when the user
switched between steps quickly: a thread that was still opening the webcam was
deleted before it noticed the stop request.

The rule used everywhere in this project:

1. Never block the GUI waiting for a thread (no long `wait()` in a button handler).
2. Ask the thread to stop with `request_stop()` and hand it to `retire_thread()`.
3. `retire_thread()` keeps the thread alive until it has really finished, and only then
   deletes it.
4. A new thread that needs the same camera is started with `call_when_idle()`, so it
   starts only after the old threads released the device.
"""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QThread

_RETIRING: set[QThread] = set()  # strong references: keeps the Python/C++ objects alive
_WAITERS: list[Callable[[], None]] = []


def retire_thread(thread: QThread) -> None:
    """Stop `thread` without blocking; it is deleted once it has really finished."""
    if thread is None:
        return
    thread.request_stop()
    if thread.isFinished() or not thread.isRunning():
        thread.deleteLater()
        _run_waiters()
        return
    thread.setParent(None)  # a parent would delete the thread together with itself
    _RETIRING.add(thread)

    def _cleanup() -> None:
        _RETIRING.discard(thread)
        thread.deleteLater()
        _run_waiters()

    thread.finished.connect(_cleanup)


def call_when_idle(callback: Callable[[], None]) -> None:
    """Run `callback` now if no thread is still shutting down, otherwise right after."""
    if not _RETIRING:
        callback()
    else:
        _WAITERS.append(callback)


def _run_waiters() -> None:
    if _RETIRING:
        return
    waiters = _WAITERS[:]
    _WAITERS.clear()
    for callback in waiters:
        callback()


def wait_for_retired_threads(timeout_ms: int = 8000) -> None:
    """Call when the application closes: let every stopping thread finish."""
    for thread in list(_RETIRING):
        if not thread.wait(timeout_ms):
            thread.terminate()  # last resort, only while the program exits
            thread.wait(1000)
    _RETIRING.clear()
    _WAITERS.clear()
