from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Condition, Lock, Semaphore
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class BoundedExecutor:
    def __init__(self, max_workers: int = 5, max_pending: int = 10) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._semaphore = Semaphore(max_pending)
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._active_tasks = 0

    def submit(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        self._semaphore.acquire()

        def wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> R:
            with self._condition:
                self._active_tasks += 1
            try:
                return fn(*args, **kwargs)
            finally:
                with self._condition:
                    self._active_tasks -= 1
                    self._condition.notify_all()
                self._semaphore.release()

        return self._executor.submit(wrapped_fn, *args, **kwargs)

    def wait_for_all(self) -> None:
        """Block until all submitted tasks have completed."""
        with self._condition:
            while self._active_tasks > 0:
                self._condition.wait()

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the underlying ThreadPoolExecutor."""
        self._executor.shutdown(wait=wait)
