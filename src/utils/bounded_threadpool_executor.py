from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore, Lock, Condition


class BoundedExecutor:
    def __init__(self, max_workers=5, max_pending=10):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._semaphore = Semaphore(max_pending)
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._active_tasks = 0

    def submit(self, fn, *args, **kwargs):
        self._semaphore.acquire()

        def wrapped_fn(*args, **kwargs):
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

    def wait_for_all(self):
        """Block until all submitted tasks have completed."""
        with self._condition:
            while self._active_tasks > 0:
                self._condition.wait()

    def shutdown(self, wait=True):
        """Shut down the underlying ThreadPoolExecutor."""
        self._executor.shutdown(wait=wait)