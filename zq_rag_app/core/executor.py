"""索引任务线程池。"""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore
from typing import Any, Callable

from .config import settings


class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    """带有界等待队列的线程池，避免任务无限堆积。"""

    def __init__(self, max_workers: int, queue_capacity: int, **kwargs: Any):
        # 初始化线程池，并创建一个有界信号量来限制队列容量
        super().__init__(max_workers=max_workers, **kwargs)
        # BoundedSemaphore 用于限制同时提交的任务数量，防止队列无限增长。
        self._slots = BoundedSemaphore(max_workers + queue_capacity)

# 重写 submit 方法，在提交任务前尝试获取信号量，如果队列已满则抛出 RuntimeError。
    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        # 尝试获取信号量，如果队列已满则抛出 RuntimeError。   信号量的获取是非阻塞的，如果无法获取则立即抛出异常，避免任务无限堆积。
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("索引任务队列已满")

# 提交任务给父类的 submit 方法，并在任务完成后释放信号量。   这样可以确保在任务执行完毕后，信号量被释放，从而允许新的任务提交。
        try:
            future = super().submit(fn, *args, **kwargs)
        except BaseException:
            self._slots.release()
            raise

        future.add_done_callback(lambda _: self._slots.release())
        return future


index_executor = BoundedThreadPoolExecutor(
    max_workers=settings.task_max_workers,
    queue_capacity=settings.task_queue_capacity,
    thread_name_prefix="index-",
)


def submit_index_task(
    fn: Callable[..., Any], /, *args: Any, **kwargs: Any
) -> Future:
    """提交一个索引任务；队列满时抛出 RuntimeError。"""

    return index_executor.submit(fn, *args, **kwargs)


def shutdown_index_executor() -> None:
    """应用关闭时等待并释放索引线程池。"""

    index_executor.shutdown(wait=True, cancel_futures=False)
