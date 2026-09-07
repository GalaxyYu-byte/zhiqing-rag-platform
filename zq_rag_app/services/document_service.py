"""文档服务中的索引任务提交入口。"""

from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from ..core.executor import submit_index_task


def submit_document_index(
    index_callable: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Future[Any]:
    """将一个文档索引函数提交到专用索引线程池。

    文档上传、解析和 Embedding 逻辑由调用方提供；所有索引任务统一从
    这里进入有界线程池，队列满时会抛出 RuntimeError。
    """

    return submit_index_task(index_callable, *args, **kwargs)
