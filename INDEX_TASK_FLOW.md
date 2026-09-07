# 索引任务线程池流程说明

## 1. 目的

文档上传后，解析、文本切分、Embedding 和向量入库都可能耗时较长。如果在 HTTP 请求中同步执行，接口会长时间占用请求线程。

当前项目使用独立的索引线程池处理这些任务，使上传接口能够快速返回“处理中”。

整体调用链如下：

```text
FastAPI 上传接口
        |
        v
submit_document_index()
        |
        v
submit_index_task()
        |
        v
index_executor.submit()
        |
        v
索引工作线程执行 build_document_index()
```

相关文件：

| 文件 | 作用 |
|---|---|
| `zq_rag_app/core/config.py` | 定义线程池参数 |
| `zq_rag_app/core/executor.py` | 创建有界索引线程池 |
| `zq_rag_app/services/document_service.py` | 提供业务层索引任务提交入口 |
| `zq_rag_app/main.py` | 应用关闭时释放线程池 |

## 2. 配置参数

配置定义在 `zq_rag_app/core/config.py`，实际值来自项目根目录的 `.env`：

```env
TASK_CORE_WORKERS=4
TASK_MAX_WORKERS=8
TASK_QUEUE_CAPACITY=100
```

当前 Python 的 `ThreadPoolExecutor` 只有 `max_workers` 参数，没有 Java 线程池中的 `corePoolSize` 概念，因此实际使用的是：

```python
max_workers=settings.task_max_workers
queue_capacity=settings.task_queue_capacity
```

`TASK_CORE_WORKERS` 目前只是保留的配置字段，暂未参与线程池创建，可以在后续确定线程池策略后删除或重新使用。

## 3. 有界线程池的实现

线程池定义在 `zq_rag_app/core/executor.py`：

```python
index_executor = BoundedThreadPoolExecutor(
    max_workers=settings.task_max_workers,
    queue_capacity=settings.task_queue_capacity,
    thread_name_prefix="index-",
)
```

线程池最多允许以下数量的任务同时存在：

```text
正在执行的任务：8 个
等待执行的任务：100 个
总任务容量：108 个
```

这里使用 `BoundedSemaphore` 限制任务总数：

```python
self._slots = BoundedSemaphore(
    max_workers + queue_capacity
)
```

提交任务前会尝试获取一个位置：

```python
if not self._slots.acquire(blocking=False):
    raise RuntimeError("索引任务队列已满")
```

使用 `blocking=False` 的效果是：队列满时立即返回错误，不会让当前接口一直等待。

## 4. 业务层提交任务

业务层通过 `zq_rag_app/services/document_service.py` 中的函数提交任务：

```python
from zq_rag_app.services.document_service import submit_document_index

future = submit_document_index(
    build_document_index,
    document_id,
)
```

`build_document_index` 是要在线程池中执行的函数，`document_id` 是传给它的参数。

内部调用关系如下：

```python
def submit_document_index(index_callable, *args, **kwargs):
    return submit_index_task(index_callable, *args, **kwargs)
```

然后进入线程池：

```python
def submit_index_task(fn, *args, **kwargs):
    return index_executor.submit(fn, *args, **kwargs)
```

因此，真正的索引函数不会在提交任务的请求线程中执行，而是在线程池的 `index-*` 工作线程中执行。

## 5. FastAPI 中的典型调用方式

未来实现文档上传接口时，可以按照以下方式调用：

```python
@router.post("/documents")
async def upload_document(file: UploadFile):
    # 1. 保存原文件到 MinIO
    # 2. 创建文档记录，状态为 PENDING
    document_id = create_document_record(file.filename)

    # 3. 提交索引任务，不等待任务完成
    submit_document_index(
        build_document_index,
        document_id,
    )

    # 4. 立即返回
    return {
        "document_id": document_id,
        "status": "PENDING",
    }
```

索引函数内部负责真正的业务处理：

```python
def build_document_index(document_id: int):
    try:
        # 1. 从 MinIO 读取原文件
        # 2. 解析 PDF、DOCX 或 Markdown
        # 3. 切分文本
        # 4. 调用 Embedding 模型
        # 5. 写入 PostgreSQL / pgvector
        # 6. 更新任务状态为 SUCCESS
        pass
    except Exception:
        # 更新任务状态为 FAILED
        logger.exception("文档索引失败: document_id=%s", document_id)
        raise
```

推荐的状态流转为：

```text
PENDING
   |
   v
PROCESSING
   | \
   |  \
   v   v
SUCCESS  FAILED
```

## 6. `Future` 的作用

`submit_document_index()` 返回一个 `Future`，它代表后台任务：

```python
future = submit_document_index(
    build_document_index,
    document_id,
)
```

可以查询任务状态：

```python
future.done()       # 是否执行完成
future.running()    # 是否正在执行
future.cancelled()  # 是否已取消
```

也可以获取任务结果：

```python
result = future.result(timeout=60)
```

但是在 FastAPI 上传接口中，一般不建议立即调用 `future.result()`，因为这会重新等待索引完成，失去异步提交的意义。

更推荐让索引函数自己更新数据库中的任务状态，前端通过查询接口查看进度。

## 7. 队列满时的处理

当 8 个工作线程和 100 个等待位置全部占用时：

```python
submit_document_index(...)
```

会抛出：

```python
RuntimeError("索引任务队列已满")
```

上传接口可以捕获这个异常并返回合适的 HTTP 状态：

```python
try:
    submit_document_index(
        build_document_index,
        document_id,
    )
except RuntimeError as exc:
    raise HTTPException(
        status_code=503,
        detail=str(exc),
    ) from exc
```

这表示服务暂时无法接受更多索引任务，客户端可以稍后重试。

## 8. 任务完成后的资源释放

每个任务完成后，线程池都会释放一个队列位置：

```python
future.add_done_callback(
    lambda _: self._slots.release()
)
```

无论索引任务成功还是失败，都会释放位置，使后续任务能够继续提交。

## 9. 应用关闭流程

应用关闭时，`zq_rag_app/main.py` 会调用：

```python
await asyncio.to_thread(shutdown_index_executor)
```

最终执行：

```python
index_executor.shutdown(
    wait=True,
    cancel_futures=False,
)
```

含义是：

1. 不再接收新任务；
2. 等待正在执行和排队中的任务完成；
3. 释放线程池资源。

## 10. 用户上下文传递

当前用户上下文使用 `ContextVar`，定义在 `zq_rag_app/core/security.py`。

需要注意：普通 `ThreadPoolExecutor` 不会自动把当前异步请求的 `ContextVar` 传递到工作线程。因此不要直接依赖工作线程中的隐式上下文，建议显式传递用户信息：

```python
user_id = get_user_id()
department_id = get_department_id()
role = get_role()

submit_document_index(
    build_document_index,
    document_id,
    user_id,
    department_id,
    role,
)
```

对应的索引函数：

```python
def build_document_index(
    document_id: int,
    user_id: int | None,
    department_id: str | None,
    role: str | None,
):
    ...
```

这样可以避免任务延迟执行时丢失创建任务的用户权限信息。

## 11. 当前项目状态

当前项目已经完成：

- 有界索引线程池；
- 统一的 `submit_document_index()` 提交入口；
- 队列满保护；
- 任务完成后自动释放队列位置；
- 应用关闭时等待并释放线程池。

当前项目尚未完成：

- `zq_rag_app/api/document.py` 上传接口；
- `build_document_index()` 具体索引函数；
- 文档解析、切分、Embedding 和向量入库流程；
- 索引任务状态表及状态查询接口。

因此，目前线程池基础设施已经可以使用，但还需要将真实的文档索引业务函数接入 `submit_document_index()`。
