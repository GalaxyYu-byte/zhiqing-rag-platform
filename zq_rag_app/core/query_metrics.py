"""低基数在线问答指标，不记录问题、实体、用户或权限范围。"""

from prometheus_client import Counter, Histogram


ANALYZER_REQUESTS = Counter(
    "zq_query_analyzer_requests_total", "Analyzer 逻辑请求数（含重新分析）", ("status", "reason"),
)
ANALYZER_CALLS = Counter("zq_query_analyzer_calls_total", "Analyzer 实际模型调用尝试数")
ANALYZER_RETRIES = Counter(
    "zq_query_analyzer_retries_total", "Analyzer 实际重试数", ("reason",),
)
ANALYZER_DURATION = Histogram(
    "zq_query_analyzer_duration_seconds", "Analyzer 请求耗时（含重试和退避）",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 60, 120),
)
CHAT_HTTP_DURATION = Histogram(
    "zq_chat_http_duration_seconds", "Chat HTTP 端到端耗时（含权限、模型调用及落库）", ("outcome",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180),
)
