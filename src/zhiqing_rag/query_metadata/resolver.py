"""将有原文依据或显式选择的条件转成确定类型；日期解析不依赖语言模型。"""

import re
from datetime import UTC, date, datetime, timedelta, timezone

import psycopg
from psycopg import sql
from starlette.concurrency import run_in_threadpool

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.infrastructure.database.connection import database_connection
from zhiqing_rag.modules.identity.auth_schemas import AuthError

from .models import MetadataFilter, MetadataFilterError, ResolvedFilter

SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")
_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_FILE_TYPES = {
    "pdf": ("application/pdf",), "docx": (_DOCX,), "word": (_DOCX,),
    "xlsx": (_XLSX,), "xls": ("application/vnd.ms-excel",),
    "excel": (_XLSX, "application/vnd.ms-excel"), "txt": ("text/plain",),
    "text": ("text/plain",), "文本": ("text/plain",),
    "md": ("text/markdown",), "markdown": ("text/markdown",),
}
_STATUS = {
    "未知": "UNKNOWN", "草稿": "DRAFT", "有效": "EFFECTIVE", "生效": "EFFECTIVE",
    "已生效": "EFFECTIVE", "归档": "ARCHIVED", "已归档": "ARCHIVED",
    "废止": "REPEALED", "已废止": "REPEALED",
}


def _next_month(value):
    return value.replace(year=value.year + 1, month=1) if value.month == 12 else (
        value.replace(month=value.month + 1)
    )


def _period(text: str, now: datetime) -> tuple[datetime, datetime]:
    text = text.strip()
    day = now.astimezone(SHANGHAI).replace(hour=0, minute=0, second=0, microsecond=0)
    month = day.replace(day=1)
    year = month.replace(month=1)
    week = day - timedelta(days=day.weekday())
    relatives = {
        "今天": (day, day + timedelta(days=1)), "昨天": (day - timedelta(days=1), day),
        "今年": (year, year.replace(year=year.year + 1)),
        "去年": (year.replace(year=year.year - 1), year),
        "本月": (month, _next_month(month)), "这个月": (month, _next_month(month)),
        "上月": ((month - timedelta(days=1)).replace(day=1), month),
        "上个月": ((month - timedelta(days=1)).replace(day=1), month),
        "本周": (week, week + timedelta(days=7)),
        "上周": (week - timedelta(days=7), week),
    }
    if text in relatives:
        return relatives[text]
    match = re.fullmatch(r"(\d{4})(?:年|[-/])?(?:(\d{1,2})(?:月|[-/])?)?(?:(\d{1,2})日?)?", text)
    if not match:
        raise MetadataFilterError("时间条件不明确，请使用具体年份、年月、日期或带时区的 ISO 时间。")
    year_number, month_number, day_number = match.groups()
    try:
        start = datetime(
            int(year_number), int(month_number or 1), int(day_number or 1), tzinfo=SHANGHAI,
        )
        end = start + timedelta(days=1) if day_number else (
            _next_month(start) if month_number else start.replace(year=start.year + 1)
        )
    except ValueError as error:
        raise MetadataFilterError("时间条件包含无效日期，请确认日期。") from error
    return start, end


def resolve_date(item: MetadataFilter, now: datetime) -> ResolvedFilter:
    field = "event_date" if item.field == "event_time" else "uploaded_at"
    if item.operator == "contains":
        raise MetadataFilterError("日期不支持包含匹配，请使用时间范围或比较条件。")
    if item.operator == "range":
        parts = re.split(r"\s*(?:至|到|~|～|\s+-\s+)\s*", item.value)
        if len(parts) == 1:
            start, end = _period(item.value, now)
        elif len(parts) == 2:
            start, _ = _period(parts[0], now)
            _, end = _period(parts[1], now)
        else:
            raise MetadataFilterError("时间范围需明确起止日期，例如 2025-01-01 至 2025-12-31。")
        if start >= end:
            raise MetadataFilterError("时间范围的起点必须早于终点。")
        operator, values = "range", (start, end)
    else:
        # 精确 ISO 时间戳保留瞬时比较；日期和月份统一为上海时区日历范围。
        instant = None
        if field == "uploaded_at" and "T" in item.value:
            try:
                instant = datetime.fromisoformat(item.value)
            except ValueError as error:
                raise MetadataFilterError("上传时间不是有效 ISO 时间戳。") from error
            if instant.tzinfo is None:
                raise MetadataFilterError("精确上传时间需要时区，例如 2025-01-01T00:00:00+08:00。")
        if instant is not None:
            operator, values = item.operator, (instant,)
        else:
            start, end = _period(item.value, now)
            if item.operator == "neq":
                raise MetadataFilterError("日历时间的不等于条件暂不支持，请改用明确起止范围。")
            operator, values = {
                "eq": ("range", (start, end)), "gte": ("gte", (start,)),
                "gt": ("gte", (end,)), "lt": ("lt", (start,)), "lte": ("lt", (end,)),
            }[item.operator]
    serialized = tuple(
        value.date().isoformat() if field == "event_date" else value.astimezone(UTC).isoformat()
        for value in values
    )
    return ResolvedFilter(field=field, operator=operator, values=serialized)


def _validate_conjunction(filters: tuple[ResolvedFilter, ...]) -> None:
    """确认 AND 条件仍有可取的值；文件类型别名按 MIME 集合求交。"""
    grouped = {}
    for item in filters:
        grouped.setdefault(item.field, []).append(item)
    for field, conditions in grouped.items():
        if field not in ("uploaded_at", "event_date"):
            allowed, excluded = None, set()
            for item in conditions:
                if item.operator == "eq":
                    values = set(item.values)
                    allowed = values if allowed is None else allowed & values
                elif item.operator == "neq":
                    excluded.update(item.values)
            if allowed is not None and not (allowed - excluded):
                raise MetadataFilterError("筛选条件相互冲突，请保留有交集的条件。")
            continue

        parser = datetime.fromisoformat if field == "uploaded_at" else date.fromisoformat
        lower, upper, excluded = None, None, set()
        for item in conditions:
            values = tuple(parser(value) for value in item.values)
            candidate_lower, candidate_upper = None, None
            if item.operator == "range":
                candidate_lower, candidate_upper = (values[0], True), (values[1], False)
            elif item.operator == "eq":
                candidate_lower = candidate_upper = (values[0], True)
            elif item.operator == "neq":
                excluded.add(values[0])
            elif item.operator in ("gt", "gte"):
                candidate_lower = (values[0], item.operator == "gte")
            else:
                candidate_upper = (values[0], item.operator == "lte")
            if candidate_lower is not None:
                if lower is None or candidate_lower[0] > lower[0]:
                    lower = candidate_lower
                elif candidate_lower[0] == lower[0]:
                    lower = (lower[0], lower[1] and candidate_lower[1])
            if candidate_upper is not None:
                if upper is None or candidate_upper[0] < upper[0]:
                    upper = candidate_upper
                elif candidate_upper[0] == upper[0]:
                    upper = (upper[0], upper[1] and candidate_upper[1])
        if lower is not None and upper is not None and (
            lower[0] > upper[0] or (
                lower[0] == upper[0] and (not (lower[1] and upper[1]) or lower[0] in excluded)
            )
        ):
            raise MetadataFilterError("时间筛选条件没有交集，请调整起止日期或比较条件。")


class MetadataFilterResolver:
    def __init__(self, settings: Settings | None = None, *, clock=None, department_lookup=None):
        self.settings = settings
        self.clock = clock or (lambda: datetime.now(UTC))
        self.department_lookup = department_lookup

    def _departments(self, tenant_id, value):
        settings = self.settings or get_settings()
        try:
            with database_connection(settings) as connection:
                return connection.execute(
                    sql.SQL("SELECT id FROM {schema}.department WHERE tenant_id=%s "
                            "AND is_active AND (name=%s OR code=%s) ORDER BY id LIMIT 2").format(
                        schema=sql.Identifier(settings.db_schema),
                    ),
                    (tenant_id, value, value),
                ).fetchall()
        except psycopg.Error as error:
            raise AuthError(
                503, "DATABASE_UNAVAILABLE", "暂时无法确认部门筛选，请稍后重试",
            ) from error

    async def resolve(self, identity, filters) -> tuple[ResolvedFilter, ...]:
        resolved = []
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("过滤解析时钟必须带时区")
        for item in filters:
            item = MetadataFilter.model_validate(item.model_dump())
            if item.field in ("uploaded_at", "event_time"):
                value = resolve_date(item, now)
            elif item.field == "department":
                if item.operator not in ("eq", "neq"):
                    raise MetadataFilterError("部门暂仅支持等于或不等于条件。")
                if self.department_lookup is None:
                    rows = await run_in_threadpool(
                        self._departments, identity.tenant_id, item.value,
                    )
                else:
                    rows = await self.department_lookup(identity.tenant_id, item.value)
                if len(rows) != 1:
                    raise MetadataFilterError(
                        "部门名称无法唯一确定，请使用当前租户的准确部门名称或代码。",
                    )
                value = ResolvedFilter(
                    field="department_id", operator=item.operator, values=(str(rows[0]["id"]),),
                )
            elif item.field in ("business_status", "file_type"):
                if item.operator not in ("eq", "neq"):
                    raise MetadataFilterError("文件类型和业务状态暂仅支持等于或不等于条件。")
                if item.field == "business_status":
                    name = _STATUS.get(item.value, item.value.upper())
                    if name not in {"UNKNOWN", "DRAFT", "EFFECTIVE", "ARCHIVED", "REPEALED"}:
                        raise MetadataFilterError(
                            "业务状态无法确定，请使用未知、草稿、生效、归档或废止。",
                        )
                    values = (name,)
                else:
                    name = item.value.lower().lstrip(".")
                    values = _FILE_TYPES.get(name)
                    known_mimes = {mime for group in _FILE_TYPES.values() for mime in group}
                    if values is None and name in known_mimes:
                        values = (name,)
                    if values is None:
                        raise MetadataFilterError(
                            "文件类型暂支持 PDF、DOCX、XLSX、XLS、TXT 和 Markdown。",
                        )
                value = ResolvedFilter(field=item.field, operator=item.operator, values=values)
            else:
                if item.operator not in ("eq", "neq", "contains"):
                    raise MetadataFilterError("文档名称和文号暂支持等于、不等于或包含条件。")
                value = ResolvedFilter(
                    field=item.field, operator=item.operator, values=(item.value,),
                )
            if value not in resolved:
                resolved.append(value)
        _validate_conjunction(tuple(resolved))
        return tuple(resolved)
