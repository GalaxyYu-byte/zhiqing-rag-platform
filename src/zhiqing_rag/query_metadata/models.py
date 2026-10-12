"""原始过滤条件与服务端解析后的白名单条件；不接收 SQL 或任意 JSON 路径。"""

import re
from datetime import date, datetime
from typing import Literal

from pydantic import Field, model_validator

from zhiqing_rag.query_analysis.models import ShortText, StrictModel

MetadataField = Literal[
    "department", "document_name", "document_code", "file_type", "business_status",
    "uploaded_at", "event_time",
]
FilterOperator = Literal["eq", "neq", "contains", "gt", "gte", "lt", "lte", "range"]


class MetadataFilter(StrictModel):
    field: MetadataField
    operator: FilterOperator
    value: ShortText


class ResolvedFilter(StrictModel):
    field: Literal[
        "department_id", "document_name", "document_code", "file_type", "business_status",
        "uploaded_at", "event_date",
    ]
    operator: FilterOperator
    values: tuple[str, ...] = Field(min_length=1, max_length=10)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_department_ids(cls, value):
        # 旧公共响应和聊天私有来源快照保存过 JSON 整数；读取时立即转成无损字符串。
        if isinstance(value, dict) and value.get("field") == "department_id":
            values = value.get("values")
            if isinstance(values, (tuple, list)):
                normalized = [str(item) if type(item) is int else item for item in values]
                return {**value, "values": tuple(normalized)}
        return value

    @model_validator(mode="after")
    def validate_typed_values(self):
        if self.field == "department_id":
            valid = self.operator in ("eq", "neq") and len(self.values) == 1 and all(
                re.fullmatch(r"[1-9][0-9]{0,18}", value) is not None
                and int(value) <= 2**63 - 1 for value in self.values
            )
        elif self.field in ("document_name", "document_code"):
            valid = self.operator in ("eq", "neq", "contains") and len(self.values) == 1
        elif self.field in ("file_type", "business_status"):
            valid = self.operator in ("eq", "neq")
        else:
            valid = self.operator != "contains" and len(self.values) == (
                2 if self.operator == "range" else 1
            )
        if not valid:
            raise ValueError("过滤字段的操作符或值数量不匹配")
        if self.field != "department_id" and not all(
            isinstance(value, str) and 0 < len(value) <= 200 for value in self.values
        ):
            raise ValueError("过滤值必须是长度受限的字符串")
        if self.field in ("uploaded_at", "event_date"):
            try:
                parser = (
                    datetime.fromisoformat if self.field == "uploaded_at" else date.fromisoformat
                )
                parsed = tuple(parser(value) for value in self.values)
            except ValueError as error:
                raise ValueError("时间过滤必须使用 ISO 日期或时间戳") from error
            if self.field == "uploaded_at" and any(item.tzinfo is None for item in parsed):
                raise ValueError("上传时间必须带时区")
            if len(parsed) == 2 and parsed[0] >= parsed[1]:
                raise ValueError("时间范围的起点必须早于终点")
        if self.field == "business_status" and any(
            value not in {"UNKNOWN", "DRAFT", "EFFECTIVE", "ARCHIVED", "REPEALED"}
            for value in self.values
        ):
            raise ValueError("无效业务状态")
        return self


class MetadataFilterError(ValueError):
    """无法完整确定条件时停止查询，要求用户澄清。"""
