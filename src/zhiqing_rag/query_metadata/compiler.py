"""固定列与固定操作符编译；用户值永远作为参数绑定。"""

import re

from .models import ResolvedFilter


def compile_filters(
    filters: tuple[ResolvedFilter, ...], *, document_alias="doc", revision_alias="r",
) -> tuple[str, tuple]:
    for alias in (document_alias, revision_alias):
        if not isinstance(alias, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", alias) is None:
            raise ValueError("无效过滤 SQL 别名")
    columns = {
        "department_id": f"{document_alias}.department_id",
        "document_name": f"{revision_alias}.original_filename",
        "document_code": f"{document_alias}.document_code",
        "file_type": f"{revision_alias}.mime_type",
        "business_status": f"{document_alias}.business_status",
        "uploaded_at": f"{revision_alias}.created_at",
        "event_date": f"{document_alias}.event_date",
    }
    operators = {"eq": "=", "neq": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
    predicates, parameters = [], []
    for item in filters:
        # 即使内部调用者使用 model_construct，也不能绕过字段和类型边界。
        item = ResolvedFilter.model_validate(item.model_dump())
        column = columns[item.field]
        cast = "::timestamptz" if item.field == "uploaded_at" else (
            "::date" if item.field == "event_date" else ""
        )
        if item.operator == "range":
            predicates.append(f"({column} >= %s{cast} AND {column} < %s{cast})")
            parameters.extend(item.values)
        elif item.operator == "contains":
            predicates.append(f"strpos(lower({column}), lower(%s)) > 0")
            parameters.extend(item.values)
        elif len(item.values) > 1:
            statement = f"{column} = ANY(%s::text[])"
            predicates.append(f"NOT ({statement})" if item.operator == "neq" else statement)
            parameters.append(list(item.values))
        else:
            predicates.append(f"{column} {operators[item.operator]} %s{cast}")
            parameters.extend(
                int(value) if item.field == "department_id" else value for value in item.values
            )
    return " AND ".join(f"({predicate})" for predicate in predicates) or "TRUE", tuple(parameters)
