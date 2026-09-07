"""文档清洗阶段的异常类型。"""


class DocumentCleaningError(RuntimeError):
    """所有文档清洗异常的基类。"""


class CleaningRuleExecutionError(DocumentCleaningError):
    """包装具体清洗规则的未预期异常。"""

    def __init__(self, rule_name: str, message: str) -> None:
        self.rule_name = rule_name
        super().__init__(f"清洗规则执行失败 [{rule_name}]: {message}")


class CleaningValidationError(DocumentCleaningError):
    """清洗结果未通过安全校验。"""

    def __init__(self, error_count: int) -> None:
        self.error_count = error_count
        super().__init__(f"清洗结果安全校验失败，共发现 {error_count} 个错误")
