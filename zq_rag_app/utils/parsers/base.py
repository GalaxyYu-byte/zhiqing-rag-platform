"""文档解析策略的公共接口和返回模型。

每一种文件格式都是一个独立的解析策略。业务代码只依赖
``DocumentParser`` 抽象类，不需要知道 PDF、Word 或 Excel 的具体实现细节。
这样新增格式时只需增加一个策略类并注册到工厂，不需要修改索引业务流程。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


class UnsupportedDocumentTypeError(ValueError):
    """请求解析的文件类型没有对应策略时抛出的异常。"""


class DocumentParseError(RuntimeError):
    """文件格式受支持，但文件内容无法被正常解析时抛出的异常。"""



@dataclass(slots=True)
class ParsedBlock:
    """解析器输出的统一文本块。

    解析阶段只负责保留文档自身的自然边界，例如 PDF 页、Word 章节和
    Excel Sheet。它不会在这里按照 RAG 的 chunk size 继续切分；真正的文本
    Chunk 应由后续 ChunkService 处理。

    Attributes:
        content: 当前自然块的纯文本内容。
        page_num: 原始 PDF 页码，从 1 开始；其他格式通常为空。
        section_title: Word/Markdown 章节名或 Excel Sheet 名。
        metadata: 不适合放在固定字段中的格式特有信息，例如文件名、编码和
            Sheet 序号。后续入库时可以按需写入 Chunk 元数据。
        heading_level: 当前章节标题级别；解析器无法确定时为空。
        heading_path: 从最高级标题到当前章节的完整路径。
    """

    content: str
    page_num: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    heading_level: int | None = None
    heading_path: list[str] = field(default_factory=list)



class DocumentParser(ABC):
    """所有文档解析策略必须实现的抽象接口。"""
    # 子类声明自己支持的扩展名。扩展名统一使用小写并包含前导点。
    
    supported_extensions: ClassVar[frozenset[str]] = frozenset()

    def supports(self, file_path: str | Path) -> bool:
        """判断当前策略是否支持给定文件。

        这里只根据扩展名选择策略；如果需要更严格的安全校验，可在上传层再
        检查 MIME type 和文件魔数，避免仅修改扩展名伪装文件格式。
        """

        extension = Path(file_path).suffix.lower()
        # 接口约定扩展名使用小写；这里仍做一次归一化，让第三方自定义策略即使
        # 误写成大写扩展名也能被正确判断，注册工厂会负责最终的统一存储。
        normalized_supported = {
            supported.lower() for supported in self.supported_extensions
        }
        return extension in normalized_supported

    def validate_file(self, file_path: str | Path) -> Path:
        """执行所有解析器共用的文件检查并返回 ``Path``。

        将校验集中到基类，可以确保所有实现对“不存在、不是普通文件、格式
        不匹配”等情况给出一致且清晰的错误信息。
        """

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"待解析文件不存在: {path}")
        if not path.is_file():
            raise ValueError(f"待解析路径不是文件: {path}")
        if not self.supports(path):
            supported = ", ".join(sorted(self.supported_extensions))
            raise UnsupportedDocumentTypeError(
                f"{type(self).__name__} 不支持 {path.suffix or '无扩展名'} 文件，"
                f"支持的格式为: {supported}"
            )
        return path


    @abstractmethod
    def parse(self, file_path: str | Path) -> list[ParsedBlock]:
        """解析文件并返回按文档自然结构组织的文本块。"""
