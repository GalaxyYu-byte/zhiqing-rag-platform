"""分析不可用时仅识别明显的上下文依赖，不把有历史等同于依赖历史。"""

import re
import unicodedata


_PRIOR_REFERENCE = re.compile(
    r"上文|上述|前述|同上|前者|后者|上一个|上一条|这两者|"
    r"(?:前面|刚才|之前)(?:提到|说到|说的|说过)|"
    r"\b(?:aforementioned|previous answer|previous question|mentioned above)\b",
    re.IGNORECASE,
)
_LEADING_PRONOUN = re.compile(
    r"^(?:(?:那么|请问|关于|对于|介绍一下|告诉我|再问一下|我想知道|那|请)\s*)*"
    r"(?:他|她|它)(?:们)?(?=$|[的还也又是否为在能会有叫从跟和为什么负责做去今明上最\d，,？?。.!！])"
)
_GENERIC_SUBJECT = re.compile(
    r"^(?:(?:那么|请问|关于|对于|那|请)\s*)*"
    r"(?:这个|那个|该|此|这份|那份)(?:项目|系统|合同|文档|文件|方案|制度|问题|功能|模型|方法|模块|版本|部门|负责人)"
    r"(?=$|[的是谁怎么如何为什么是否能还也去今明上最近\d，,？?。.!！])"
)
_FOLLOW_UP = re.compile(
    r"^(?:继续(?:说|讲)?|接着(?:说|讲)?|展开说说|再详细(?:一点|点|介绍)|详细(?:一点|点)|"
    r"再解释一下|还有呢|然后呢|那(?:么)?呢|那(?:么)?怎么办|"
    r"(?:那|那么)?(?:去年|今年|明年|上个月|上周|原因|流程|费用|金额|时间|负责人|进度|其他的|其他)呢)"
    r"[？?。.!！]*$"
)
_ENGLISH_REFERENCE = re.compile(
    r"^(?:(?i:please|and|also|then|what about|tell me about)\s+)*"
    r"(?:[Hh]e|[Hh]is|[Ss]he|[Hh]er|[Tt]hey|[Tt]heir|[Tt]hem|[Ii]ts|[Ii]t)\b|"
    r"\b(?:it|them)\s*[?.!]*$",
)


def requires_context_on_failure(query: str) -> bool:
    """有限文本规则，不补全对象、不读取旧主题，也不依赖故障模型的标记。

    只检查当前输入；词内的“其他”“吉他”和当前问题已经引入主体后的代词
    不按句首指代处理。这是降级保护，不替代正常 Analyzer 的语义识别。
    """
    text = unicodedata.normalize("NFKC", query).strip()
    compact = re.sub(r"\s+", "", text)
    return bool(
        _PRIOR_REFERENCE.search(text) or _LEADING_PRONOUN.search(compact)
        or _GENERIC_SUBJECT.search(compact) or _FOLLOW_UP.fullmatch(compact)
        or _ENGLISH_REFERENCE.search(text)
    )
