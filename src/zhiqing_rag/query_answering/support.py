"""对容易被改写成另一流程状态的陈述执行原文保护检查。

这不是中文蕴含分类器。状态认定、不能推出结论及条件性职责采用保守的
复制约束；普通事实仍由引用检查和模型复核审核。保护语句完全来自当前
授权原文，不从图三元组或模型的 supported 标记产生新的事实。
"""

import hashlib
import re
from dataclasses import dataclass

from zhiqing_rag.query_retrieval.models import Evidence

from .models import DraftAnswer

_JUDGEMENT = re.compile(r"视为|视作|当成|认定为|等同于|意味着|代表|标为|算作")
_NEGATIVE = re.compile(r"不能|不得|不应|不要|不可|并不|不代表|不意味着|未必|不$")
_CONDITION = re.compile(r"如果|除非|仅在|只有|若|(?:确认|批准|审核|接收|完成|确定).{0,16}[后前时]")
_RESPONSIBILITY = re.compile(r"负责(?!人)")
_COMPLETION = re.compile(r"已|未|没有|完成|解决|办结|结束|关闭|生效|开通|获批")
_BOUNDARY = re.compile(r"[。；！？;!?]|(?:\r?\n){2,}")


@dataclass(frozen=True)
class ProtectedClause:
    kind: str
    text: str
    start: int
    end: int
    subject: str = ""
    result: str = ""
    required_fragments: tuple[str, ...] = ()

    @property
    def answer_text(self) -> str:
        return "，".join(self.required_fragments) if self.required_fragments else self.text

    def payload(self) -> dict:
        return {
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "answer_text": self.answer_text,
        }


def _clauses(text: str):
    start = 0
    for boundary in (*_BOUNDARY.finditer(text), None):
        end = boundary.start() if boundary is not None else len(text)
        raw = text[start:end]
        left = len(raw) - len(raw.lstrip())
        value = raw.strip()
        heading = re.match(r"[^\n]{1,80}\r?\n[-=]{3,}\r?\n", value)
        if heading:
            left += heading.end()
            value = value[heading.end():]
            left += len(value) - len(value.lstrip())
            value = value.strip()
        if value:
            yield value, start + left, start + left + len(value)
        if boundary is not None:
            start = boundary.end()


def _compact(text: str) -> str:
    # 排版换行不改变被保护的主体、条件、否定和状态词；不改标点或词语。
    return re.sub(r"\s+", "", text).strip("\"'“”‘’。；！？;!?")


def _protected_text(text: str) -> str:
    # 同一禁止把某对象认作另一状态的句式；不折叠不应、可能、未必等模态。
    return re.sub(r"(?:不能|不得|不要|不可)把", "不要把", _compact(text))


def _source_units(text: str):
    for value, start, end in _clauses(text):
        if not any(line.strip().startswith("|") for line in value.splitlines()):
            yield value, start, end
            continue
        # 表格的一行或一列不能借另一行/列的条件形成风险语句。
        # 跨列含义由完整列头及模型审核检查，不在这里猜测业务关系。
        offset = start
        prose, prose_start = [], offset
        for line in value.splitlines(keepends=True):
            if line.strip().startswith("|"):
                for clause, left, right in _clauses("".join(prose)):
                    yield clause, prose_start + left, prose_start + right
                prose = []
                boundaries = [match.start() for match in re.finditer(r"(?<!\\)\|", line)]
                for left, right in zip(boundaries, boundaries[1:]):
                    raw = line[left + 1:right]
                    cell = raw.strip()
                    cell_start = offset + left + 1 + len(raw) - len(raw.lstrip())
                    if cell:
                        yield cell, cell_start, cell_start + len(cell)
            else:
                if not prose:
                    prose_start = offset
                prose.append(line)
            offset += len(line)
        for clause, left, right in _clauses("".join(prose)):
            yield clause, prose_start + left, prose_start + right


def protected_clauses(item: Evidence) -> tuple[ProtectedClause, ...]:
    clauses = []
    for text, start, end in _source_units(item.content):
        judgement = _JUDGEMENT.search(text)
        if judgement is not None:
            subject = ""
            before = text[:judgement.start()]
            object_match = re.search(r"(?:把|将)([^，。；]{1,40}?)(?:直接)?$", before)
            if object_match:
                subject = object_match[1]
            result = text[judgement.end():].strip(" ，、：:‘’“”\"'")
            kind = "non_implication" if _NEGATIVE.search(before) else "state_judgement"
            parts = text.split("，")
            fragments = ()
            if (
                len(parts) >= 3 and parts[0].endswith(("时", "后", "前"))
                and _JUDGEMENT.search(parts[-1])
                and all(re.match(r"^(?:记录|登记|说明|列出|补充)", part)
                        for part in parts[1:-1])
            ):
                # 同句中独立的登记动作不是状态认定的前提，不强迫答案追加它。
                # 条件与判断各自仍是可定位的连续原文，不能删掉条件或换结论。
                fragments = (parts[0], parts[-1])
            clauses.append(ProtectedClause(kind, text, start, end, subject, result, fragments))
        elif (responsibility := _RESPONSIBILITY.search(text)) and _CONDITION.search(text):
            prefix = text[:responsibility.start()].rsplit("，", 1)[-1]
            before = re.split(r"[后前时]", prefix)[-1]
            if not before and "在" in prefix:
                before = prefix.split("在", 1)[0]
            subject = re.sub(r"(?:仍|继续|还|将|应当|应)$", "", before).strip()
            if not subject or re.search(r"谁|何人|哪方|哪个", subject):
                # “先确定谁负责”没有给出职责主体，不等于带条件的具体职责。
                continue
            clauses.append(
                ProtectedClause("conditional_responsibility", text, start, end, subject)
            )
    return tuple(clauses)


@dataclass(frozen=True)
class ConfirmationClause:
    stage: str
    text: str
    start: int
    end: int
    topic: str = ""

    def payload(self):
        return {"stage": self.stage, "answer_text": self.text,
                "start": self.start, "end": self.end, "source_topic": self.topic}


def general_confirmation_question(question: str) -> bool:
    """只保护一般确认要求；特定阶段/确认类型和字段问题不要求全流程。"""
    return bool(re.search(r"(?:什么|哪些)(?:确认|核对)(?:条件|要求)", question)) and not bool(
        re.search(r"启动|开始|进入下一步|上一环节|结束前|最后一道|第[一二三四五六七八九十\d]+步"
                  r"|接收[前后]|交接[前后]|状态|表格|信息项|字段", question)
    )


def source_document_key(item: Evidence):
    return (item.knowledge_base_id, item.document_id, item.revision_id, item.generation_id)


def _confirmation_topic(item, offset):
    section = (item.section_title or "").strip()
    generic = r"(?:接收前应看|交接动作|接收确认|办理次序|办理步骤|启动确认|结束确认|最后核对)"
    headings = re.finditer(r"(?:^|\n)(?:#{1,6}\s+([^\n]+)|([^\n]{1,80})\r?\n[-=]{3,})",
                           item.content[:offset])
    topics = [match.group(1) or match.group(2) for match in headings]
    explicit = next((topic.strip() for topic in reversed(topics)
                     if re.search(r"流程|手续|事项|办理", topic)), "")
    if explicit:
        return explicit
    if section and not re.fullmatch(generic, section):
        return re.split(r"\s*[/＞>:：]\s*", section, maxsplit=1)[0]
    return ""


def confirmation_clauses(item: Evidence) -> tuple[ConfirmationClause, ...]:
    """只标原文明示的正向阶段句；偏移仍对应当前同一 chunk 的连续原文。"""
    result, previous_end = [], False
    for text, start, end in _source_units(item.content):
        prefix = re.match(r"\d+[)）.、]\s*", text)
        if prefix:
            start += prefix.end()
            text = text[prefix.end():]
        stage = None
        if not _NEGATIVE.search(text) and not re.match(r"如果|若|除非", text):
            if re.match(r"(?:启动|开始)[^。；]{0,24}时[，,]", text) and "先确认" in text:
                stage = "start"
            elif re.match(r"进入下一步前[，,].*先确认", text):
                stage = "transition"
            elif re.match(r"第[一二三四五六七八九十\d]+步[，,]", text) and "确认" in text:
                stage = "step"
            elif re.search(r"最后一道(?:核对|确认|检查)", text):
                stage = "finish"
            elif previous_end and re.match(r"[^，。；]{1,15}应(?:查看|检查|核对)", text):
                stage = "finish"
        previous_end = stage == "finish"
        if stage and len(text) <= 1200:
            assert item.content[start:end] == text
            result.append(ConfirmationClause(stage, text, start, end,
                                             _confirmation_topic(item, start)))
    return tuple(result)


def _state_table_cells(item):
    rows, offset = [], 0
    for line in item.content.splitlines(keepends=True):
        if line.strip().startswith("|"):
            cells, cursor = [], line.find("|") + 1
            for cell in line.strip().strip("|").split("|"):
                cell = cell.strip()
                start = line.find(cell, cursor) if cell else cursor
                cells.append((cell, offset + start, offset + start + len(cell)))
                cursor = start + len(cell) + 1
            rows.append(cells)
        offset += len(line)
    if not rows or not any("状态" in cell[0] for cell in rows[0]) or not any(
        "离开" in cell[0] for cell in rows[0]
    ):
        return ()
    return tuple({"text": cell, "start": start, "end": end,
                  "state": row[0][0],
                  "column": rows[0][index][0] if index < len(rows[0]) else ""}
                 for row in rows[2:] for index, (cell, start, end) in enumerate(row)
                 if len(cell) >= 4)


def _topic_matches(topic, text):
    subject = re.sub(r"(?:流程|手续|事项|办理|条件|要求|确认)$", "", topic)
    return len(subject) >= 2 and subject in text


def _confirmation_frame(text):
    # 仅归一局部泛指框架；先/再、序号、实质条件/角色和否定均不改。
    text = _compact(text).replace("启动这项工作时，", "启动时")
    text = re.sub(r"(第[一二三四五六七八九十\d]+步)[，,为]", r"\1", text)
    return text.replace("本事项以", "以")


def confirmation_diagnostics(question, draft, evidence) -> tuple[dict, ...]:
    """收集同一保护扫描的具体失败；来源只是当前已授权原句和匹配的表格单元格。"""
    if not general_confirmation_question(question):
        return ()
    by_id = {item.citation_id: item for item in evidence}
    phases = {item.citation_id: confirmation_clauses(item) for item in evidence}
    diagnostics = []
    for statement_index, statement in enumerate(draft.statements):
        selected = {source_document_key(by_id[c.citation_id]) for c in statement.citations}
        selected_ids = {c.citation_id for c in statement.citations}
        required = []
        for document in selected:
            same_document = [(item, clause) for item in evidence
                             if source_document_key(item) == document
                             for clause in phases[item.citation_id]]
            topics = {clause.topic for _, clause in same_document if clause.topic}
            requested = {topic for topic in topics if _topic_matches(topic, question)}
            statement_topics = {topic for topic in requested
                                if _topic_matches(topic, statement.text)}
            cited_topics = {clause.topic for item, clause in same_document
                            if item.citation_id in selected_ids and clause.topic in requested}
            chosen = statement_topics or (cited_topics if len(cited_topics) == 1 else set())
            required.extend((item, clause) for item, clause in same_document
                            if (not clause.topic and not topics) or clause.topic in chosen)
        for item, clause in required:
            # 同文档的另一个 chunk 仍须由模型显式选 CID，不能服务器默添新来源。
            quoted = any(source_document_key(by_id[c.citation_id]) == source_document_key(item)
                         and _compact(clause.text) in _compact(c.quote)
                         for c in statement.citations)
            in_text = _confirmation_frame(clause.text) in _confirmation_frame(statement.text)
            if not quoted or not in_text:
                diagnostics.append({"kind": "missing_literal_phase",
                                    "statement_index": statement_index,
                                    "citation_id": item.citation_id,
                                    "required_clause": clause.payload(),
                                    "quote_present": quoted, "text_present": in_text})
        if required:
            direct = [_compact(clause.text).replace("是否", "") for _, clause in required]
            claim = _compact(statement.text).replace("是否", "")
            for citation in statement.citations:
                for cell in _state_table_cells(by_id[citation.citation_id]):
                    fact = _compact(cell["text"]).replace("是否", "")
                    if fact in claim and not any(fact in source for source in direct):
                        diagnostics.append({"kind": "non_phase_state_cell",
                                            "statement_index": statement_index,
                                            "citation_id": citation.citation_id,
                                            "matched_cell": cell})
        for clause, left, _ in _clauses(statement.text):
            # 否定状态和失败处置可单独说明限制，不能挂在正向条件冒号的分号清单里。
            if (left and statement.text[left - 1] in "；;"
                and re.search(r"确认(?:条件|要求)", statement.text[:left])
                and not clause.startswith(("限制", "例外", "状态约束"))
                and ((_NEGATIVE.search(clause) and _JUDGEMENT.search(clause))
                     or clause.startswith(("如果", "若")))):
                diagnostics.append({"kind": "negative_state_as_positive",
                                    "statement_index": statement_index, "claim": clause})
    return tuple(diagnostics)


def confirmation_guard_failure(question, draft, evidence) -> bool:
    return bool(confirmation_diagnostics(question, draft, evidence))


def semantic_guard_failure(draft: DraftAnswer, evidence: tuple[Evidence, ...]) -> bool:
    """风险陈述须逐字保留完整来源语句，模型复核不能覆盖此检查。

仅使用陈述自己的 citations。引用后半句不能省掉同句前提；同一原文内
相邻的完成状态不能成为另一状态的依据。另一份直接来源若确有该完整
事实，可照原文陈述，不能凭一个 NON_IMPLICATION 规则推定它已经发生。
"""
    by_id = {item.citation_id: item for item in evidence}
    for statement in draft.statements:
        references = [(by_id[c.citation_id], c.quote) for c in statement.citations]
        protected = [rule for item, _ in references for rule in protected_clauses(item)]
        for claim, _, _ in _clauses(statement.text):
            normalized = _compact(claim)
            risky = _JUDGEMENT.search(claim) is not None
            for rule in protected:
                if rule.kind == "conditional_responsibility":
                    risky |= _RESPONSIBILITY.search(claim) is not None or bool(
                        rule.subject and rule.subject in claim
                    )
                elif rule.kind == "non_implication" and _COMPLETION.search(claim):
                    result = re.sub(r"已经|已", "", _compact(rule.result))
                    current = re.sub(r"已经|已|尚未|并未|仍未|没有|未", "", normalized)
                    core = re.sub(r"完成|办结|结束|关闭|生效|开通|获批", "", result)
                    core_status = re.search(
                        rf"(?:已经|已|尚未|并未|仍未|没有|未)(?:完成)?{re.escape(core)}"
                        rf"|{re.escape(core)}(?:已经|已|尚未|未|没有)?"
                        r"(?:完成|办结|结束|关闭|生效|开通|获批)", claim
                    ) if len(core) >= 2 else None
                    risky |= (
                        len(result) >= 2 and result in current
                        and (len(result) >= 4 or bool(rule.subject and rule.subject in claim))
                    ) or core_status is not None
            if not risky:
                continue
            supported = False
            for item, quote in references:
                for rule in protected_clauses(item):
                    if _protected_text(claim) == _protected_text(rule.answer_text) and all(
                        any(_compact(fragment) in _compact(c.quote)
                            for c in statement.citations if c.citation_id == item.citation_id)
                        for fragment in (rule.required_fragments or (rule.text,))
                    ):
                        supported = True
                        break
                if supported:
                    break
                for source, _, _ in _source_units(item.content):
                    if normalized == _compact(source) and _compact(source) in _compact(quote):
                        supported = True
                        break
                if supported:
                    break
            if not supported:
                return True
    return False


def quote_candidates(item: Evidence) -> list[dict]:
    """给模型可直接复制的连续片段；表头与数据行各自保留，绝不跨行拼接。"""
    spans = []
    if any(line.strip().startswith("|") for line in item.content.splitlines()):
        offset = 0
        for line in item.content.splitlines(keepends=True):
            text = line.strip()
            if text.startswith("|") and len(text) >= 2:
                start = offset + len(line) - len(line.lstrip())
                spans.append((text, start, start + len(text)))
            offset += len(line)
    else:
        spans.extend(_clauses(item.content))
    selected, seen, total = [], set(), 0
    for text, start, end in spans:
        if (start, end) in seen or len(text) > 1200:
            continue
        if total + len(text) > 600:
            continue
        if len(selected) >= 10:
            break
        assert item.content[start:end] == text
        selected.append({"index": len(selected), "text": text, "start": start, "end": end})
        seen.add((start, end))
        total += len(text)
    return selected


def _path_sources_cited(citations, by_chunk, path):
    sources = [hop.source for hop in path.hops]
    sources.extend(source for bridge in path.bridges for source in bridge.sources)
    return all(
        any(citation.citation_id == by_chunk.get(source.chunk_id)
            and _compact(source.source_quote) in _compact(citation.quote)
            for citation in citations)
        for source in sources
    )


def path_citations_complete(draft, evidence, paths) -> bool:
    by_chunk = {item.chunk_id: item.citation_id for item in evidence}
    citations = [citation for statement in draft.statements for citation in statement.citations]
    return any(_path_sources_cited(citations, by_chunk, path) for path in paths)


def verified_path_conditions(evidence, path):
    """条件提示仅取当前原文中逐字存在的已验证 hop 条件，不提供图三元组事实。"""
    by_chunk = {item.chunk_id: item for item in evidence}
    sources = [hop.source for hop in path.hops]
    sources.extend(source for bridge in path.bridges for source in bridge.sources)
    if set(path.required_chunk_ids) != {source.chunk_id for source in sources}:
        return None
    for source in sources:
        item = by_chunk.get(source.chunk_id)
        if item is None or not (
            all(getattr(source, key) == getattr(item, key) for key in (
                "knowledge_base_id", "document_id", "revision_id", "generation_id",
            ))
            and hashlib.sha256(item.content.encode()).hexdigest() == source.content_sha256
            and 0 <= source.quote_start < source.quote_end <= len(item.content)
            and item.content[source.quote_start:source.quote_end] == source.source_quote
        ):
            return None
    conditions = []
    for hop_index, hop in enumerate(path.hops):
        relation, source = hop.relation, hop.source
        if (relation not in by_chunk[source.chunk_id].graph_relations
            or relation.source_quote != source.source_quote
            or (relation.quote_start, relation.quote_end)
            != (source.quote_start, source.quote_end)):
            return None
        condition = relation.condition_text
        if not condition:
            continue
        offset = source.source_quote.find(condition)
        if offset < 0:
            return None
        start = source.quote_start + offset
        conditions.append({"citation_id": by_chunk[source.chunk_id].citation_id,
                           "hop_index": hop_index,
                           "condition_text": condition, "start": start,
                           "end": start + len(condition)})
    return tuple(conditions)


def _condition_in_text(text, condition):
    # 原条件须保持逐字及原时序；提到“无需/并非该条件”不等于保留条件。
    negative = re.compile(r"不能|不得|不应|不要|不可|并非|不是|无需|无须|不必|不需"
                          r"|无论|不论|即使|即便|否定|尚未|并未|未(?:在|经|按)|没有"
                          r"|不要求|不以|不受")
    for clause, _, _ in _clauses(text):
        claim, value = _compact(clause), _compact(condition)
        offset = claim.find(value)
        while offset >= 0:
            prefix, suffix = claim[:offset], claim[offset + len(value):]
            changed = re.match(
                r"^[‘’“”\"'，,:：]*(?:(?:这一|该|此|这个|上述)(?:要求|条件|前提))?"
                r"(?:之前|以前|前|并不是|并非|并不(?:是|要求|适用|需要)|不是|无需|无须|不必"
                r"|不需|无关|不构成|不作为|不属于|不适用|不要求|不用|不再(?:需要|要求|受)"
                r"|可有可无)", suffix
            )
            if not negative.search(prefix) and not changed:
                return True
            offset = claim.find(value, offset + 1)
    return False


def graph_condition_diagnostics(draft, evidence, paths) -> tuple[dict, ...]:
    """任一已完整引用路径的条件可分句说明，但每句必须有该 hop 自己的原文出处。"""
    by_chunk = {item.chunk_id: item.citation_id for item in evidence}
    citations = [citation for statement in draft.statements for citation in statement.citations]
    failures = []
    for path in paths:
        conditions = verified_path_conditions(evidence, path)
        if conditions is None or not _path_sources_cited(citations, by_chunk, path):
            continue
        missing = []
        for condition in conditions:
            source = path.hops[condition["hop_index"]].source
            own_source = [index for index, statement in enumerate(draft.statements) if any(
                citation.citation_id == condition["citation_id"]
                and _compact(source.source_quote) in _compact(citation.quote)
                for citation in statement.citations
            )]
            if not any(_condition_in_text(draft.statements[index].text, condition["condition_text"])
                       for index in own_source):
                missing.append({**condition, "source_statement_indices": own_source})
        if not missing:
            return ()
        failures.append({"path_id": path.path_id,
                         "required_citation_ids": [by_chunk[value]
                                                   for value in path.required_chunk_ids],
                         "missing_conditions": missing})
    # 只反馈一条已有完整引用候选，不把不同 alternative 的条件合并成新要求。
    if not failures:
        return ()
    return (min(failures, key=lambda item: (len(item["missing_conditions"]), item["path_id"])),)


def expand_selected_path_quotes(draft, evidence, paths):
    """只扩展已明确选齐来源的路径引用，连续原文承担完整的审计范围。

    调用前必须验证草稿原始引用；不增加 CID、陈述或事实，不救回无效引文。
    同 CID 已有 quote 也保留在切片中，不能因补路径出处而丢失已有陈述依据。
    """
    by_chunk = {item.chunk_id: item for item in evidence}
    by_id = {item.citation_id: item for item in evidence}
    selected = {c.citation_id for statement in draft.statements for c in statement.citations}
    for path in paths:
        if not all(value in by_chunk and by_chunk[value].citation_id in selected
                   for value in path.required_chunk_ids):
            continue
        spans, valid = {}, True
        sources = [hop.source for hop in path.hops]
        sources.extend(source for bridge in path.bridges for source in bridge.sources)
        for source in sources:
            item = by_chunk.get(source.chunk_id)
            if item is None or not (
                0 <= source.quote_start < source.quote_end <= len(item.content)
                and hashlib.sha256(item.content.encode()).hexdigest() == source.content_sha256
                and all(getattr(source, key) == getattr(item, key) for key in (
                    "knowledge_base_id", "document_id", "revision_id", "generation_id",
                ))
                and item.content[source.quote_start:source.quote_end] == source.source_quote
                and source.quote_end == source.quote_start + len(source.source_quote)
            ):
                valid = False
                break
            old = spans.setdefault(item.citation_id, (source.quote_start, source.quote_end))
            spans[item.citation_id] = (
                min(old[0], source.quote_start), max(old[1], source.quote_end)
            )
        if not valid or any(end - start > 2000 for start, end in spans.values()):
            continue
        statements = []
        for statement in draft.statements:
            citations, seen = [], set()
            for citation in statement.citations:
                quote = citation.quote
                if citation.citation_id in spans:
                    item = by_id[citation.citation_id]
                    left, right = spans[citation.citation_id]
                    positions, offset = [], item.content.find(quote)
                    while offset >= 0:
                        positions.append((min(left, offset), max(right, offset + len(quote))))
                        offset = item.content.find(quote, offset + 1)
                    if not positions:
                        valid = False
                        break
                    start, end = min(positions, key=lambda span: (span[1] - span[0], span[0]))
                    if end - start > 2000:
                        valid = False
                        break
                    quote = item.content[start:end]
                key = (citation.citation_id, quote)
                if key not in seen:
                    citations.append({"citation_id": citation.citation_id, "quote": quote})
                    seen.add(key)
            if not valid:
                break
            statements.append({"text": statement.text, "citations": citations})
        if valid:
            return DraftAnswer.model_validate({"status": draft.status, "statements": statements})
    return draft


def _same_topic(first, second):
    document, title, subject, _ = first
    other_document, other_title, other_subject, _ = second
    return document == other_document or any(
        title[offset:offset + 4] in other_title for offset in range(max(0, len(title) - 3))
    ) or bool(subject and other_subject and (
        subject == other_subject or any(subject[offset:offset + 4] in other_subject
            for offset in range(max(0, len(subject) - 3)))
    ))


def _explicit_list_subject(text):
    operator = re.search(r"(?:至少)?(?:说明|包含|包括|保存|填写|写明)", text)
    if operator is None:
        return ""
    subject = text[:operator.start()].strip()
    subject = re.sub(r"(?:还|也|应当|需要|应|需|须|至少)+$", "", subject).strip()
    return subject if len(subject) >= 4 and not re.search(r"[。；！？\n]", subject) else ""


def source_scope_conflicts(question, draft, evidence, paths=()) -> tuple[dict, ...]:
    """只拦截信息项问题中明显来自不同主题的独立清单混合。

相同主题的补充、明确汇总要求和完整已验证跨文档来源组保留；这里不
宣称能识别所有主题或完成实体共指。
"""
    if not re.search(r"包含|包括|哪些信息|哪些字段|有哪些|写明什么", question):
        return ()
    if re.search(r"汇总|综合|合并|跨文档|各份|多份|对照|不同文档", question):
        return ()
    by_id = {item.citation_id: item for item in evidence}
    by_chunk = {item.chunk_id: item.citation_id for item in evidence}
    previous_lists = []
    def conflict(kind, index, subject, first, second, previous_index=None):
        return ({"kind": kind, "statement_index": index,
                 "previous_statement_index": previous_index, "claim_subject": subject,
                 "sources": [{"citation_id": row[3],
                              "source_document": list(source_document_key(by_id[row[3]])),
                              "document_title": by_id[row[3]].document_title,
                              "list_subject": row[2][:256]} for row in (first, second)]},)

    for statement_index, statement in enumerate(draft.statements):
        covered_groups = [
            {by_chunk[value] for value in path.required_chunk_ids}
            for path in paths if all(value in by_chunk for value in path.required_chunk_ids)
            and _path_sources_cited(statement.citations, by_chunk, path)
        ]
        lists = []
        for citation in statement.citations:
            text = citation.quote
            enumeration = re.search(r"(至少说明|应说明|包含|包括|保存|填写)([^。；\n]+)", text)
            if enumeration is None or len(re.findall(r"、|和|与|及|或", enumeration[2])) < 2:
                continue
            item = by_id[citation.citation_id]
            title = re.sub(r"\.[a-zA-Z0-9]+$", "", item.document_title)
            title = re.sub(r"编写指南|操作指南|字段要求|补充说明|管理规范|信息要求", "", title)
            subject = text[:enumeration.start()].strip()
            lists.append((item.document_id, title, subject, item.citation_id))
        for index, (document, title, subject, cid) in enumerate(lists):
            for other in lists[index + 1:]:
                other_cid = other[3]
                if any({cid, other_cid} <= group for group in covered_groups):
                    continue
                if not _same_topic((document, title, subject, cid), other):
                    return conflict("independent_list_topics", statement_index,
                                    _explicit_list_subject(statement.text),
                                    (document, title, subject, cid), other)
        claim_subject = _explicit_list_subject(statement.text)
        if claim_subject:
            for previous_index, previous_subject, previous in previous_lists:
                if claim_subject == previous_subject:
                    for first in lists:
                        for second in previous:
                            if not _same_topic(first, second):
                                return conflict("same_claim_cross_statement", statement_index,
                                                claim_subject, first, second, previous_index)
            previous_lists.append((statement_index, claim_subject, lists))
    return ()


def source_scope_ambiguous(question, draft, evidence, paths=()) -> bool:
    return bool(source_scope_conflicts(question, draft, evidence, paths))
