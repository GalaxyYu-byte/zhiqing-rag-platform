"""只读检查 Router 和 Graph 检索的 Cypher 编译，不执行检索或修改图谱。"""

import asyncio

from zq_rag_app.core.config import settings
from zq_rag_app.core.neo4j import close_neo4j, get_neo4j_session
from zq_rag_app.services.graph_retrieval_service import _GRAPH_CLAIM_SEARCH
from zq_rag_app.services.graph_routing_service import _ENTITY_COVERAGE


async def main() -> int:
    if not settings.neo4j_password:
        print("SKIP: Neo4j not configured")
        return 1
    try:
        async with asyncio.timeout(20):
            async with get_neo4j_session() as session:
                for name, query, parameters in [
                    ("entity_coverage", _ENTITY_COVERAGE, {
                        "entity_names": [], "entity_candidates": [], "kb_ids": [], "document_versions": [],
                    }),
                    ("graph_search", _GRAPH_CLAIM_SEARCH, {
                        "query_normalized": "", "kb_ids": [], "doc_ids": [],
                        "seed_entity_uids": [], "entity_limit": 20, "claim_limit": 500,
                    }),
                ]:
                    result = await session.run("EXPLAIN " + query, **parameters)
                    await result.consume()
                    print(f"{name}: Cypher compiled", flush=True)
        return 0
    except Exception as exc:
        # 不输出服务地址、认证数据或原始异常响应。
        print(f"FAIL: {type(exc).__name__}")
        return 1
    finally:
        await close_neo4j()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
