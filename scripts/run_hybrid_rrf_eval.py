"""Dense + BM25 归一化加权并叠加 RRF 的离线评估入口。"""

from __future__ import annotations

import asyncio
import sys

from zq_rag_app.evaluation.dense_offline import async_main


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    arguments = ["--strategy", "hybrid_rrf", *sys.argv[1:]]
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            raise SystemExit(runner.run(async_main(arguments)))
    raise SystemExit(asyncio.run(async_main(arguments)))
