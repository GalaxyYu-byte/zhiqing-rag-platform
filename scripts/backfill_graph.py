"""入口：uv run python scripts/backfill_graph.py [--dry-run|--run]。"""

from zhiqing_rag.workers.backfill_graph import main

if __name__ == "__main__":
    main()
