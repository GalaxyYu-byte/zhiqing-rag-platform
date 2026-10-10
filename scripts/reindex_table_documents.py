"""历史表格索引诊断/重建入口；默认仅预览，不调用模型、不写数据库。"""

from zhiqing_rag.workers.reindex_tables import main

if __name__ == "__main__":
    main()
