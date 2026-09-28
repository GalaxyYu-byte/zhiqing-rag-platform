"""按请求建立短连接；启动时不连接数据库。"""

from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from zhiqing_rag.core.config import Settings


@contextmanager
def database_connection(settings: Settings):
    with psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_username,
        password=settings.db_password.get_secret_value(),
        connect_timeout=max(1, int(settings.service_timeout_seconds)),
        options="-c statement_timeout=15000 -c lock_timeout=5000",
        row_factory=dict_row,
    ) as connection:
        yield connection
