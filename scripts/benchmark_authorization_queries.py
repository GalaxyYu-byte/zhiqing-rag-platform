"""只读比较同一会话的两次身份查询与合并查询，不输出身份或凭据。"""

import argparse
import json
import time
from statistics import median
from uuid import uuid4

from zhiqing_rag.core.config import get_settings
from zhiqing_rag.infrastructure.database.connection import (
    close_database_pool,
    database_connection,
)
from zhiqing_rag.modules.identity.auth_repository import AuthRepository


def separate_identity(repository, session_key):
    session = repository.session(session_key)
    row = (
        repository.identity(session["user_id"], session["tenant_id"], session["tenant_member_id"])
        if session
        else None
    )
    return session, row


def same_identity(separate, combined):
    old_session, old_row = separate
    session, row = combined
    if old_session is None:
        return session is None and row is None
    session_fields = ("user_id", "tenant_id", "tenant_member_id", "revoked_at", "expires_at")
    return (
        session is not None
        and all(old_session[key] == session[key] for key in session_fields)
        and (
            old_row is None
            and row is None
            or old_row is not None
            and row is not None
            and all(value == row[key] for key, value in old_row.items())
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, choices=range(1, 21), default=5)
    args = parser.parse_args()
    settings = get_settings()
    try:
        samples = {"separate": [], "combined": []}
        with database_connection(settings) as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            repository = AuthRepository(connection, settings.db_schema)
            assert repository.session_identity(uuid4()) == (None, None)
            selected = repository.query(
                "SELECT session_key FROM {schema}.auth_session "
                "WHERE revoked_at IS NULL AND expires_at > clock_timestamp() "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if selected is None:
                print(json.dumps({"status": "skipped", "reason": "no_current_session"}))
                return 0
            session_key = selected["session_key"]
            for _ in range(args.samples):
                started = time.perf_counter()
                separate = separate_identity(repository, session_key)
                samples["separate"].append(round((time.perf_counter() - started) * 1000, 1))
                started = time.perf_counter()
                combined = repository.session_identity(session_key)
                samples["combined"].append(round((time.perf_counter() - started) * 1000, 1))
                assert same_identity(separate, combined), "authorization changed during comparison"
        print(
            json.dumps(
                {
                    "status": "ok",
                    "result_equivalent": True,
                    "query_count_per_check": {"separate": 2, "combined": 1},
                    "samples_ms": samples,
                    "median_ms": {name: round(median(rows), 1) for name, rows in samples.items()},
                },
                indent=2,
            )
        )
        return 0
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}))
        return 1
    finally:
        close_database_pool(settings)


if __name__ == "__main__":
    raise SystemExit(main())
