"""复用持久化 outbox 的图谱清理队列；文档记录删除后事件仍可独立重放。"""

import random
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from zhiqing_rag.modules.documents import OutboxEvent


@dataclass(frozen=True)
class CleanupLease:
    event_id: int
    tenant_id: int
    event_type: str
    payload: dict
    owner: str


class GraphCleanupQueue:
    def __init__(self, engine, lease_seconds: int):
        self.engine = engine
        self.lease_seconds = lease_seconds

    def claim(self, owner: str) -> CleanupLease | None:
        with Session(self.engine) as session, session.begin():
            event = session.scalar(
                select(OutboxEvent)
                .where(
                    OutboxEvent.event_type.in_(["GRAPH_PURGE_GENERATION", "GRAPH_PURGE_DOCUMENT"]),
                    or_(
                        (OutboxEvent.status == "PENDING")
                        & (OutboxEvent.available_at <= func.clock_timestamp()),
                        (OutboxEvent.status == "PUBLISHING")
                        & (OutboxEvent.locked_until <= func.clock_timestamp()),
                    ),
                )
                .order_by(OutboxEvent.id)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if event is None:
                return None
            now = session.scalar(select(func.clock_timestamp()))
            event.status = "PUBLISHING"
            event.attempt_count += 1
            event.locked_by = owner
            event.locked_until = now + timedelta(seconds=self.lease_seconds)
            return CleanupLease(event.id, event.tenant_id, event.event_type, event.payload, owner)

    def complete(self, lease: CleanupLease) -> None:
        with Session(self.engine) as session, session.begin():
            event = self._locked(session, lease)
            event.status = "PUBLISHED"
            event.published_at = session.scalar(select(func.clock_timestamp()))
            event.locked_by = event.locked_until = None
            event.last_error_code = None

    def fail(self, lease: CleanupLease, code: str) -> None:
        with Session(self.engine) as session, session.begin():
            event = self._locked(session, lease)
            now = session.scalar(select(func.clock_timestamp()))
            event.status = "PENDING"
            event.available_at = now + timedelta(
                seconds=min(3600, 15 * 2 ** min(event.attempt_count - 1, 8)) + random.uniform(0, 5)
            )
            event.locked_by = event.locked_until = None
            event.last_error_code = code

    @staticmethod
    def _locked(session: Session, lease: CleanupLease) -> OutboxEvent:
        event = session.scalar(
            select(OutboxEvent)
            .where(
                OutboxEvent.id == lease.event_id,
                OutboxEvent.tenant_id == lease.tenant_id,
                OutboxEvent.status == "PUBLISHING",
                OutboxEvent.locked_by == lease.owner,
                OutboxEvent.locked_until > func.clock_timestamp(),
            )
            .with_for_update()
        )
        if event is None:
            raise ValueError("GRAPH_CLEANUP_LEASE_LOST")
        return event
