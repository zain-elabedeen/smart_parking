from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OutboundEvent:
    """One persisted event waiting for cloud delivery.

    Input fields:
        id: Stable event id and SQLite primary key.
        topic: Destination Kafka topic.
        payload_json: Serialized event payload.
        status: Queue status, such as ``PENDING``, ``SENT``, or ``FAILED``.
        attempts: Number of delivery attempts recorded.
        created_at: Time the event was first stored.
        last_attempt_at: Time the most recent send attempt happened, if any.

    Output:
        Returned by queue reads and passed to publisher retry logic.
    """

    id: str
    topic: str
    payload_json: str
    status: str
    attempts: int
    created_at: datetime
    last_attempt_at: datetime | None


class LocalEventQueue:
    """SQLite-backed durable outbound event queue.

    Input:
        Created with a SQLite file path and retry-backoff settings.

    Output:
        Persists idempotent events and returns due events for publishing.
    """

    def __init__(self, sqlite_path: str | Path, base_backoff_seconds: float = 2.0, max_backoff_seconds: float = 300.0) -> None:
        """Create or open the local event queue database.

        Args:
            sqlite_path: Path to the SQLite database file.
            base_backoff_seconds: Initial retry delay after a failed attempt.
            max_backoff_seconds: Upper bound for exponential retry delay.

        Returns:
            ``None``.
        """
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._initialize()

    def enqueue(self, event_id: str, topic: str, payload_json: str, created_at: datetime | None = None) -> bool:
        """Insert an outbound event if it is not already present.

        Args:
            event_id: Stable idempotency key for the event.
            topic: Destination topic.
            payload_json: Serialized JSON payload.
            created_at: Optional creation time; defaults to current UTC time.

        Returns:
            ``True`` if inserted, or ``False`` if an event with this id exists.
        """
        created_at = created_at or _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO outbound_events (id, topic, payload_json, status, attempts, created_at, last_attempt_at)
                VALUES (?, ?, ?, 'PENDING', 0, ?, NULL)
                """,
                (event_id, topic, payload_json, _format_dt(created_at)),
            )
            inserted = cur.rowcount == 1
            logger.info("local queue enqueue event_id=%s topic=%s inserted=%s", event_id, topic, inserted)
            return inserted

    def due_pending(self, limit: int = 100, now: datetime | None = None) -> list[OutboundEvent]:
        """Return pending events whose retry backoff has elapsed.

        Args:
            limit: Maximum number of events to return.
            now: Optional current time for deterministic tests.

        Returns:
            Oldest due ``OutboundEvent`` records first.
        """
        now = now or _utc_now()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, topic, payload_json, status, attempts, created_at, last_attempt_at
                FROM outbound_events
                WHERE status = 'PENDING'
                ORDER BY created_at ASC
                """
            ).fetchall()

        due: list[OutboundEvent] = []
        for row in rows:
            event = _event_from_row(row)
            if self._is_due(event, now):
                due.append(event)
                if len(due) >= limit:
                    break
        return due

    def record_attempt(self, event_id: str, *, success: bool, permanent_failure: bool = False, now: datetime | None = None) -> None:
        """Record the result of a delivery attempt.

        Args:
            event_id: Event id to update.
            success: Whether the delivery succeeded.
            permanent_failure: Whether a failed delivery should become
                ``FAILED`` instead of remaining retryable.
            now: Optional attempt timestamp; defaults to current UTC time.

        Returns:
            ``None``.
        """
        now = now or _utc_now()
        status = "SENT" if success else ("FAILED" if permanent_failure else "PENDING")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE outbound_events
                SET status = ?, attempts = attempts + 1, last_attempt_at = ?
                WHERE id = ?
                """,
                (status, _format_dt(now), event_id),
            )
        logger.info("local queue attempt event_id=%s status=%s success=%s permanent_failure=%s", event_id, status, success, permanent_failure)

    def mark_sent(self, event_id: str, now: datetime | None = None) -> None:
        """Mark an event as successfully sent.

        Args:
            event_id: Event id to update.
            now: Optional send timestamp.

        Returns:
            ``None``.
        """
        self.record_attempt(event_id, success=True, now=now)

    def mark_failed(self, event_id: str, *, permanent: bool = False, now: datetime | None = None) -> None:
        """Record a failed send attempt.

        Args:
            event_id: Event id to update.
            permanent: Whether the event should stop retrying.
            now: Optional failure timestamp.

        Returns:
            ``None``.
        """
        self.record_attempt(event_id, success=False, permanent_failure=permanent, now=now)

    def counts_by_status(self) -> dict[str, int]:
        """Count queued events grouped by status.

        Returns:
            Mapping from status string to row count.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM outbound_events
                GROUP BY status
                """
            ).fetchall()
        return {row["status"]: int(row["count"]) for row in rows}

    def _initialize(self) -> None:
        """Create queue tables and indexes if they do not already exist.

        Returns:
            ``None``.
        """
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outbound_events (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_attempt_at TEXT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outbound_status_created ON outbound_events(status, created_at)")

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection configured for row-name access.

        Returns:
            ``sqlite3.Connection`` with ``row_factory`` set to ``sqlite3.Row``.
        """
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _is_due(self, event: OutboundEvent, now: datetime) -> bool:
        """Decide whether an event can be retried at the given time.

        Args:
            event: Pending event to inspect.
            now: Current timestamp for backoff comparison.

        Returns:
            ``True`` when the event is ready to send, otherwise ``False``.
        """
        if event.last_attempt_at is None:
            return True
        backoff = min(self.max_backoff_seconds, self.base_backoff_seconds * (2 ** max(0, event.attempts - 1)))
        return event.last_attempt_at + timedelta(seconds=backoff) <= now


def _event_from_row(row: sqlite3.Row) -> OutboundEvent:
    """Convert a SQLite row into an ``OutboundEvent``.

    Args:
        row: SQLite row from the ``outbound_events`` table.

    Returns:
        Parsed ``OutboundEvent`` instance.
    """
    return OutboundEvent(
        id=row["id"],
        topic=row["topic"],
        payload_json=row["payload_json"],
        status=row["status"],
        attempts=int(row["attempts"]),
        created_at=_parse_dt(row["created_at"]),
        last_attempt_at=_parse_dt(row["last_attempt_at"]) if row["last_attempt_at"] else None,
    )


def _utc_now() -> datetime:
    """Return the current UTC time.

    Returns:
        A timezone-aware ``datetime`` set to UTC.
    """
    return datetime.now(tz=timezone.utc)


def _format_dt(value: datetime) -> str:
    """Serialize a timestamp for SQLite storage.

    Args:
        value: Timestamp to serialize. Naive values are treated as UTC.

    Returns:
        ISO-8601 string normalized to UTC.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_dt(value: str) -> datetime:
    """Parse a timestamp read from SQLite.

    Args:
        value: ISO-8601 timestamp string.

    Returns:
        Timezone-aware UTC ``datetime``.
    """
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
