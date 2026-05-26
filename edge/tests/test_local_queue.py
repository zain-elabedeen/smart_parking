from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.local_queue import LocalEventQueue


def test_enqueue_is_idempotent(tmp_path) -> None:
    """Verify inserting the same event id twice creates only one row.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    queue = LocalEventQueue(tmp_path / "events.db")

    assert queue.enqueue("event-1", "plate-crops", "{}") is True
    assert queue.enqueue("event-1", "plate-crops", "{}") is False

    assert queue.counts_by_status() == {"PENDING": 1}


def test_due_pending_respects_exponential_backoff(tmp_path) -> None:
    """Verify retry reads wait until exponential backoff has elapsed.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    queue = LocalEventQueue(tmp_path / "events.db", base_backoff_seconds=10)
    now = datetime.now(tz=timezone.utc)
    queue.enqueue("event-1", "plate-crops", "{}", created_at=now)
    queue.mark_failed("event-1", now=now)

    assert queue.due_pending(now=now + timedelta(seconds=9)) == []
    assert [event.id for event in queue.due_pending(now=now + timedelta(seconds=10))] == ["event-1"]


def test_mark_sent_updates_status(tmp_path) -> None:
    """Verify successful delivery changes queue status to SENT.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    queue = LocalEventQueue(tmp_path / "events.db")
    queue.enqueue("event-1", "plate-crops", "{}")
    queue.mark_sent("event-1")

    assert queue.counts_by_status() == {"SENT": 1}
