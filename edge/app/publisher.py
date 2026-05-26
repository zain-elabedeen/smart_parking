from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from .local_queue import LocalEventQueue

logger = logging.getLogger(__name__)


class ProducerClient(Protocol):
    """Protocol for publisher implementations.

    Input:
        Implementations accept a topic, partition key, and JSON payload.

    Output:
        Raises on delivery failure; returns ``None`` on accepted delivery.
    """

    def publish(self, topic: str, key: str, payload: str) -> None:
        """Publish one JSON payload.

        Args:
            topic: Destination topic name.
            key: Partition key or routing key.
            payload: Serialized JSON payload.

        Returns:
            ``None``. Implementations raise an exception on failure.
        """
        ...


class KafkaProducerClient:
    """Kafka implementation of the producer protocol.

    Input:
        Created with Kafka bootstrap server addresses.

    Output:
        Sends JSON payloads to Kafka and raises if delivery fails.
    """

    def __init__(self, bootstrap_servers: str) -> None:
        """Create an idempotent Kafka producer client.

        Args:
            bootstrap_servers: Kafka bootstrap server list.

        Returns:
            ``None``.
        """
        try:
            from confluent_kafka import Producer
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("confluent-kafka is required for Kafka publishing") from exc
        self._producer = Producer({"bootstrap.servers": bootstrap_servers, "enable.idempotence": True})

    def publish(self, topic: str, key: str, payload: str) -> None:
        """Publish one message to Kafka and wait for delivery acknowledgement.

        Args:
            topic: Kafka topic name.
            key: Kafka message key.
            payload: JSON payload string.

        Returns:
            ``None``. Raises ``RuntimeError`` when Kafka reports delivery
            failure.
        """
        delivery_errors: list[object] = []

        def on_delivery(error, _message) -> None:
            """Capture Kafka delivery errors from the async callback.

            Args:
                error: Error object supplied by confluent-kafka, if any.
                _message: Delivered Kafka message metadata, unused here.

            Returns:
                ``None``.
            """
            if error is not None:
                delivery_errors.append(error)

        self._producer.produce(topic=topic, key=key, value=payload, on_delivery=on_delivery)
        self._producer.flush(5.0)
        if delivery_errors:
            raise RuntimeError(f"Kafka delivery failed: {delivery_errors[0]}")

    def flush(self, timeout: float = 5.0) -> None:
        """Wait for buffered Kafka messages to finish delivery.

        Args:
            timeout: Maximum number of seconds to wait.

        Returns:
            ``None``.
        """
        self._producer.flush(timeout)


@dataclass(slots=True)
class QueueBackedPublisher:
    """Publisher facade that writes through the durable local queue.

    Input fields:
        queue: Local SQLite-backed queue.
        producer: Optional producer client; ``None`` means queue-only mode.

    Output:
        Enqueues all events and flushes due events when a producer is present.
    """

    queue: LocalEventQueue
    producer: ProducerClient | None

    def enqueue(self, *, event_id: str, topic: str, payload_json: str) -> bool:
        """Store an event in the local durable queue.

        Args:
            event_id: Stable idempotency key.
            topic: Destination topic.
            payload_json: Serialized event payload.

        Returns:
            ``True`` if inserted, ``False`` if already present.
        """
        return self.queue.enqueue(event_id, topic, payload_json)

    def flush_pending(self, *, key: str, limit: int = 100) -> int:
        """Try to publish due pending events from the local queue.

        Args:
            key: Partition/routing key to use for each published message.
            limit: Maximum due events to attempt.

        Returns:
            Number of events successfully marked as sent.
        """
        sent = 0
        if self.producer is None:
            logger.info("publisher flush skipped; producer disabled pending_limit=%s", limit)
            return sent

        for event in self.queue.due_pending(limit=limit):
            try:
                self.producer.publish(event.topic, key, event.payload_json)
            except Exception:
                logger.exception("publisher failed event_id=%s topic=%s", event.id, event.topic)
                self.queue.mark_failed(event.id)
                continue
            self.queue.mark_sent(event.id)
            sent += 1
            logger.info("publisher sent event_id=%s topic=%s", event.id, event.topic)
        return sent
