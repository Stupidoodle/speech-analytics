# src/events/bus.py
from typing import Dict, Set, Callable, Coroutine, Any
import asyncio
import logging
from types import Event, EventType

logger = logging.getLogger(__name__)

CallbackType = Callable[[Event], Coroutine[Any, Any, None]]

class EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[EventType, Set[CallbackType]] = {event: set() for event in EventType}
        self._queue = asyncio.Queue()

    def subscribe(self, event_type: EventType, callback: CallbackType) -> None:
        """Subscribe to an event type."""
        self._subscribers[event_type].add(callback)

    def unsubscribe(self, event_type: EventType, callback: CallbackType) -> None:
        """Unsubscribe from an event type."""
        self._subscribers[event_type].discard(callback)

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers asynchronously."""
        await self._queue.put(event)

    async def start(self) -> None:
        """Start processing events from the queue."""
        while True:
            event = await self._queue.get()
            if event.type in self._subscribers:
                for callback in self._subscribers[event.type]:
                    try:
                        await callback(event)
                    except Exception as e:
                        logger.error(f"Error in {callback}: {e}")
            self._queue.task_done()
