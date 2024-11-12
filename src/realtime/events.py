from typing import Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum
import asyncio


class EventType(Enum):
    TRANSCRIPT = "transcript"
    SUGGESTION = "suggestion"
    ANALYSIS = "analysis"
    ERROR = "error"
    AUDIO_LEVEL = "audio_level"
    SPEAKER_CHANGE = "speaker_change"
    QUESTION_DETECTED = "question_detected"
    CONTEXT_UPDATE = "context_update"


@dataclass
class Event:
    """Event data container."""
    type: EventType
    data: Dict[str, Any]
    timestamp: str


class EventEmitter:
    """Handles real-time event emission and subscription."""

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        self._queue = asyncio.Queue()
        self._tasks = []

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None]
    ) -> None:
        """Subscribe to an event type."""
        self._subscribers[event_type].append(callback)

    def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None]
    ) -> None:
        """Unsubscribe from an event type."""
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    async def emit(self, event: Event) -> None:
        """Emit an event to subscribers."""
        await self._queue.put(event)

    async def start_processing(self) -> None:
        """Start processing events."""
        self._tasks.append(
            asyncio.create_task(self._process_events())
        )

    async def stop_processing(self) -> None:
        """Stop processing events."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        # Clear the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _process_events(self) -> None:
        """Process events from queue."""
        while True:
            try:
                event = await self._queue.get()

                # Notify subscribers
                for callback in self._subscribers[event.type]:
                    try:
                        # Run callback in task to prevent blocking
                        asyncio.create_task(
                            self._run_callback(callback, event)
                        )
                    except Exception as e:
                        print(f"Error in event callback: {e}")

                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing event: {e}")
                await asyncio.sleep(0.1)

    async def _run_callback(
        self,
        callback: Callable[[Event], None],
        event: Event
    ) -> None:
        """Run callback safely."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            print(f"Error in callback: {e}")


# Global event emitter instance
event_emitter = EventEmitter()
