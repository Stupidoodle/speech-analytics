from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Set, Callable, Optional

from src.conversation.roles import Role


class EventType(Enum):
    AUDIO_CHUNK = "audio_chunk"
    TRANSCRIPTION = "transcription"
    ASSISTANCE = "assistance"
    CONTEXT_UPDATE = "context_update"
    DOCUMENT_PROCESSED = "document_processed"
    ERROR = "error"


@dataclass
class Event:
    type: EventType
    data: Any
    timestamp: datetime
    role: Optional[Role] = None
    metadata: Optional[Dict[str, Any]] = None


class EventBus:
    def __init__(self):
        self._subscribers: Dict[EventType, Set[Callable]] = {
            event_type: set() for event_type in EventType
        }
        self._role_filters: Dict[Callable, Set[Role]] = {}

    async def publish(self, event: Event) -> None:
        """Publish event to relevant subscribers."""
        subscribers = self._subscribers[event.type]

        for subscriber in subscribers:
            if (event.role and
                subscriber in self._role_filters and
                    event.role not in self._role_filters[subscriber]):
                continue

            try:
                await subscriber(event)
            except Exception as e:
                await self.publish(Event(
                    type=EventType.ERROR,
                    data=str(e),
                    timestamp=datetime.now(),
                    metadata={"original_event": event.type.value}
                ))

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable,
        roles: Optional[Set[Role]] = None
    ) -> None:
        self._subscribers[event_type].add(callback)
        if roles:
            self._role_filters[callback] = roles
