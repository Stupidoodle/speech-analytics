"""Core context management functionality."""
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import asyncio
from dataclasses import dataclass, field

from src.conversation.types import Document
from src.document.processor import DocumentProcessor
from src.events.bus import EventBus
from src.events.types import Event, EventType
from src.conversation.manager import ConversationManager


@dataclass
class ContextEntry:
    """Individual context entry with metadata."""
    content: Any
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    priority: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    references: Set[str] = field(default_factory=set)


@dataclass
class ContextUpdate:
    """Context update event with tracking."""
    content: Dict[str, Any]
    source: str
    priority: float
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False
    validation_needed: bool = False


class ContextStore:
    """Manages active context storage and retrieval."""

    def __init__(self, max_entries: int = 1000):
        """Initialize context store.

        Args:
            max_entries: Maximum number of context entries to maintain
        """
        self.max_entries = max_entries
        self._store: Dict[str, Dict[str, ContextEntry]] = {}
        self._updates: List[ContextUpdate] = []
        self._references: Dict[str, Set[str]] = {}

    async def add_entry(
            self,
            context_id: str,
            key: str,
            entry: ContextEntry
    ) -> None:
        """Add context entry.

        Args:
            context_id: Context identifier
            key: Entry key
            entry: Context entry
        """
        if context_id not in self._store:
            self._store[context_id] = {}

        # Check size limit
        if len(self._store[context_id]) >= self.max_entries:
            # Remove oldest entry
            oldest_key = min(
                self._store[context_id].keys(),
                key=lambda k: self._store[context_id][k].timestamp
            )
            await self.remove_entry(context_id, oldest_key)

        # Add new entry
        self._store[context_id][key] = entry

        # Update references
        for ref in entry.references:
            if ref not in self._references:
                self._references[ref] = set()
            self._references[ref].add(f"{context_id}:{key}")

    async def get_entry(
            self,
            context_id: str,
            key: str
    ) -> Optional[ContextEntry]:
        """Get context entry.

        Args:
            context_id: Context identifier
            key: Entry key

        Returns:
            Context entry if found
        """
        return self._store.get(context_id, {}).get(key)

    async def remove_entry(
            self,
            context_id: str,
            key: str
    ) -> None:
        """Remove context entry.

        Args:
            context_id: Context identifier
            key: Entry key
        """
        if context_id in self._store and key in self._store[context_id]:
            entry = self._store[context_id][key]

            # Remove references
            for ref in entry.references:
                if ref in self._references:
                    self._references[ref].remove(f"{context_id}:{key}")
                    if not self._references[ref]:
                        del self._references[ref]

            del self._store[context_id][key]

    async def get_references(
            self,
            reference: str
    ) -> List[ContextEntry]:
        """Get entries referencing a key.

        Args:
            reference: Reference key

        Returns:
            List of referencing entries
        """
        entries = []
        if reference in self._references:
            for ref in self._references[reference]:
                context_id, key = ref.split(":")
                entry = await self.get_entry(context_id, key)
                if entry:
                    entries.append(entry)
        return entries


class ContextManager:
    """Manages context integration and updates."""

    def __init__(
            self,
            event_bus: EventBus,
            conversation_manager: ConversationManager,
            document_processor: DocumentProcessor,
            store: Optional[ContextStore] = None
    ):
        """Initialize context manager.

        Args:
            event_bus: Event bus instance
            conversation_manager: Conversation manager instance
            document_processor: Document processor instance
            store: Optional context store instance
        """
        self._updates: List[ContextUpdate] = []
        self.event_bus = event_bus
        self.conversation = conversation_manager
        self.doc_processor = document_processor
        self.store = store or ContextStore()

        # Set up event handlers
        self.event_bus.subscribe(
            EventType.DOCUMENT_PROCESSED,
            self._handle_document_processed
        )
        self.event_bus.subscribe(
            EventType.TRANSCRIPT,
            self._handle_transcript
        )
        self.event_bus.subscribe(
            EventType.CONTEXT_UPDATE,
            self._handle_context_update
        )

    async def add_document(
            self,
            context_id: str,
            document: Document,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add document to context.

        Args:
            context_id: Context identifier
            document: Document to add
            metadata: Optional document metadata
        """
        # Process document
        # TODO: ProcessingContext
        async for result in self.doc_processor.process_document(
                document.content,
                document.doc_type
        ):
            # Create context entry
            entry = ContextEntry(
                content=result.analysis,
                source=document.name,
                priority=2.0,  # Higher priority for documents
                metadata={
                    "doc_type": document.doc_type.value,
                    "format": document.format,
                    **(metadata or {})
                },
                references={document.name}
            )

            # Add to store
            key = f"doc_{document.name}_{datetime.now().isoformat()}"
            await self.store.add_entry(context_id, key, entry)

            # Emit event
            await self.event_bus.publish(Event(
                type=EventType.CONTEXT_UPDATE,
                data={
                    "context_id": context_id,
                    "key": key,
                    "source": document.name,
                    "doc_type": document.doc_type.value
                }
            ))

    async def update_context(
            self,
            context_id: str,
            updates: Dict[str, Any],
            source: str,
            priority: float = 1.0
    ) -> None:
        """Update context with new information.

        Args:
            context_id: Context identifier
            updates: Context updates
            source: Update source
            priority: Update priority
        """
        update = ContextUpdate(
            content=updates,
            source=source,
            priority=priority
        )
        self._updates.append(update)

        # Create context entry
        entry = ContextEntry(
            content=updates,
            source=source,
            priority=priority,
            metadata={"update_type": "dynamic"}
        )

        # Add to store
        key = f"update_{source}_{datetime.now().isoformat()}"
        await self.store.add_entry(context_id, key, entry)

        # Emit event
        await self.event_bus.publish(Event(
            type=EventType.CONTEXT_UPDATE,
            data={
                "context_id": context_id,
                "key": key,
                "source": source,
                "priority": priority
            }
        ))

    async def get_context(
            self,
            context_id: str,
            source_filter: Optional[str] = None,
            min_priority: float = 0.0
    ) -> Dict[str, ContextEntry]:
        """Get current context entries.

        Args:
            context_id: Context identifier
            source_filter: Optional source filter
            min_priority: Minimum priority threshold

        Returns:
            Dictionary of context entries
        """
        if context_id not in self.store._store:
            return {}

        entries = {}
        for key, entry in self.store._store[context_id].items():
            if entry.priority >= min_priority:
                if not source_filter or entry.source == source_filter:
                    entries[key] = entry

        return entries

    async def _handle_document_processed(self, event: Event) -> None:
        """Handle document processed events.

        Args:
            event: Document processed event
        """
        if "result" in event.data:
            context_id = event.data.get("context_id")
            if context_id:
                await self.update_context(
                    context_id=context_id,
                    updates=event.data["result"],
                    source="document_processor",
                    priority=1.5
                )

    async def _handle_transcript(self, event: Event) -> None:
        """Handle transcript events.

        Args:
            event: Transcript event
        """
        if "text" in event.data:
            context_id = event.data.get("context_id")
            if context_id:
                await self.update_context(
                    context_id=context_id,
                    updates={"transcript": event.data["text"]},
                    source="transcription",
                    priority=1.0
                )

    async def _handle_context_update(self, event: Event) -> None:
        """Handle context update events.

        Args:
            event: Context update event
        """
        context_id = event.data.get("context_id")
        if context_id:
            # Get existing context
            # current = await self.get_context(context_id)

            # Apply updates
            updates = event.data.get("updates", {})
            if updates:
                await self.update_context(
                    context_id=context_id,
                    updates=updates,
                    source=event.data.get("source", "unknown"),
                    priority=event.data.get("priority", 1.0)
                )


class DynamicContextUpdater:
    """Handles dynamic context updates and validation."""

    def __init__(
            self,
            context_manager: ContextManager,
            update_interval: float = 1.0
    ):
        """Initialize dynamic updater.

        Args:
            context_manager: Context manager instance
            update_interval: Update interval in seconds
        """
        self.context = context_manager
        self.update_interval = update_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start dynamic updates."""
        self._running = True
        self._task = asyncio.create_task(self._update_loop())

    async def stop(self) -> None:
        """Stop dynamic updates."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _update_loop(self) -> None:
        """Main update loop."""
        while self._running:
            # Process pending updates
            updates = [u for u in self.context._updates if not u.processed]
            for update in updates:
                if update.validation_needed:
                    # Validate update
                    if await self._validate_update(update):
                        update.processed = True
                else:
                    update.processed = True

            await asyncio.sleep(self.update_interval)

    async def _validate_update(self, update: ContextUpdate) -> bool:
        """Validate context update.

        Args:
            update: Update to validate

        Returns:
            Whether update is valid
        """
        # Implement validation logic
        # For now, accept all updates
        return True