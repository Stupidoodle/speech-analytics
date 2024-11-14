"""Core context management functionality."""
from typing import Dict, Any, List, Optional, Set, AsyncIterator
from datetime import datetime, timedelta
import asyncio
import uuid

from src.events.bus import EventBus
from src.events.types import Event, EventType
from src.conversation.manager import ConversationManager
from src.analysis.engine import AnalysisEngine
from src.document.processor import DocumentProcessor

from .types import (
    ContextLevel,
    ContextSource,
    ContextState,
    ContextMetadata,
    ContextEntry,
    ContextQuery,
    ContextUpdate,
    ContextConfig
)
from .exceptions import (
    ContextError,
    ContextNotFoundError,
    ContextValidationError,
    ContextUpdateError,
    ContextQueryError
)


class ContextManager:
    """Manages context storage, retrieval, and updates."""

    def __init__(
        self,
        event_bus: EventBus,
        conversation_manager: ConversationManager,
        analysis_engine: AnalysisEngine,
        document_processor: DocumentProcessor,
        config: Optional[ContextConfig] = None
    ):
        """Initialize context manager.

        Args:
            event_bus: Event bus instance
            conversation_manager: For AI assistance
            analysis_engine: For context analysis
            document_processor: For document handling
            config: Optional configuration
        """
        self.event_bus = event_bus
        self.conversation = conversation_manager
        self.analysis = analysis_engine
        self.doc_processor = document_processor
        self.config = config or ContextConfig(
            enabled_sources={
                ContextSource.CONVERSATION,
                ContextSource.DOCUMENT,
                ContextSource.ANALYSIS
            }
        )

        # Context storage
        self._store: Dict[str, ContextEntry] = {}
        self._index: Dict[str, Set[str]] = {
            source.value: set() for source in ContextSource
        }
        self._tags: Dict[str, Set[str]] = {}
        self._references: Dict[str, Set[str]] = {}

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start context management."""
        if not self._running:
            self._running = True

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop()
            )

            # Subscribe to events
            self._setup_event_handlers()

    async def stop(self) -> None:
        """Stop context management."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def add_context(
        self,
        content: Any,
        metadata: ContextMetadata
    ) -> ContextEntry:
        """Add new context entry.

        Args:
            content: Context content
            metadata: Context metadata

        Returns:
            Created context entry

        Raises:
            ContextError: If addition fails
        """
        try:
            # Validate source
            if metadata.source not in self.config.enabled_sources:
                raise ContextError(f"Source not enabled: {metadata.source}")

            # Create entry
            entry_id = str(uuid.uuid4())
            entry = ContextEntry(
                id=entry_id,
                content=content,
                metadata=metadata
            )

            # Validate if required
            if self.config.validation_required:
                validation_result = await self._validate_entry(entry)
                entry.validation_info = validation_result
                if validation_result.get("errors"):
                    raise ContextValidationError(
                        "Context validation failed",
                        validation_result["errors"]
                    )

            # Store entry
            await self._store_entry(entry)

            # Emit event
            await self.event_bus.publish(Event(
                type=EventType.CONTEXT_UPDATE,
                data={
                    "action": "add",
                    "entry_id": entry_id,
                    "source": metadata.source
                }
            ))

            return entry

        except Exception as e:
            if isinstance(e, ContextError):
                raise
            raise ContextError(f"Failed to add context: {str(e)}")

    async def get_context(
        self,
        query: ContextQuery
    ) -> List[ContextEntry]:
        """Get context entries matching query.

        Args:
            query: Context query

        Returns:
            Matching context entries

        Raises:
            ContextQueryError: If query invalid
        """
        try:
            entries = set(self._store.keys())

            # Filter by source
            if query.sources:
                source_entries = set()
                for source in query.sources:
                    source_entries.update(self._index[source.value])
                entries &= source_entries

            # Filter by level
            if query.levels:
                entries = {
                    entry_id for entry_id in entries
                    if self._store[entry_id].metadata.level in query.levels
                }

            # Filter by state
            if query.states:
                entries = {
                    entry_id for entry_id in entries
                    if self._store[entry_id].metadata.state in query.states
                }

            # Filter by tags
            if query.tags:
                tag_entries = set()
                for tag in query.tags:
                    tag_entries.update(self._tags.get(tag, set()))
                entries &= tag_entries

            # Filter by time range
            if query.start_time or query.end_time:
                entries = {
                    entry_id for entry_id in entries
                    if self._in_time_range(
                        self._store[entry_id],
                        query.start_time,
                        query.end_time
                    )
                }

            # Apply limit
            results = sorted(
                [self._store[id] for id in entries],
                key=lambda x: x.metadata.timestamp,
                reverse=True
            )
            if query.limit:
                results = results[:query.limit]

            return results

        except Exception as e:
            raise ContextQueryError(
                f"Query failed: {str(e)}",
                query.model_dump()
            )

    async def update_context(
        self,
        update: ContextUpdate
    ) -> ContextEntry:
        """Update context entry.

        Args:
            update: Context update

        Returns:
            Updated entry

        Raises:
            ContextUpdateError: If update fails
        """
        try:
            # Get existing entry
            entry = self._store.get(update.entry_id)
            if not entry:
                raise ContextNotFoundError(update.entry_id)

            # Apply content update
            if update.content is not None:
                entry.content = update.content

            # Apply metadata updates
            if update.metadata_updates:
                for key, value in update.metadata_updates.items():
                    if hasattr(entry.metadata, key):
                        setattr(entry.metadata, key, value)

            # Update validation info
            if update.validation_info:
                entry.validation_info = update.validation_info

            # Update timestamp
            entry.metadata.timestamp = datetime.now()

            # Revalidate if required
            if self.config.validation_required:
                validation_result = await self._validate_entry(entry)
                entry.validation_info = validation_result
                if validation_result.get("errors"):
                    raise ContextValidationError(
                        "Updated context validation failed",
                        validation_result["errors"]
                    )

            # Update indexes
            await self._update_indexes(entry)

            # Emit event
            await self.event_bus.publish(Event(
                type=EventType.CONTEXT_UPDATE,
                data={
                    "action": "update",
                    "entry_id": entry.id
                }
            ))

            return entry

        except Exception as e:
            if isinstance(e, (ContextNotFoundError, ContextValidationError)):
                raise
            raise ContextUpdateError(
                f"Update failed: {str(e)}",
                update.entry_id,
                "update"
            )

    async def remove_context(
        self,
        entry_id: str
    ) -> None:
        """Remove context entry.

        Args:
            entry_id: Entry to remove

        Raises:
            ContextNotFoundError: If entry not found
        """
        if entry_id not in self._store:
            raise ContextNotFoundError(entry_id)

        entry = self._store[entry_id]

        # Remove from indexes
        self._index[entry.metadata.source.value].discard(entry_id)
        for tag in entry.metadata.tags:
            if tag in self._tags:
                self._tags[tag].discard(entry_id)
        for ref in entry.metadata.references:
            if ref in self._references:
                self._references[ref].discard(entry_id)

        # Remove entry
        del self._store[entry_id]

        # Emit event
        await self.event_bus.publish(Event(
            type=EventType.CONTEXT_UPDATE,
            data={
                "action": "remove",
                "entry_id": entry_id
            }
        ))

    async def _store_entry(
        self,
        entry: ContextEntry
    ) -> None:
        """Store context entry and update indexes.

        Args:
            entry: Entry to store
        """
        # Check size limit
        if len(self._store) >= self.config.max_entries:
            await self._archive_old_entries()

        # Store entry
        self._store[entry.id] = entry

        # Update indexes
        await self._update_indexes(entry)

    async def _update_indexes(
        self,
        entry: ContextEntry
    ) -> None:
        """Update context indexes.

        Args:
            entry: Entry to index
        """
        # Source index
        self._index[entry.metadata.source.value].add(entry.id)

        # Tag index
        for tag in entry.metadata.tags:
            if tag not in self._tags:
                self._tags[tag] = set()
            self._tags[tag].add(entry.id)

        # Reference index
        for ref in entry.metadata.references:
            if ref not in self._references:
                self._references[ref] = set()
            self._references[ref].add(entry.id)

    async def _validate_entry(
        self,
        entry: ContextEntry
    ) -> Dict[str, Any]:
        """Validate context entry.

        Args:
            entry: Entry to validate

        Returns:
            Validation results
        """
        # Get validation prompt based on source
        prompts = {
            ContextSource.CONVERSATION: "Validate conversation context",
            ContextSource.DOCUMENT: "Validate document context",
            ContextSource.ANALYSIS: "Validate analysis context",
            ContextSource.USER_INPUT: "Validate user input",
            ContextSource.SYSTEM: "Validate system context",
            ContextSource.EXTERNAL: "Validate external context"
        }
        prompt = prompts.get(
            entry.metadata.source,
            "Validate context"
        )

        # Create validation request
        validation_prompt = (
            f"{prompt}. Provide results as JSON:\n"
            "{\n"
            '  "is_valid": boolean,\n'
            '  "confidence": float,\n'
            '  "errors": [string],\n'
            '  "warnings": [string]\n'
            "}\n\n"
            f"Content: {entry.content}\n"
            f"Metadata: {entry.metadata.__dict__}"
        )

        # Get AI validation
        results = {"is_valid": True, "errors": [], "warnings": []}
        async for response in self.conversation.send_message(
            validation_prompt
        ):
            if response.text:
                try:
                    import json
                    results = json.loads(response.text)
                except json.JSONDecodeError:
                    continue

        return results

    def _in_time_range(
        self,
        entry: ContextEntry,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> bool:
        """Check if entry is in time range.

        Args:
            entry: Entry to check
            start_time: Range start
            end_time: Range end

        Returns:
            Whether entry is in range
        """
        if start_time and entry.metadata.timestamp < start_time:
            return False
        if end_time and entry.metadata.timestamp > end_time:
            return False
        return True

    async def _archive_old_entries(self) -> None:
        """Archive old context entries."""
        if not self.config.auto_archive:
            return

        # Sort by timestamp
        entries = sorted(
            self._store.values(),
            key=lambda x: x.metadata.timestamp
        )

        # Archive oldest entries until under limit
        while len(self._store) >= self.config.max_entries:
            entry = entries.pop(0)
            entry.metadata.state = ContextState.ARCHIVED

            # Remove from active indexes
            await self.remove_context(entry.id)

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                # Check retention period
                if self.config.retention_period:
                    cutoff = datetime.now() - timedelta(
                        days=self.config.retention_period
                    )
                    for entry_id in list(self._store.keys()):
                        entry = self._store[entry_id]
                        if entry.metadata.timestamp < cutoff:
                            await self.remove_context(entry_id)

                # Check expired entries
                for entry_id in list(self._store.keys()):
                    entry = self._store[entry_id]
                    if (entry.metadata.expiry and
                            datetime.now() > entry.metadata.expiry):
                        await self.remove_context(entry_id)

                await asyncio.sleep(3600)  # Check every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Cleanup error: {e}")
                await asyncio.sleep(60)

    def _setup_event_handlers(self) -> None:
        """Set up event handlers."""
        self.event_bus.subscribe(
            EventType.CONTEXT_UPDATE,
            self._handle_context_event
        )

    async def _handle_context_event(
        self,
        event: Event
    ) -> None:
        """Handle context events."""
        pass  # Implement event handling