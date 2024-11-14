"""Integration with conversation and document processing."""
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime
import asyncio

from src.conversation.manager import ConversationManager
from src.document.processor import DocumentProcessor
from src.events.bus import EventBus
from src.events.types import Event, EventType

from .manager import ContextManager
from .types import (
    ContextSource,
    ContextPriority,
    ContextState,
    ContextMetadata,
    ContextConfig,
    ContextEntry
)
from .exceptions import (
    ContextError,
    ContextUpdateError
)


class ContextIntegration:
    """Handles real-time context integration."""

    def __init__(
            self,
            event_bus: EventBus,
            conversation_manager: ConversationManager,
            document_processor: DocumentProcessor,
            context_manager: ContextManager,
            config: Optional[ContextConfig] = None
    ):
        """Initialize context integration.

        Args:
            event_bus: Event bus instance
            conversation_manager: Conversation manager
            document_processor: Document processor
            context_manager: Context manager
            config: Optional configuration
        """
        self.event_bus = event_bus
        self.conversation = conversation_manager
        self.doc_processor = document_processor
        self.context = context_manager
        self.config = config or ContextConfig()

        # Set up event handlers
        self._setup_event_handlers()

        # Integration state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

    def _setup_event_handlers(self) -> None:
        """Set up event handlers."""
        self.event_bus.subscribe(
            EventType.DOCUMENT_PROCESSED,
            self._handle_document_event
        )
        self.event_bus.subscribe(
            EventType.TRANSCRIPT,
            self._handle_transcript_event
        )
        self.event_bus.subscribe(
            EventType.CONTEXT_UPDATE,
            self._handle_context_event
        )
        self.event_bus.subscribe(
            EventType.MESSAGE_SENT,
            self._handle_message_event
        )
        self.event_bus.subscribe(
            EventType.RESPONSE_RECEIVED,
            self._handle_response_event
        )

    async def start(self) -> None:
        """Start context integration."""
        if not self._running:
            self._running = True
            self._update_task = asyncio.create_task(
                self._update_loop()
            )
            await self.event_bus.publish(Event(
                type=EventType.CONTEXT_UPDATE,
                data={
                    "status": "integration_started",
                    "config": self.config.dict()
                }
            ))

    async def stop(self) -> None:
        """Stop context integration."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None

        await self.event_bus.publish(Event(
            type=EventType.CONTEXT_UPDATE,
            data={"status": "integration_stopped"}
        ))

    async def create_session(
            self,
            session_id: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create new integration session.

        Args:
            session_id: Session identifier
            metadata: Optional session metadata
        """
        self.active_sessions[session_id] = {
            "created_at": datetime.now(),
            "metadata": metadata or {},
            "updates_pending": False,
            "last_update": None
        }

        await self.event_bus.publish(Event(
            type=EventType.CONTEXT_UPDATE,
            data={
                "status": "session_created",
                "session_id": session_id
            }
        ))

    async def close_session(
            self,
            session_id: str
    ) -> None:
        """Close integration session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

            await self.event_bus.publish(Event(
                type=EventType.CONTEXT_UPDATE,
                data={
                    "status": "session_closed",
                    "session_id": session_id
                }
            ))

    async def process_document(
            self,
            session_id: str,
            content: bytes,
            doc_type: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process document and update context.

        Args:
            session_id: Session identifier
            content: Document content
            doc_type: Document type
            metadata: Optional document metadata

        Yields:
            Processing results

        Raises:
            ContextError: If processing fails
        """
        try:
            # Process document
            async for result in self.doc_processor.process_document(
                    content=content,
                    doc_type=doc_type
            ):
                # Create context entry
                entry = ContextEntry(
                    content=result.analysis,
                    source=ContextSource.DOCUMENT,
                    metadata=ContextMetadata(
                        source=ContextSource.DOCUMENT,
                        priority=ContextPriority.HIGH,
                        state=ContextState.ACTIVE,
                        custom_data=metadata or {}
                    )
                )

                # Update context
                await self.context.update_context(
                    session_id,
                    entry.content,
                    entry.metadata.source,
                    entry.metadata.priority
                )

                yield result.model_dump()

        except Exception as e:
            raise ContextError(
                f"Document processing failed: {str(e)}",
                details={
                    "session_id": session_id,
                    "doc_type": doc_type
                }
            )

    async def _update_loop(self) -> None:
        """Main update loop."""
        while self._running:
            try:
                # Process pending updates for each session
                for session_id, session in self.active_sessions.items():
                    if session["updates_pending"]:
                        await self._process_session_updates(session_id)
                        session["updates_pending"] = False
                        session["last_update"] = datetime.now()

                await asyncio.sleep(self.config.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Update loop error: {e}")
                await asyncio.sleep(1.0)

    async def _process_session_updates(
            self,
            session_id: str
    ) -> None:
        """Process pending updates for session.

        Args:
            session_id: Session identifier
        """
        # Get current context
        context = await self.context.get_context(session_id)

        # Get conversation history
        history = self.conversation.get_conversation_history()

        # Create context summary
        summary = await self._create_context_summary(
            context,
            history
        )

        # Update conversation context
        await self.conversation.update_context({
            "context_summary": summary
        })

        # Emit update event
        await self.event_bus.publish(Event(
            type=EventType.CONTEXT_UPDATE,
            data={
                "session_id": session_id,
                "status": "updates_processed",
                "summary": summary
            }
        ))

    async def _create_context_summary(
            self,
            context: Dict[str, ContextEntry],
            history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create context summary.

        Args:
            context: Current context
            history: Conversation history

        Returns:
            Context summary
        """
        return {
            "context_entries": len(context),
            "conversation_turns": len(history),
            "active_sources": list({
                entry.metadata.source
                for entry in context.values()
            }),
            "last_update": datetime.now().isoformat()
        }

    async def _handle_document_event(
            self,
            event: Event
    ) -> None:
        """Handle document processing events.

        Args:
            event: Document event
        """
        if event.data.get("status") == "processed":
            session_id = event.data.get("session_id")
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["updates_pending"] = True

    async def _handle_transcript_event(
            self,
            event: Event
    ) -> None:
        """Handle transcription events.

        Args:
            event: Transcript event
        """
        if "text" in event.data:
            session_id = event.data.get("session_id")
            if session_id in self.active_sessions:
                # Create context entry
                entry = ContextEntry(
                    content={"transcript": event.data["text"]},
                    source=ContextSource.TRANSCRIPT,
                    metadata=ContextMetadata(
                        source=ContextSource.TRANSCRIPT,
                        priority=ContextPriority.MEDIUM,
                        state=ContextState.ACTIVE
                    )
                )

                # Update context
                await self.context.update_context(
                    session_id,
                    entry.content,
                    entry.metadata.source,
                    entry.metadata.priority
                )

    async def _handle_context_event(
            self,
            event: Event
    ) -> None:
        """Handle context update events.

        Args:
            event: Context event
        """
        session_id = event.data.get("session_id")
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["updates_pending"] = True

    async def _handle_message_event(
            self,
            event: Event
    ) -> None:
        """Handle message events.

        Args:
            event: Message event
        """
        session_id = event.data.get("session_id")
        if session_id in self.active_sessions:
            # Create context entry
            entry = ContextEntry(
                content={"message": event.data.get("text", "")},
                source=ContextSource.CONVERSATION,
                metadata=ContextMetadata(
                    source=ContextSource.CONVERSATION,
                    priority=ContextPriority.MEDIUM,
                    state=ContextState.ACTIVE
                )
            )

            # Update context
            await self.context.update_context(
                session_id,
                entry.content,
                entry.metadata.source,
                entry.metadata.priority
            )

    async def _handle_response_event(
            self,
            event: Event
    ) -> None:
        """Handle response events.

        Args:
            event: Response event
        """
        session_id = event.data.get("session_id")
        if session_id in self.active_sessions:
            # Create context entry
            entry = ContextEntry(
                content={"response": event.data.get("text", "")},
                source=ContextSource.SYSTEM,
                metadata=ContextMetadata(
                    source=ContextSource.SYSTEM,
                    priority=ContextPriority.MEDIUM,
                    state=ContextState.ACTIVE
                )
            )

            # Update context
            await self.context.update_context(
                session_id,
                entry.content,
                entry.metadata.source,
                entry.metadata.priority
            )