# src/context/conversation_integration.py
from typing import Dict, Any, Optional
from src.conversation.manager import ConversationManager
from src.context.context_manager import ContextManager, ContextUpdate
from src.events.bus import EventBus
from src.events.types import Event, EventType

class ContextConversationIntegration:
    def __init__(
        self,
        conversation_manager: ConversationManager,
        context_manager: ContextManager,
        event_bus: EventBus
    ):
        self.conversation_manager = conversation_manager
        self.context_manager = context_manager
        self.event_bus = event_bus

        self.event_bus.subscribe(EventType.CONTEXT_UPDATE, self.handle_context_update)

    async def handle_context_update(self, event: Event):
        """Handle context update events and propagate to the conversation state."""
        context_id = event.data.get("context_id")
        if context_id:
            updates = event.data.get("updates", {})
            if updates:
                conversation = self.conversation_manager.get_conversation(context_id)
                if conversation:
                    # Update the conversation state with the context updates
                    conversation.update_context(updates)

    async def get_conversation_context(self, conversation_id: str) -> Dict[str, Any]:
        """Retrieve the context for a given conversation."""
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if conversation:
            context_id = conversation.context_id
            context_entries = await self.context_manager.get_context(context_id)
            return {entry_key: entry.content for entry_key, entry in context_entries.items()}
        return {}

    async def update_conversation_context(
        self,
        conversation_id: str,
        updates: Dict[str, Any],
        source: str,
        priority: float = 1.0
    ):
        """Update the context for a given conversation."""
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if conversation:
            context_id = conversation.context_id
            await self.context_manager.update_context(
                context_id,
                updates=updates,
                source=source,
                priority=priority
            )

    async def handle_context_query(self, conversation_id: str, query: str) -> Optional[Dict[str, Any]]:
        """Handle a context query for a given conversation."""
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if conversation:
            context_id = conversation.context_id
            search_results = await self.context_manager.search(context_id, query)
            return {entry.source: entry.content for entry in search_results}
        return None