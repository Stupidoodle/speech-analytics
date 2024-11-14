"""Conversation context management."""
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from src.context.manager import ContextManager
from src.events.bus import EventBus
from src.events.types import Event, EventType

from .types import (
    Role,
    MessageRole,
    Message,
    MessageContent,
    MessageType,
    SessionConfig,
    ToolConfig
)
from .roles import RoleManager
from .exceptions import ConversationError


class ConversationContext:
    """Manages conversation context and state."""

    def __init__(
        self,
        event_bus: EventBus,
        context_manager: ContextManager,
        role_manager: RoleManager,
        config: Optional[SessionConfig] = None
    ):
        """Initialize conversation context.

        Args:
            event_bus: Event bus instance
            context_manager: Context manager instance
            role_manager: Role manager instance
            config: Optional session configuration
        """
        self.event_bus = event_bus
        self.context = context_manager
        self.roles = role_manager
        self.config = config or SessionConfig()

        # Conversation state
        self.turns: List[Message] = []
        self.tool_states: Dict[str, Dict[str, Any]] = {}
        self.active_tools: Set[str] = set()
        self.context_references: Dict[str, Set[str]] = {}

    async def add_message(
        self,
        message: Message
    ) -> None:
        """Add message to conversation.

        Args:
            message: Message to add
        """
        # Process message content
        for content in message.content:
            # Handle tool use
            if content.type == MessageType.TOOL_USE:
                await self._handle_tool_use(content)
            # Handle tool results
            elif content.type == MessageType.TOOL_RESULT:
                await self._handle_tool_result(content)
            # Handle context references
            elif content.type == MessageType.CONTEXT:
                await self._handle_context_reference(content)

        # Add to turns
        self.turns.append(message)

        # Emit message event
        await self.event_bus.publish(Event(
            type=EventType.CONVERSATION,
            data={
                "action": "message_added",
                "role": message.role,
                "timestamp": message.timestamp
            }
        ))

    async def get_context(
        self,
        include_tools: bool = True
    ) -> Dict[str, Any]:
        """Get current conversation context.

        Args:
            include_tools: Whether to include tool state

        Returns:
            Current context
        """
        context = {
            "turns": len(self.turns),
            "last_speaker": self.turns[-1].role if self.turns else None,
            "timestamp": datetime.now(),
            "references": dict(self.context_references)
        }

        if include_tools:
            context["tools"] = {
                "active": list(self.active_tools),
                "states": self.tool_states
            }

        return context

    async def _handle_tool_use(
        self,
        content: MessageContent
    ) -> None:
        """Handle tool use in message.

        Args:
            content: Tool use content

        Raises:
            ConversationError: If tool use is invalid
        """
        if not content.tool_use:
            return

        tool_name = content.tool_use.get("name")
        tool_id = content.tool_use.get("tool_use_id")

        # Validate tool
        tool_config = next(
            (t for t in self.roles.get_tools(self.config.role)
             if t.name == tool_name),
            None
        )
        if not tool_config:
            raise ConversationError(
                f"Tool not allowed: {tool_name}"
            )

        # Track tool state
        self.active_tools.add(tool_name)
        self.tool_states[tool_id] = {
            "name": tool_name,
            "status": "active",
            "start_time": datetime.now(),
            "input": content.tool_use.get("input", {})
        }

        # Emit tool event
        await self.event_bus.publish(Event(
            type=EventType.TOOL_USE,
            data={
                "tool": tool_name,
                "tool_id": tool_id,
                "status": "started"
            }
        ))

    async def _handle_tool_result(
        self,
        content: MessageContent
    ) -> None:
        """Handle tool result in message.

        Args:
            content: Tool result content
        """
        if not content.tool_result:
            return

        tool_id = content.tool_result.get("tool_use_id")
        if tool_id in self.tool_states:
            # Update tool state
            self.tool_states[tool_id].update({
                "status": "completed",
                "end_time": datetime.now(),
                "result": content.tool_result.get("content", {})
            })

            # Remove from active tools
            tool_name = self.tool_states[tool_id]["name"]
            self.active_tools.discard(tool_name)

            # Emit tool event
            await self.event_bus.publish(Event(
                type=EventType.TOOL_USE,
                data={
                    "tool": tool_name,
                    "tool_id": tool_id,
                    "status": "completed"
                }
            ))

    async def _handle_context_reference(
        self,
        content: MessageContent
    ) -> None:
        """Handle context reference in message.

        Args:
            content: Context content
        """
        if not content.context_data:
            return

        # Track references
        refs = content.context_data.get("references", [])
        for ref in refs:
            if ref not in self.context_references:
                self.context_references[ref] = set()
            self.context_references[ref].add(
                content.context_data.get("source", "unknown")
            )

    def get_tool_state(
        self,
        tool_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get state of specific tool.

        Args:
            tool_id: Tool identifier

        Returns:
            Tool state if found
        """
        return self.tool_states.get(tool_id)

    def get_active_tools(self) -> List[str]:
        """Get currently active tools.

        Returns:
            List of active tool names
        """
        return list(self.active_tools)

    async def get_tool_metrics(self) -> Dict[str, Any]:
        """Get tool usage metrics.

        Returns:
            Tool metrics
        """
        metrics = {
            "total_uses": len(self.tool_states),
            "by_tool": {},
            "average_duration": 0.0,
            "success_rate": 0.0
        }

        if not self.tool_states:
            return metrics

        # Calculate tool-specific metrics
        tool_counts = {}
        total_duration = 0.0
        successful = 0

        for state in self.tool_states.values():
            tool_name = state["name"]
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

            if state["status"] == "completed":
                successful += 1
                if "start_time" in state and "end_time" in state:
                    duration = (
                        state["end_time"] -
                        state["start_time"]
                    ).total_seconds()
                    total_duration += duration

        metrics.update({
            "by_tool": tool_counts,
            "average_duration": (
                total_duration / successful if successful else 0.0
            ),
            "success_rate": successful / len(self.tool_states)
        })

        return metrics

    async def cleanup_tools(self) -> None:
        """Clean up tool states."""
        # Clear completed tools
        completed_tools = [
            tool_id
            for tool_id, state in self.tool_states.items()
            if state["status"] == "completed"
        ]
        for tool_id in completed_tools:
            del self.tool_states[tool_id]

        # Clear active tools
        self.active_tools.clear()

    def get_last_turn(
        self,
        role: Optional[MessageRole] = None
    ) -> Optional[Message]:
        """Get last conversation turn.

        Args:
            role: Optional role filter

        Returns:
            Last message if found
        """
        if not self.turns:
            return None

        if role:
            for turn in reversed(self.turns):
                if turn.role == role:
                    return turn
            return None

        return self.turns[-1]

    def get_turn_count(
        self,
        role: Optional[MessageRole] = None
    ) -> int:
        """Get number of conversation turns.

        Args:
            role: Optional role filter

        Returns:
            Turn count
        """
        if role:
            return sum(1 for turn in self.turns if turn.role == role)
        return len(self.turns)

    async def get_summary(self) -> Dict[str, Any]:
        """Get conversation summary.

        Returns:
            Conversation summary
        """
        return {
            "turns": self.get_turn_count(),
            "by_role": {
                role: self.get_turn_count(role)
                for role in MessageRole
            },
            "tools": await self.get_tool_metrics(),
            "context_usage": {
                ref: len(sources)
                for ref, sources in self.context_references.items()
            },
            "duration": (
                self.turns[-1].timestamp - self.turns[0].timestamp
                if len(self.turns) > 1 else 0
            ).total_seconds()
        }