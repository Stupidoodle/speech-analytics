"""Conversation management with Bedrock integration."""
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime
import asyncio
import uuid

from src.events.bus import EventBus
from src.events.types import Event, EventType
from src.context.manager import ContextManager
from bedrock import BedrockClient

from .types import (
    Role,
    MessageRole,
    Message,
    MessageContent,
    MessageType,
    SessionConfig,
    SessionState,
    BedrockConfig,
    StreamResponse,
    SystemPrompt
)
from .context import ConversationContext
from .roles import RoleManager
from .exceptions import (
    ConversationError,
    SessionError,
    MessageError,
    StreamError
)


class ConversationManager:
    """Manages conversations and integrates all components."""

    def __init__(
        self,
        event_bus: EventBus,
        bedrock_client: BedrockClient,
        context_manager: ContextManager,
        config: Optional[BedrockConfig] = None
    ):
        """Initialize conversation manager.

        Args:
            event_bus: Event bus instance
            bedrock_client: Bedrock API client
            context_manager: Context manager
            config: Optional Bedrock configuration
        """
        self.event_bus = event_bus
        self.bedrock = bedrock_client
        self.context_manager = context_manager
        self.config = config or BedrockConfig()

        # Component initialization
        self.role_manager = RoleManager()

        # Session management
        self.active_sessions: Dict[str, ConversationContext] = {}
        self.session_configs: Dict[str, SessionConfig] = {}
        self.session_status: Dict[str, Dict[str, Any]] = {}

    async def create_session(
        self,
        role: Role,
        config: Optional[SessionConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new conversation session.

        Args:
            role: Session role
            config: Optional session configuration
            metadata: Optional session metadata

        Returns:
            Session identifier

        Raises:
            ConversationError: If session creation fails
        """
        try:
            # Generate session ID
            session_id = str(uuid.uuid4())

            # Create session config
            session_config = config or SessionConfig(role=role)
            self.session_configs[session_id] = session_config

            # Initialize conversation context
            context = ConversationContext(
                event_bus=self.event_bus,
                context_manager=self.context_manager,
                role_manager=self.role_manager,
                config=session_config
            )
            self.active_sessions[session_id] = context

            # Initialize session status
            self.session_status[session_id] = {
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "metadata": metadata or {},
                "metrics": {
                    "messages": 0,
                    "tools_used": 0,
                    "errors": 0
                }
            }

            # Set up role-specific system prompts
            await self._setup_role_prompts(session_id, role)

            # Emit session created event
            await self.event_bus.publish(Event(
                type=EventType.CONVERSATION,
                data={
                    "action": "session_created",
                    "session_id": session_id,
                    "role": role.value
                }
            ))

            return session_id

        except Exception as e:
            raise ConversationError(
                f"Failed to create session: {str(e)}"
            )

    async def send_message(
        self,
        session_id: str,
        content: str,
        role: Optional[MessageRole] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[StreamResponse]:
        """Send message and get streaming response.

        Args:
            session_id: Session identifier
            content: Message content
            role: Optional message role
            metadata: Optional message metadata

        Yields:
            Streaming responses

        Raises:
            SessionError: If session not found
            MessageError: If message processing fails
        """
        try:
            # Validate session
            session = self._get_session(session_id)
            session_config = self.session_configs[session_id]

            # Create message
            message = Message(
                role=role or MessageRole.USER,
                content=[MessageContent(
                    type=MessageType.TEXT,
                    text=content,
                    metadata=metadata or {}
                )],
                timestamp=datetime.now()
            )

            # Add to conversation context
            await session.add_message(message)
            self._update_session_metrics(session_id, "messages")

            # Get conversation context
            context_data = await session.get_context()
            system_prompts = await self._get_system_prompts(
                session_id,
                context_data
            )

            # Prepare messages for Bedrock
            messages = self._prepare_messages(session, message)

            try:
                # Get streaming response
                async for response in self.bedrock.generate_stream(
                    messages,
                    system_prompts
                ):
                    # Handle assistant message
                    if response.content and response.content.text:
                        await self._handle_assistant_message(
                            session_id,
                            response
                        )

                    # Handle tool use
                    if (response.content and
                            response.content.tool_use):
                        await self._handle_tool_use(
                            session_id,
                            response
                        )
                        self._update_session_metrics(
                            session_id,
                            "tools_used"
                        )

                    yield response

            except StreamError as e:
                raise MessageError(
                    f"Stream error: {str(e)}",
                    session_id,
                    message.role
                )

            # Update session status
            self.session_status[session_id]["last_activity"] = datetime.now()

        except KeyError:
            raise SessionError(
                f"Session not found: {session_id}",
                session_id,
                SessionState.ERROR
            )
        except Exception as e:
            self._update_session_metrics(session_id, "errors")
            if isinstance(e, (SessionError, MessageError)):
                raise
            raise ConversationError(f"Message processing failed: {str(e)}")

    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """Get conversation history.

        Args:
            session_id: Session identifier
            limit: Optional message limit
            include_metadata: Whether to include metadata

        Returns:
            Conversation history

        Raises:
            SessionError: If session not found
        """
        session = self._get_session(session_id)

        history = []
        messages = session.turns[-limit:] if limit else session.turns

        for msg in messages:
            msg_data = {
                "role": msg.role,
                "content": [
                    {
                        "type": content.type,
                        "text": content.text
                    }
                    for content in msg.content
                ],
                "timestamp": msg.timestamp.isoformat()
            }
            if include_metadata:
                msg_data["metadata"] = msg.metadata
            history.append(msg_data)

        return history

    async def end_session(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """End conversation session.

        Args:
            session_id: Session identifier

        Returns:
            Session summary

        Raises:
            SessionError: If session not found
        """
        try:
            # Get session
            session = self._get_session(session_id)

            # Get session summary
            summary = await session.get_summary()
            summary.update({
                "duration": (
                    datetime.now() -
                    self.session_status[session_id]["created_at"]
                ).total_seconds(),
                "metrics": self.session_status[session_id]["metrics"]
            })

            # Cleanup session
            await session.cleanup_tools()
            del self.active_sessions[session_id]
            del self.session_configs[session_id]
            del self.session_status[session_id]

            # Emit session ended event
            await self.event_bus.publish(Event(
                type=EventType.CONVERSATION,
                data={
                    "action": "session_ended",
                    "session_id": session_id,
                    "summary": summary
                }
            ))

            return summary

        except KeyError:
            raise SessionError(
                f"Session not found: {session_id}",
                session_id,
                SessionState.ERROR
            )

    def _get_session(
        self,
        session_id: str
    ) -> ConversationContext:
        """Get session context.

        Args:
            session_id: Session identifier

        Returns:
            Session context

        Raises:
            SessionError: If session not found
        """
        if session_id not in self.active_sessions:
            raise SessionError(
                f"Session not found: {session_id}",
                session_id,
                SessionState.ERROR
            )
        return self.active_sessions[session_id]

    async def _setup_role_prompts(
        self,
        session_id: str,
        role: Role
    ) -> None:
        """Set up role-specific prompts.

        Args:
            session_id: Session identifier
            role: Session role
        """
        session = self._get_session(session_id)

        # Get role prompts
        # NOTE: We might need to make this async in the future
        prompts = self.role_manager.get_system_prompts(role)

        # Add to conversation context
        for prompt in prompts:
            await self._add_system_prompt(
                session_id,
                prompt
            )

    async def _add_system_prompt(
        self,
        session_id: str,
        prompt: SystemPrompt
    ) -> None:
        """Add system prompt to session.

        Args:
            session_id: Session identifier
            prompt: System prompt
        """
        message = Message(
            role=MessageRole.SYSTEM,
            content=[MessageContent(
                type=MessageType.SYSTEM,
                text=prompt.text,
                metadata=prompt.metadata
            )],
            timestamp=datetime.now()
        )

        session = self._get_session(session_id)
        await session.add_message(message)

    def _prepare_messages(
        self,
        session: ConversationContext,
        message: Message
    ) -> List[Dict[str, Any]]:
        """Prepare messages for Bedrock.

        Args:
            session: Session context
            message: New message

        Returns:
            Prepared messages
        """
        messages = []

        # Add context messages
        for turn in session.turns:
            messages.append({
                "role": turn.role.value,
                "content": [
                    {
                        "type": content.type.value,
                        "text": content.text,
                        **({"tool_use": content.tool_use}
                           if content.tool_use else {}),
                        **({"tool_result": content.tool_result}
                           if content.tool_result else {})
                    }
                    for content in turn.content
                ]
            })

        return messages

    async def _get_system_prompts(
        self,
        session_id: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Get system prompts for session.

        Args:
            session_id: Session identifier
            context: Current context

        Returns:
            Combined system prompts
        """
        session = self._get_session(session_id)
        session_config = self.session_configs[session_id]

        # Get role-specific prompts
        # NOTE: We might need to make this async in the future
        if session_config.role:
            prompts = self.role_manager.get_system_prompts(
                session_config.role,
                context
            )
            if prompts:
                return "\n\n".join(p.text for p in prompts)

        return None

    async def _handle_assistant_message(
        self,
        session_id: str,
        response: StreamResponse
    ) -> None:
        """Handle assistant message.

        Args:
            session_id: Session identifier
            response: Response content
        """
        if response.content and response.content.text:
            message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(
                    type=MessageType.TEXT,
                    text=response.content.text
                )],
                timestamp=datetime.now()
            )

            session = self._get_session(session_id)
            await session.add_message(message)

    async def _handle_tool_use(
        self,
        session_id: str,
        response: StreamResponse
    ) -> None:
        """Handle tool use in response.

        Args:
            session_id: Session identifier
            response: Response containing tool use
        """
        if response.content and response.content.tool_use:
            message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(
                    type=MessageType.TOOL_USE,
                    tool_use=response.content.tool_use
                )],
                timestamp=datetime.now()
            )

            session = self._get_session(session_id)
            await session.add_message(message)

    def _update_session_metrics(
        self,
        session_id: str,
        metric: str
    ) -> None:
        """Update session metrics.

        Args:
            session_id: Session identifier
            metric: Metric to update
        """
        if session_id in self.session_status:
            self.session_status[session_id]["metrics"][metric] += 1