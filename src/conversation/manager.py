"""Conversation management with Bedrock integration."""
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime

from .bedrock import BedrockClient
from .types import (
    BedrockConfig,
    MessageRole,
    Message,
    StreamResponse,
    DocumentContent
)
from .exceptions import (
    ConversationError,
    ServiceError,
    ValidationError
)
from src.context.conversation_integration import ContextConversationIntegration


class ConversationManager:
    """Manages conversation flow and Bedrock integration."""

    def __init__(
            self,
            eve
            region: str,
            config: Optional[BedrockConfig] = None
    ) -> None:
        """Initialize conversation manager.

        Args:
            region: AWS region
            config: Optional Bedrock configuration
        """
        self.config = config or BedrockConfig()
        self.bedrock = BedrockClient(region, self.config)
        self.messages: List[Message] = []
        self.system_messages: List[Dict[str, Any]] = []
        self.current_context: Dict[str, Any] = {}

    async def __aenter__(self):
        """Enter async context."""
        await self.bedrock.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        await self.bedrock.__aexit__(exc_type, exc_val, exc_tb)

    async def add_system_prompt(
            self,
            prompt: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a system prompt to the conversation.

        Args:
            prompt: System prompt text
            metadata: Optional metadata
        """
        self.system_messages.append({
            "text": prompt,
            "metadata": metadata or {"type": "system_prompt"}
        })

    async def add_document(
            self,
            content: bytes,
            name: str,
            format: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a document to the conversation context.

        Args:
            content: Document content
            name: Document name
            format: Document format
            metadata: Optional metadata

        Raises:
            ValidationError: On invalid document format
        """
        try:
            doc = DocumentContent(
                format=format,
                name=name,
                source=content
            )
            self.system_messages.append({
                "text": f"Context from document: {name}",
                "document": doc.model_dump(),
                "metadata": metadata or {"type": "document"}
            })
        except Exception as e:
            raise ValidationError(f"Invalid document: {str(e)}")

    def _prepare_messages(
            self,
            text: str,
            files: Optional[List[Dict[str, Any]]] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Prepare messages for Bedrock API.

        Args:
            text: Message text
            files: Optional list of files
            metadata: Optional metadata

        Returns:
            Formatted messages
        """
        # Add user message
        content = [text]
        if files:
            content.extend(files)

        msg_metadata = metadata or {}
        if self.current_context:
            msg_metadata["context"] = self.current_context

        self.messages.append(Message(
            role=MessageRole.USER,
            content=content,
            metadata=msg_metadata,
            timestamp=datetime.now()
        ))

        # Format messages for API
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                "metadata": msg.metadata,
            }
            for msg in self.messages
        ]

    def update_context(
            self,
            context_updates: Dict[str, Any]
    ) -> None:
        """Update conversation context.

        Args:
            context_updates: Context updates
        """
        self.current_context.update(context_updates)

    async def send_message(
            self,
            text: str,
            files: Optional[List[Dict[str, Any]]] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[StreamResponse]:
        """Send a message and get streaming response.

        Args:
            text: Message text
            files: Optional list of files
            metadata: Optional metadata

        Yields:
            Streaming responses

        Raises:
            ConversationError: On conversation errors
        """
        try:
            messages = self._prepare_messages(text, files, metadata)

            # Get system prompt if any
            system_prompt = None
            if self.system_messages:
                system_prompt = "\n".join(
                    msg["text"] for msg in self.system_messages
                )

            # Get streaming response
            response_text = ""
            async for response in self.bedrock.generate_stream(
                    messages,
                    system_prompt
            ):
                # Collect response text
                if response.content and response.content.text:
                    response_text += response.content.text

                # Collect metadata
                if response.metadata:
                    metadata["usage"] = response.metadata.get("usage", {})
                    metadata["metrics"] = response.metadata.get("metrics", {})
                    metadata["trace"] = response.metadata.get("trace", {})

                yield response

            # Add assistant's response to history
            self.messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=[response_text],
                timestamp=datetime.now(),
                metadata=metadata
            ))

        except ServiceError as e:
            raise ConversationError(
                f"Service error: {str(e)}",
                details={
                    "service": e.service,
                    "error_code": e.error_code
                }
            )
        except Exception as e:
            raise ConversationError(f"Failed to send message: {str(e)}")

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get formatted conversation history.

        Returns:
            List of conversation messages
        """
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in self.messages
        ]

    async def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self.system_messages.clear()

    async def export_conversation(self) -> Dict[str, Any]:
        """Export conversation history and metadata.

        Returns:
            Conversation export data
        """
        export_data = {
            "conversation": self.get_conversation_history(),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_id": self.config.model_id,
            }
        }

        if self.system_messages:
            export_data["system_context"] = self.system_messages

        return export_data