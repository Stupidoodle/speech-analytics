from typing import Dict, Any, Optional, List, AsyncIterator, Union
import asyncio
import aioboto3 as boto3
from datetime import datetime
from pathlib import Path
import aiofiles
import json

from .types import (
    Role, Message, MessageContent,
    DocumentContent, StreamResponse, InferenceConfig
)
from .exceptions import ConversationError


class ConversationManager:
    """Manages real-time conversation with context awareness."""

    def __init__(
        self,
        region: str,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        client=None,
    ):
        """Initialize conversation manager."""
        self.region = region
        self.session = None
        self.client = None
        self.model_id = model_id
        self.inference_config = InferenceConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        self.messages: List[Message] = []
        self.system_messages: List[Dict[str, Any]] = []

    async def __aenter__(self):
        if not self.client:
            self.session = boto3.Session()
            self.client = await self.session.client(
                "bedrock-runtime",
                region_name=self.region
            ).__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
            self.client = None
        if self.session:
            self.session = None

    async def add_document(
        self,
        content: Union[str, bytes, Path],
        name: str,
        format: str
    ) -> None:
        """Add a document to the conversation context."""
        try:
            if isinstance(content, (str, Path)):
                path = Path(content)
                async with aiofiles.open(path, 'rb') as f:
                    doc_bytes = await f.read()
            else:
                doc_bytes = content

            doc_content = MessageContent(
                document=DocumentContent(
                    format=format,
                    name=name,
                    source=doc_bytes
                )
            )

            self.system_messages.append({
                'text': f"Context from document: {name}",
                'document': doc_content.document.__dict__
            })

        except Exception as e:
            raise ConversationError(f"Failed to add document: {e}")

    async def add_system_prompt(self, prompt: str) -> None:
        """Add a system prompt."""
        self.system_messages.append({'text': prompt})

    async def send_message(
        self,
        text: str,
        files: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncIterator[StreamResponse]:
        """Send a message and get streaming response."""
        try:
            content = [{'text': text}]
            if files:
                content.extend(files)

            # Add message to history
            self.messages.append(Message(
                role=Role.USER,
                content=content,
            ))

            # Make API call
            response = await self.client.converse_stream(
                modelId=self.model_id,
                messages=[{
                    "role": msg.role.value,
                    "content": msg.content
                } for msg in self.messages],
                system=self.system_messages,
                inferenceConfig=self.inference_config
            )

            current_text = ""
            async for chunk in response["stream"]:
                # Check for error first
                if "modelStreamErrorException" in chunk:
                    error = chunk["modelStreamErrorException"]
                    raise ConversationError(
                        error["message"],
                        details={
                            "status_code": error.get("originalStatusCode"),
                            "original_message": error.get("originalMessage")
                        }
                    )

                if "contentBlockDelta" in chunk:
                    delta = chunk["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        current_text += delta["text"]
                        yield StreamResponse(text=delta["text"])
                elif "messageStop" in chunk:
                    yield StreamResponse(
                        stop_reason=chunk["messageStop"]["stopReason"]
                    )
                elif "metadata" in chunk:
                    yield StreamResponse(metadata=chunk["metadata"])

            # Add assistant's response to history
            self.messages.append(Message(
                role=Role.ASSISTANT,
                content=[{'text': current_text}],
                timestamp=datetime.now()
            ))

        except ConversationError:
            raise
        except Exception as e:
            raise ConversationError(f"Failed to send message: {e}")

    async def _process_stream(
        self,
        stream: AsyncIterator[Dict[str, Any]]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process the response stream."""
        async for chunk in stream:
            if 'contentBlockDelta' in chunk:
                delta = chunk['contentBlockDelta']['delta']
                if 'text' in delta:
                    yield {'text': delta['text']}
                if 'toolUse' in delta:
                    yield {'tool_use': delta['toolUse']}

            elif 'messageStop' in chunk:
                yield {
                    'stop_reason': chunk['messageStop']['stopReason']
                }

            elif 'metadata' in chunk:
                yield {'metadata': chunk['metadata']}

            await asyncio.sleep(0)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get formatted conversation history."""
        return [
            {
                'role': msg.role.value,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat()
                if msg.timestamp else None
            }
            for msg in self.messages
        ]

    async def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self.system_messages.clear()

    async def export_conversation(
        self,
        format_type: str = 'json'
    ) -> Dict[str, Any]:
        """Export conversation history and analysis in specified format."""
        export_data = {
            'conversation': self.get_conversation_history(),
            'analysis': self._get_latest_analysis(),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'format_version': '1.0',
            }
        }

        if self.system_messages:
            export_data['system_context'] = [
                msg.content for msg in self.system_messages
            ]

        return export_data

    def _get_latest_analysis(self) -> Optional[Dict[str, Any]]:
        """Get the most recent conversation analysis if available."""
        if not hasattr(self, 'latest_analysis'):
            return None

        analysis = self.latest_analysis
        return {
            'key_points': analysis.key_points,
            'action_items': [
                {
                    'description': item.description,
                    'assignee': item.assignee,
                    'deadline': item.deadline.isoformat()
                    if item.deadline else None,
                    'status': item.status,
                    'priority': item.priority
                } for item in analysis.action_items
            ],
            'questions': analysis.questions,
            'follow_up_topics': analysis.follow_up_topics,
            'context_specific': analysis.context_specific,
            'timestamp': analysis.timestamp.isoformat(),
            'validated': analysis.validated,
            'validation_notes': analysis.validation_notes
        }

    async def handle_context_update(
        self,
        context_type: str,
        context_data: Dict[str, Any]
    ) -> None:
        """Handle real-time context updates."""
        # Update internal context
        if not hasattr(self, 'context'):
            self.context = {}
        self.context[context_type] = context_data

        # Create system message for context update
        context_msg = self._create_context_message(context_type, context_data)
        self.system_messages.append(context_msg)

        # Trigger reanalysis if needed
        if self.latest_analysis and self._should_reanalyze():
            await self._update_analysis()

    def _create_context_message(
        self,
        context_type: str,
        context_data: Dict[str, Any]
    ) -> Message:
        """Create system message for context update."""
        content = (
            f"Updated {context_type} context:\n"
            f"{json.dumps(context_data, indent=2)}"
        )
        return Message(
            role=Role.SYSTEM,
            content=[content],
            timestamp=datetime.now()
        )

    def _should_reanalyze(self) -> bool:
        """Determine if reanalysis is needed based on context updates."""
        if not hasattr(self, 'last_analysis_time'):
            return True

        time_diff = (datetime.now() - self.last_analysis_time).total_seconds()
        return time_diff >= 60  # Reanalyze after 1 minute

    async def _update_analysis(self) -> None:
        """Update analysis based on new context."""
        if not self.messages:
            return

        # Get recent conversation context
        recent_messages = self.messages[-10:]  # Last 10 messages
        context = '\n'.join([msg.content[0] for msg in recent_messages])

        # Create analysis prompt
        prompt = self._create_analysis_prompt(context)

        # Get new analysis
        responses = []
        async for response in self.send_message(prompt):
            if response.text:
                responses.append(response.text)

        analysis_text = ''.join(responses)
        self.latest_analysis = self._parse_analysis_response(analysis_text)
        self.last_analysis_time = datetime.now()
