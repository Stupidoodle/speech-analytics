from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path
import aiofiles
from enum import Enum

from .manager import ConversationManager

from .exceptions import (
    StreamError,
    ValidationError,
    ServiceError,
    DocumentError
)


class ConversationState(Enum):
    INITIAL = "initial"
    ACTIVE = "active"
    WAITING_RESPONSE = "waiting_response"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    CONCLUDING = "concluding"


@dataclass
class ConversationTurn:
    speaker: str
    content: str
    timestamp: datetime
    state: ConversationState
    context: Dict[str, Any]
    analysis: Optional[Dict[str, Any]] = None


class ConversationContext:
    """Manages context data for conversations."""

    def __init__(self, conversation_manager: ConversationManager):
        """Initialize conversation context."""
        self.ai = conversation_manager
        self.files: Dict[str, Dict[str, Any]] = {}
        self.structured_data: Dict[str, List[Dict[str, Any]]] = {}
        self.metadata: Dict[str, Any] = {}
        self.turns: List[ConversationTurn] = []
        self.current_state = ConversationState.INITIAL
        self.context_history: List[Dict[str, Any]] = []
        self.active_topics: List[str] = []
        self._file_handlers = {
            'application/pdf': self._handle_pdf,
            'text/plain': self._handle_text,
            'application/json': self._handle_json,
            'text/csv': self._handle_csv,
            'application/vnd.openxmlformats-officedocument.wordprocessingml'
            '.document': self._handle_docx
        }

    async def add_file(
        self,
        file_path: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a file to the conversation context.

        Args:
            file_path: Path to the file
            file_type: Type of context (e.g., 'cv', 'document')
            metadata: Additional metadata about the file
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise DocumentError(f"File not found: {file_path}")

            mime_type = mimetypes.guess_type(file_path)[0]
            if not mime_type:
                raise ValidationError(f"Unsupported file type: {file_path}")

            handler = self._file_handlers.get(mime_type)
            if not handler:
                raise ServiceError(f"No handler for mime type: {mime_type}")

            content = await handler(path)

            self.files[str(path)] = {
                'content': content,
                'type': file_type,
                'mime_type': mime_type,
                'metadata': metadata or {},
                'added_at': str(datetime.now())
            }

        except DocumentError as e:
            raise e
        except ValidationError as e:
            raise e
        except ServiceError as e:
            raise e
        except Exception as e:
            raise StreamError(f"Failed to add file {file_path}: {e}")

    async def add_data(
        self,
        data: Dict[str, Any],
        category: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add structured data to the context.

        Args:
            data: Data dictionary to add
            category: Category of the data
            metadata: Additional metadata about the data
        """
        if category not in self.structured_data:
            self.structured_data[category] = []

        self.structured_data[category].append({
            'data': data,
            'metadata': metadata or {},
            'added_at': str(datetime.now())
        })

    def get_formatted_context(self) -> str:
        """Get formatted context string for model input."""
        context_parts = []

        # Add file contexts
        for file_path, file_info in self.files.items():
            context_parts.append(
                f"Context from {file_info['type']} "
                f"({Path(file_path).name}):\n"
                f"{file_info['content']}\n"
            )

        # Add structured data contexts
        for category, items in self.structured_data.items():
            for item in items:
                context_parts.append(
                    f"Additional {category} information:\n"
                    f"{json.dumps(item['data'], indent=2)}\n"
                )

        return "\n".join(context_parts) if context_parts else ""

    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of all context data used."""
        return {
            'files': [
                {
                    'path': str(path),
                    'type': info['type'],
                    'mime_type': info['mime_type'],
                    'metadata': info['metadata']
                }
                for path, info in self.files.items()
            ],
            'structured_data': [
                {
                    'category': category,
                    'count': len(items),
                    'metadata': [item['metadata'] for item in items]
                }
                for category, items in self.structured_data.items()
            ]
        }

    async def clear_context(
        self,
        context_type: Optional[str] = None
    ) -> None:
        """Clear specified type of context or all context."""
        if context_type == 'files':
            self.files.clear()
        elif context_type == 'structured_data':
            self.structured_data.clear()
        elif context_type is None:
            self.files.clear()
            self.structured_data.clear()
            self.metadata.clear()

    async def _handle_text(self, path: Path) -> str:
        """Handle text file reading."""
        async with aiofiles.open(path, 'r') as file:
            return await file.read()

    async def _handle_json(self, path: Path) -> str:
        """Handle JSON file reading."""
        async with aiofiles.open(path, 'r') as file:
            content = await file.read()
            try:
                parsed = json.loads(content)
                return json.dumps(parsed, indent=2)
            except json.JSONDecodeError as e:
                raise DocumentError(f"Invalid JSON file {path}: {e}")

    async def _handle_pdf(self, path: Path) -> str:
        """Handle PDF file reading."""
        try:
            # This would use PyPDF2 or similar library
            # For now, return placeholder
            return f"[PDF content from {path}]"
        except Exception as e:
            raise ServiceError(f"Failed to process PDF {path}: {e}")

    async def _handle_csv(self, path: Path) -> str:
        """Handle CSV file reading."""
        try:
            # This would use pandas or similar library
            # For now, return placeholder
            return f"[CSV content from {path}]"
        except Exception as e:
            raise ServiceError(f"Failed to process CSV {path}: {e}")

    async def _handle_docx(self, path: Path) -> str:
        """Handle DOCX file reading."""
        try:
            # This would use python-docx or similar library
            # For now, return placeholder
            return f"[DOCX content from {path}]"
        except Exception as e:
            raise ServiceError(f"Failed to process DOCX {path}: {e}")

    async def add_turn(
        self,
        speaker: str,
        content: str,
        context: Dict[str, Any]
    ) -> ConversationTurn:
        """Add and analyze new conversation turn using AI."""
        turn = ConversationTurn(
            speaker=speaker,
            content=content,
            timestamp=datetime.now(),
            state=self.current_state,
            context=context
        )

        # Get AI analysis of the turn
        turn.analysis = await self._analyze_turn(turn)

        self.turns.append(turn)
        await self._update_state(turn)
        await self._update_topics(turn)
        return turn

    async def _analyze_turn(
        self,
        turn: ConversationTurn
    ) -> Dict[str, Any]:
        """Analyze conversation turn using AI."""
        recent_context = self.get_recent_context(5)
        context_str = "\n".join([
            f"{t.speaker}: {t.content}" for t in recent_context
        ])

        prompt = (
            "Analyze this conversation turn in context. "
            "Provide analysis in JSON format including:\n"
            "1. Speaker's intent and tone\n"
            "2. Key points made\n"
            "3. Relation to previous context\n"
            "4. Topics discussed\n"
            "5. Conversation flow impact\n\n"
            f"Recent context:\n{context_str}\n\n"
            f"Current turn ({turn.speaker}): {turn.content}"
        )

        async for response in self.ai.send_message(prompt):
            if response.text:
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    continue

        return {}

    async def _update_state(self, turn: ConversationTurn) -> None:
        """Update conversation state based on new turn."""
        if self.current_state == ConversationState.INITIAL:
            self.current_state = ConversationState.ACTIVE
        elif '?' in turn.content:
            self.current_state = ConversationState.WAITING_RESPONSE
        elif any(
            phrase in turn.content.lower()
            for phrase in ['could you clarify', 'what do you mean']
        ):
            self.current_state = ConversationState.CLARIFICATION
        elif len(self.turns) > 10 and any(
            phrase in turn.content.lower()
            for phrase in ['thank you', 'goodbye', 'conclude']
        ):
            self.current_state = ConversationState.CONCLUDING

    async def _update_topics(self, turn: ConversationTurn) -> None:
        """Update active topics based on conversation content."""
        # Extract potential topics from content
        words = turn.content.lower().split()
        potential_topics = [
            w for w in words
            if len(w) > 4 and w not in self.get_common_words()
        ]

        # Update active topics
        self.active_topics.extend(
            topic for topic in potential_topics
            if topic not in self.active_topics
        )

        # Keep only recent topics (last 5)
        self.active_topics = self.active_topics[-5:]

    def get_recent_context(self, turns: int = 3) -> List[ConversationTurn]:
        """Get recent conversation turns with context."""
        return self.turns[-turns:] if self.turns else []

    def get_topic_history(self, topic: str) -> List[ConversationTurn]:
        """Get all turns related to a specific topic."""
        return [
            turn for turn in self.turns
            if topic.lower() in turn.content.lower()
        ]

    @staticmethod
    def get_common_words() -> set:
        """Get set of common words to filter out."""
        return {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
            'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
            'if', 'about', 'who', 'get', 'which', 'go', 'me'
        }

    async def get_summary(self) -> Dict[str, Any]:
        """Get AI-generated conversation summary."""
        if not self.turns:
            return {}

        conversation = "\n".join([
            f"{turn.speaker}: {turn.content}"
            for turn in self.turns
        ])

        prompt = (
            "Provide a comprehensive summary of this conversation. "
            "Include in JSON format:\n"
            "1. Main topics discussed\n"
            "2. Key decisions or conclusions\n"
            "3. Action items\n"
            "4. Important insights\n"
            "5. Follow-up needs\n\n"
            f"Conversation:\n{conversation}"
        )

        async for response in self.ai.send_message(prompt):
            if response.text:
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    continue

        return {}
