from enum import Enum
from typing import Dict, Any, Optional
import json

from src.conversation.manager import ConversationManager
from src.conversation.roles import Role
from src.conversation.context import ConversationContext


class ExportFormat(Enum):
    JSON = "json"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"


class ExportManager:
    def __init__(
        self,
        conversation_manager: ConversationManager,
        role: Role
    ):
        self.ai = conversation_manager
        self.role = role
        self.validated_summaries: Dict[str, Dict[str, Any]] = {}

    async def export_conversation(
        self,
        conversation_context: ConversationContext,
        format: ExportFormat,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Export conversation with role-specific formatting."""
        # Generate role-appropriate summary
        summary = await self._generate_summary(conversation_context)

        # Format based on role and export type
        if format == ExportFormat.JSON:
            return await self._format_json(summary, metadata)
        elif format == ExportFormat.PDF:
            return await self._format_pdf(summary, metadata)
        # ... implement other formats

        raise ValueError(f"Unsupported format: {format}")

    async def _generate_summary(
        self,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Generate role-specific conversation summary."""
        conversation = "\n".join([
            f"{turn.speaker}: {turn.content}"
            for turn in context.turns
        ])

        prompt = (
            f"As a {self.role.value}, create a detailed summary of this "
            f"conversation.\n\n"
            f"Role Context:\n{self.role.get_prompt_context()}\n\n"
            f"Conversation:\n{conversation}\n\n"
            "Include in JSON format:\n"
            "1. Key points relevant to the role\n"
            "2. Action items and responsibilities\n"
            "3. Critical insights and decisions\n"
            "4. Follow-up requirements\n"
            "5. Role-specific recommendations"
        )

        async for response in self.ai.send_message(prompt):
            if response.text:
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    continue
        return {}

    async def validate_summary(
        self,
        summary_id: str,
        feedback: str,
        editor_role: Role
    ) -> Dict[str, Any]:
        """Update summary based on role-specific feedback."""
        original = self.validated_summaries.get(summary_id, {})

        prompt = (
            f"As a {editor_role.value}, incorporate this feedback into "
            f"the summary.\n\n"
            f"Original Summary:\n{json.dumps(original, indent=2)}\n\n"
            f"Feedback:\n{feedback}\n\n"
            "Provide updated summary maintaining:\n"
            "1. Professional tone\n"
            "2. Role-appropriate context\n"
            "3. Key information integrity\n"
            "4. Clear action items\n"
            "5. Feedback incorporation"
        )

        async for response in self.ai.send_message(prompt):
            if response.text:
                try:
                    updated = json.loads(response.text)
                    self.validated_summaries[summary_id] = updated
                    return updated
                except json.JSONDecodeError:
                    continue
        return original
