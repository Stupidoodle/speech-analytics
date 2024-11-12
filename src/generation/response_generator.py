from typing import Dict, Any, List, AsyncIterator
import json

from src.conversation.roles import Role
from src.conversation.manager import ConversationManager


class ResponseGenerator:
    def __init__(
        self,
        conversation_manager: ConversationManager,
        role: Role
    ):
        self.ai = conversation_manager
        self.role = role

    async def generate_response(
        self,
        query: str,
        context: Dict[str, Any],
        conversation_history: List[str]
    ) -> AsyncIterator[str]:
        """Generate role-appropriate responses."""
        role_context = self.role.get_prompt_context()

        prompt = (
            f"Acting as a {self.role.value}, generate a response.\n\n"
            f"Role Context:\n{role_context}\n\n"
            f"Situation Context:\n{json.dumps(context, indent=2)}\n\n"
            f"Conversation History:\n{conversation_history}\n\n"
            f"Query: {query}\n\n"
            "Generate a response that:\n"
            "1. Aligns with the role's objectives\n"
            "2. Uses appropriate tone and terminology\n"
            "3. Incorporates relevant context\n"
            "4. Maintains professional standards\n"
            "5. Advances the conversation purposefully"
        )

        async for response in self.ai.send_message(prompt):
            if response.text:
                yield response.text

    async def suggest_questions(
        self,
        context: Dict[str, Any],
        conversation_history: List[str]
    ) -> AsyncIterator[str]:
        """Generate role-specific questions."""
        if self.role == Role.INTERVIEWER:
            prompt = (
                "Based on the candidate's profile and responses, "
                "suggest relevant interview questions that:\n"
                "1. Assess technical competency\n"
                "2. Evaluate experience claims\n"
                "3. Probe problem-solving ability\n"
                "4. Explore soft skills\n"
                "5. Clarify career goals"
            )
        elif self.role == Role.SUPPORT_AGENT:
            prompt = (
                "Based on the customer's issue, suggest questions that:\n"
                "1. Clarify the problem\n"
                "2. Gather technical details\n"
                "3. Verify attempted solutions\n"
                "4. Assess impact\n"
                "5. Determine urgency"
            )
        else:
            prompt = (
                "Based on the conversation context, suggest questions that:\n"
                "1. Clarify understanding\n"
                "2. Gather important details\n"
                "3. Move discussion forward\n"
                "4. Address key concerns\n"
                "5. Ensure alignment"
            )

        prompt += f"\n\nContext:\n{json.dumps(context, indent=2)}\n\n"
        prompt += f"Conversation History:\n{conversation_history}"

        async for response in self.ai.send_message(prompt):
            if response.text:
                yield response.text
