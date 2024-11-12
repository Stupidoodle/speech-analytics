from typing import Dict, Any, List, AsyncIterator
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

from ..conversation.context import ConversationContext
from ..conversation.manager import ConversationManager
from .assistant import ConversationAssistant
from src.conversation.roles import Role


@dataclass
class AssistanceResponse:
    suggestion: str
    confidence: float
    context: Dict[str, Any]
    type: str
    priority: float
    timestamp: datetime


class AssistanceType(Enum):
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"
    INSIGHT = "insight"
    WARNING = "warning"
    ACTION = "action"


class EnhancedAssistant:
    """Enhanced assistant with role-specific behavior."""
    
    def __init__(
        self,
        conversation_context: ConversationContext,
        conversation_manager: ConversationManager,
        role: Role,
        priority_threshold: float = 0.5
    ):
        self.context = conversation_context
        self.ai = conversation_manager
        self.role = role
        self.priority_threshold = priority_threshold
        self.assistance_history: List[AssistanceResponse] = []

    async def process_turn(
        self,
        turn_content: str,
        speaker: str
    ) -> AsyncIterator[AssistanceResponse]:
        """Process conversation turn with role-specific assistance."""
        role_context = self.role.get_prompt_context()
        recent_turns = self.context.get_recent_context(5)
        conversation_history = "\n".join([
            f"{t.speaker}: {t.content}" for t in recent_turns
        ])

        prompt = (
            f"As a {self.role.value}, analyze this conversation turn.\n\n"
            f"Role Context:\n{role_context}\n\n"
            f"History:\n{conversation_history}\n\n"
            f"Current turn ({speaker}): {turn_content}\n\n"
            "Provide role-appropriate assistance including:\n"
            "1. Response suggestions\n"
            "2. Follow-up questions\n"
            "3. Points to emphasize\n"
            "4. Areas to explore\n"
            "5. Potential concerns\n\n"
            "Format: JSON with 'type', 'suggestion', 'confidence', "
            "'priority', and 'context' for each response."
        )

        # Get AI analysis
        async for response in self.ai.send_message(prompt):
            if response.text:
                try:
                    analyses = json.loads(response.text)
                    for analysis in analyses:
                        assistance = AssistanceResponse(
                            suggestion=analysis['suggestion'],
                            confidence=analysis['confidence'],
                            context=analysis['context'],
                            type=analysis['type'],
                            priority=analysis['priority'],
                            timestamp=datetime.now()
                        )
                        self.assistance_history.append(assistance)
                        yield assistance
                except json.JSONDecodeError:
                    continue
