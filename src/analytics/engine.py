from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import json

from conversation.context import ConversationContext
from conversation.manager import ConversationManager


@dataclass
class ConversationMetrics:
    turn_count: int
    avg_turn_length: float
    topic_distribution: Dict[str, float]
    speaker_distribution: Dict[str, float]
    clarity_scores: Dict[str, float]
    engagement_metrics: Dict[str, float]
    timestamp: datetime


class AnalyticsEngine:
    def __init__(self, conversation_manager: ConversationManager):
        self.ai = conversation_manager
        self.metrics_history: List[Dict[str, Any]] = []

    async def analyze_conversation(
        self,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Analyze conversation using AI."""
        # Get full conversation history
        conversation_history = "\n".join([
            f"{turn.speaker}: {turn.content}"
            for turn in context.turns
        ])

        prompt = (
            "Analyze this conversation and provide metrics. "
            "Consider engagement, clarity, topic coverage, and effectiveness.\
                \n\n"
            f"Conversation:\n{conversation_history}\n\n"
            "Provide the following metrics in JSON format:\n"
            "1. Engagement metrics (participation, responsiveness)\n"
            "2. Clarity metrics (understanding, communication)\n"
            "3. Topic analysis (coverage, depth, relevance)\n"
            "4. Conversation effectiveness\n"
            "5. Recommendations for improvement"
        )

        # responses = []
        async for response in self.ai.send_message(prompt):
            if response.text:
                try:
                    metrics = json.loads(response.text)
                    self.metrics_history.append(metrics)
                    return metrics
                except json.JSONDecodeError:
                    continue

        return {}
