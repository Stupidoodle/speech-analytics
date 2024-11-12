from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.conversation.manager import ConversationManager
from src.assistance.types import ConversationAnalysis, ActionItem


class BaseAnalyzer(ABC):
    """Base class for all analyzers with common functionality."""

    def __init__(self, conversation_manager: ConversationManager):
        self.conversation = conversation_manager
        self.context_cache: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    async def analyze_segment(
        self,
        segment: str,
        timestamp: Optional[datetime] = None
    ) -> ConversationAnalysis:
        """Analyze conversation segment."""
        pass

    async def _parse_base_analysis(self, analysis: str) -> Dict[str, Any]:
        """Common analysis parsing logic."""
        sections = {}
        current_section = None

        for line in analysis.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line[0].isdigit() and '.' in line[:3]:
                current_section = line.split('.', 1)[1].strip().lower()
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

        return sections

    async def _extract_action_items(
        self,
        items: List[str],
        context: Dict[str, Any]
    ) -> List[ActionItem]:
        """Common action item extraction logic."""
        action_items = []
        for item in items:
            action_items.append(
                ActionItem(
                    description=item,
                    assignee=self._determine_assignee(item),
                    deadline=None,
                    status="pending",
                    priority=self._determine_priority(item),
                    context=context,
                    timestamp=datetime.now()
                )
            )
        return action_items
