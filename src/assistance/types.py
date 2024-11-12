from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from dataclasses import field


class ContextType(str, Enum):
    INTERVIEW = "interview"
    SUPPORT = "support"
    MEETING = "meeting"
    GENERAL = "general"


@dataclass
class Suggestion:
    text: str
    confidence: float
    context: Dict[str, Any]
    timestamp: datetime


@dataclass
class ActionItem:
    description: str
    assignee: Optional[str]
    deadline: Optional[datetime]
    status: str
    priority: str
    context: Dict[str, Any]
    timestamp: datetime


@dataclass
class Insight:
    category: str
    content: str
    importance: float
    related_context: Dict[str, Any]
    timestamp: datetime


@dataclass
class ConversationAnalysis:
    key_points: List[str]
    action_items: List[ActionItem]
    questions: List[str]
    follow_up_topics: List[str]
    context_specific: Dict[str, Any]
    timestamp: datetime
    validated: bool = False
    validation_notes: Optional[str] = None

    def validate(self, notes: Optional[str] = None) -> None:
        """Mark analysis as validated with optional notes."""
        self.validated = True
        self.validation_notes = notes

    def invalidate(self, notes: Optional[str] = None) -> None:
        """Mark analysis as invalid with required notes."""
        self.validated = False
        self.validation_notes = notes


@dataclass
class InterviewAnalysis(ConversationAnalysis):
    candidate_strengths: List[str] = field(default_factory=list)
    areas_to_explore: List[str] = field(default_factory=list)
    qualification_matches: Dict[str, float] = field(default_factory=dict)
    cv_references: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SupportAnalysis(ConversationAnalysis):
    issue_type: str = ""
    issue_status: str = ""
    solution_progress: float = 0.0
    customer_sentiment: str = ""
    relevant_docs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MeetingAnalysis(ConversationAnalysis):
    agenda_progress: Dict[str, float] = field(default_factory=dict)
    decisions_made: List[str] = field(default_factory=list)
    attendance: List[str] = field(default_factory=list)
    time_tracking: Dict[str, float] = field(default_factory=dict)
