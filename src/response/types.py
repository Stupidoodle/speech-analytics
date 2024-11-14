"""Response generation type definitions."""
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel

from src.analysis.types import AnalysisResult
from src.context.types import ContextEntry


class ResponseType(str, Enum):
    """Types of generated responses."""
    DIRECT = "direct"           # Direct answer/response
    CLARIFYING = "clarifying"   # Asking for clarification
    FOLLOW_UP = "follow_up"     # Follow-up question
    SUGGESTION = "suggestion"   # Suggestion/recommendation
    SUMMARY = "summary"         # Summarizing information
    ACTION = "action"           # Action-oriented response
    FALLBACK = "fallback"       # Fallback response


class ResponsePriority(float, Enum):
    """Priority levels for responses."""
    CRITICAL = 3.0
    HIGH = 2.0
    MEDIUM = 1.0
    LOW = 0.5


@dataclass
class ResponseCandidate:
    """Candidate response with metadata."""
    content: str
    type: ResponseType
    confidence: float
    context_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ResponseConfig(BaseModel):
    """Configuration for response generation."""
    max_candidates: int = 3
    min_confidence: float = 0.7
    enable_streaming: bool = True
    default_type: ResponseType = ResponseType.DIRECT
    fallback_responses: Dict[str, List[str]] = {}
    context_weight: float = 0.6
    analysis_weight: float = 0.4


class ResponseResult(BaseModel):
    """Generated response result."""
    content: str
    type: ResponseType
    confidence: float
    alternatives: List[ResponseCandidate]
    context_used: List[str]
    analysis_used: List[str]
    metadata: Dict[str, Any] = {}
    timestamp: datetime = datetime.now()


class ResponseRequest(BaseModel):
    """Request for response generation."""
    query: str
    context: Optional[ContextEntry] = None
    analysis: Optional[AnalysisResult] = None
    response_type: Optional[ResponseType] = None
    config: Optional[ResponseConfig] = None