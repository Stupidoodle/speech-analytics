"""Response layer type definitions."""
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel

from src.analysis.types import AnalysisResult
from src.context.types import ContextEntry
from src.conversation.types import Role, MessageRole

class ResponseType(str, Enum):
    """Types of generated responses."""
    DIRECT = "direct"           # Direct answer
    CLARIFICATION = "clarify"   # Asking for clarification
    FOLLOW_UP = "follow_up"     # Follow-up question
    SUGGESTION = "suggest"      # Suggestion/recommendation
    SUMMARY = "summary"         # Summarizing information
    ACTION = "action"           # Action-oriented response
    FALLBACK = "fallback"       # Fallback response

class ResponsePriority(float, Enum):
    """Priority levels for responses."""
    CRITICAL = 3.0  # Must respond immediately
    HIGH = 2.0      # Should respond quickly
    MEDIUM = 1.0    # Normal priority
    LOW = 0.5       # Can be delayed

class ResponseState(str, Enum):
    """States of response generation."""
    PENDING = "pending"     # Awaiting generation
    GENERATING = "generating"  # Currently generating
    COMPLETED = "completed"    # Generation complete
    FAILED = "failed"         # Generation failed
    CANCELED = "canceled"     # Generation canceled

@dataclass
class ResponseTemplate:
    """Template for response generation."""
    name: str
    content: str
    variables: Set[str]
    conditions: Dict[str, Any]
    role_specific: Dict[Role, str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class GenerationConfig(BaseModel):
    """Configuration for response generation."""
    max_length: int = 1000
    temperature: float = 0.7
    include_context: bool = True
    use_templates: bool = True
    filter_responses: bool = True
    role_specific: bool = True
    metadata: Dict[str, Any] = {}

class ResponseRequest(BaseModel):
    """Request for response generation."""
    query: str
    role: Optional[Role] = None
    response_type: Optional[ResponseType] = None
    priority: ResponsePriority = ResponsePriority.MEDIUM
    context: Optional[ContextEntry] = None
    analysis: Optional[AnalysisResult] = None
    config: Optional[GenerationConfig] = None
    metadata: Dict[str, Any] = {}

@dataclass
class GeneratedResponse:
    """Generated response with metadata."""
    content: str
    type: ResponseType
    priority: ResponsePriority
    context_used: List[str]
    analysis_used: List[str]
    template_used: Optional[str] = None
    duration: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class ResponseValidation(BaseModel):
    """Validation result for generated response."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []
    metadata: Dict[str, Any] = {}