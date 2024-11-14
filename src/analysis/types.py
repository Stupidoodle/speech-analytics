"""Core types for the analysis layer."""
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import BaseModel

from src.conversation.types import Role
from src.context.types import ContextEntry, ContextMetadata


class AnalysisType(str, Enum):
    """Types of analysis that can be performed."""
    CONVERSATION = "conversation"
    SENTIMENT = "sentiment"
    TOPIC = "topic"
    QUALITY = "quality"
    ENGAGEMENT = "engagement"
    SUMMARY = "summary"
    ROLE_SPECIFIC = "role_specific"
    CUSTOM = "custom"
    BEHAVIORAL = "behavioral"
    COMPLIANCE = "compliance"


class AnalysisPriority(float, Enum):
    """Priority levels for analysis tasks."""
    CRITICAL = 3.0  # Real-time, blocking
    HIGH = 2.0      # Real-time, non-blocking
    MEDIUM = 1.0    # Near real-time
    LOW = 0.5       # Background


class AnalysisState(str, Enum):
    """States of analysis tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class AnalysisMetrics:
    """Metrics for conversation analysis."""
    duration: float = 0.0
    turn_count: int = 0
    speaker_ratio: Dict[str, float] = field(default_factory=dict)
    avg_response_time: float = 0.0
    topic_distribution: Dict[str, float] = field(default_factory=dict)
    engagement_score: float = 0.0
    clarity_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AnalysisInsight:
    """Individual analysis insight."""
    type: AnalysisType
    content: Any
    confidence: float
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    references: Set[str] = field(default_factory=set)


class AnalysisTask(BaseModel):
    """Task configuration for analysis."""
    id: str
    type: AnalysisType
    priority: AnalysisPriority
    role: Optional[Role] = None
    config: Dict[str, Any] = {}
    dependencies: List[str] = []
    timeout: Optional[float] = None


class AnalysisResult(BaseModel):
    """Result of analysis task."""
    task_id: str
    type: AnalysisType
    insights: List[AnalysisInsight]
    metrics: Optional[AnalysisMetrics] = None
    confidence: float
    duration: float
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class AnalysisSummary(BaseModel):
    """Summary of analysis results."""
    key_points: List[str]
    topics: List[Dict[str, float]]
    sentiment: Dict[str, float]
    action_items: List[Dict[str, Any]]
    metrics: AnalysisMetrics
    recommendations: List[Dict[str, Any]]
    context_updates: List[Dict[str, Any]]


class AnalysisConfig(BaseModel):
    """Configuration for analysis system."""
    enabled_analyzers: Set[AnalysisType]
    priority_threshold: AnalysisPriority = AnalysisPriority.MEDIUM
    max_concurrent_tasks: int = 10
    default_timeout: float = 30.0
    role_configs: Dict[Role, Dict[str, Any]] = {}
    custom_analyzers: Dict[str, Dict[str, Any]] = {}


class AnalysisPipeline(BaseModel):
    """Configuration for analysis pipeline."""
    stages: List[Dict[str, List[AnalysisTask]]]
    parallel_stages: bool = True
    max_stage_duration: float = 60.0
    error_handling: str = "continue"
    fallback_handlers: Dict[str, str] = {}


class AnalysisRequest(BaseModel):
    """Request for analysis processing."""
    session_id: str
    content: Dict[str, Any]
    context: Optional[ContextEntry] = None
    metadata: Optional[ContextMetadata] = None
    config: Optional[AnalysisConfig] = None
    pipeline: Optional[AnalysisPipeline] = None