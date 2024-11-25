"""Core conversation types supporting all roles."""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field


class Role(str, Enum):
    """All possible conversation roles."""

    # Interview roles
    INTERVIEWER = "interviewer"
    INTERVIEWEE = "interviewee"

    # Support roles
    SUPPORT_AGENT = "support_agent"
    CUSTOMER = "customer"

    # Meeting roles
    MEETING_HOST = "meeting_host"
    MEETING_PARTICIPANT = "meeting_participant"


class ConversationStage(str, Enum):
    """Stages of different conversation types."""

    # Interview stages
    INTERVIEW_INTRO = "interview_introduction"
    TECHNICAL_QUESTIONS = "technical_questions"
    EXPERIENCE_DISCUSSION = "experience_discussion"
    PROJECT_DETAILS = "project_details"
    CANDIDATE_QUESTIONS = "candidate_questions"

    # Support stages
    ISSUE_IDENTIFICATION = "issue_identification"
    TROUBLESHOOTING = "troubleshooting"
    SOLUTION_IMPLEMENTATION = "solution_implementation"
    VERIFICATION = "verification"

    # Meeting stages
    MEETING_OPENING = "meeting_opening"
    AGENDA_DISCUSSION = "agenda_discussion"
    DECISION_MAKING = "decision_making"
    ACTION_ITEMS = "action_items"
    MEETING_CLOSING = "meeting_closing"


class ResponseType(str, Enum):
    """Types of responses the system can generate."""

    # General responses
    SUGGESTION = "suggestion"  # Real-time suggestions
    GUIDANCE = "guidance"  # Step-by-step guidance
    QUESTION = "question"  # Generated questions
    ANALYSIS = "analysis"  # Content analysis
    FEEDBACK = "feedback"  # Direct feedback
    SUMMARY = "summary"  # Summarized content

    # Role-specific responses
    TECHNICAL_EVALUATION = "technical_evaluation"  # For technical assessment
    SOLUTION_STEPS = "solution_steps"  # For support solutions
    DISCUSSION_POINTS = "discussion_points"  # For meeting contributions


class Source(BaseModel):
    bytes: Optional[bytes]


class Image(BaseModel):
    format: str


class Document(BaseModel):
    format: str
    name: str
    source: Source


class ToolUse(BaseModel):
    toolUseId: str
    name: str
    input: Union[dict, list, int, float, str, bool, None]


class ToolResultContent(BaseModel):
    json: Optional[Union[dict, list, int, float, str, bool, None]] = None
    text: Optional[str] = None
    image: Optional[Image] = None
    document: Optional[Document] = None


class ToolResult(BaseModel):
    toolUseId: str
    content: List[ToolResultContent]
    status: str


class GuardContentText(BaseModel):
    text: str
    qualifiers: Optional[List[str]] = None


class GuardContent(BaseModel):
    text: Optional[GuardContentText] = None


class ContentBlock(BaseModel):
    text: Optional[str] = None
    image: Optional[Image] = None
    document: Optional[Document] = None
    toolUse: Optional[ToolUse] = None
    toolResult: Optional[ToolResult] = None
    guardContent: Optional[GuardContent] = None


class Message(BaseModel):
    role: str
    content: List[ContentBlock]


class SystemContent(BaseModel):
    text: Optional[str] = None
    guardContent: Optional[GuardContent] = None


class InferenceConfig(BaseModel):
    maxTokens: Optional[int] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    stopSequences: Optional[List[str]] = None


class ToolSpec(BaseModel):
    name: str
    description: Optional[str] = None
    inputSchema: Union[dict, list, int, float, str, bool, None]


class ToolChoice(BaseModel):
    auto: Optional[dict] = None
    any: Optional[dict] = None
    tool: Optional[Dict[str, str]] = None


class ToolConfig(BaseModel):
    tools: List[ToolSpec]
    toolChoice: ToolChoice


class GuardrailConfig(BaseModel):
    guardrailIdentifier: str
    guardrailVersion: str
    trace: Optional[str] = None
    streamProcessingMode: Optional[str] = None


class PromptVariable(BaseModel):
    text: str


class Request(BaseModel):
    messages: List[Message]
    system: Optional[List[SystemContent]] = None
    modelId: Optional[str] = None
    inferenceConfig: Optional[InferenceConfig] = None
    toolConfig: Optional[ToolConfig] = None
    guardrailConfig: Optional[GuardrailConfig] = None
    additionalModelRequestFields: Union[dict, list, int, float, str, bool, None] = None
    promptVariables: Optional[Dict[str, PromptVariable]] = None
    additionalModelResponseFieldPaths: Optional[List[str]] = None


class ConversationState(BaseModel):
    """Current state of a conversation."""

    session_id: str
    role: Role
    current_stage: ConversationStage
    previous_messages: List[Message]
    context_refs: List[str]  # References to context entries
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    """Usage metrics for a conversation."""

    inputTokens: int
    outputTokens: int
    totalTokens: int


class Metrics(BaseModel):
    """Metrics for a conversation."""

    latencyMs: int


class Metadata(BaseModel):
    """Metadata for a conversation."""

    usage: Usage
    metrics: Metrics
    trace: Dict[str, Any]


class ResponseConfig(BaseModel):
    """Configuration for response generation."""

    response_types: List[ResponseType]
    max_length: Optional[int] = None
    include_context: bool = True
    include_analysis: bool = True
    format_json: bool = False
    temperature: float = 0.7


class GeneratedResponse(BaseModel):
    """A generated response."""

    text: str
    response_type: ResponseType
    confidence: float
    context_used: List[str]  # Context references used
    analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationMetrics(BaseModel):
    """Metrics for conversation tracking."""

    total_messages: int = 0
    stage_duration: Dict[ConversationStage, float] = Field(default_factory=dict)
    response_counts: Dict[ResponseType, int] = Field(default_factory=dict)
    average_response_time: float = 0.0
    context_usage: Dict[str, int] = Field(default_factory=dict)
