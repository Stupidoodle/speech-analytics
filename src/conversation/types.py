from enum import Enum
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel


class MessageRole(str, Enum):
    """Roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Role(str, Enum):
    """User roles in system."""
    INTERVIEWER = "interviewer"
    INTERVIEWEE = "interviewee"
    SUPPORT_AGENT = "support_agent"
    CUSTOMER = "customer"
    MEETING_HOST = "meeting_host"
    MEETING_PARTICIPANT = "meeting_participant"


class MessageType(str, Enum):
    """Types of messages."""
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    CONTEXT = "context"
    SYSTEM = "system"


class SessionState(str, Enum):
    """States of conversation sessions."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class MessageContent:
    """Content of a message."""
    type: MessageType
    text: Optional[str] = None
    tool_use: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    context_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """A conversation message."""
    role: MessageRole
    content: List[MessageContent]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionConfig(BaseModel):
    """Configuration for conversation session."""
    role: Optional[Role] = None
    max_turns: Optional[int] = None
    auto_context: bool = True
    tool_configs: Dict[str, Dict[str, Any]] = {}
    custom_prompts: Dict[str, str] = {}
    guardrails: Dict[str, Any] = {}


class BedrockConfig(BaseModel):
    """Configuration for Bedrock service."""
    model_id: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None
    tool_config: Optional[Dict[str, Any]] = None
    guardrail_config: Optional[Dict[str, Any]] = None


class StreamResponse(BaseModel):
    """Response from streaming conversation."""
    class Delta(BaseModel):
        """Content delta in stream."""
        text: Optional[str] = None
        tool_use: Optional[Dict[str, Any]] = None

    class Metadata(BaseModel):
        """Stream metadata."""
        usage: Dict[str, int]
        metrics: Dict[str, int]
        trace: Optional[Dict[str, Any]] = None

    content: Optional[Delta] = None
    metadata: Optional[Metadata] = None
    stop_reason: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


class SystemPrompt(BaseModel):
    """System prompt configuration."""
    text: str
    metadata: Dict[str, Any] = {}
    priority: int = 0
    condition: Optional[str] = None


class ToolConfig(BaseModel):
    """Configuration for conversation tools."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: bool = False
    timeout: float = 30.0
