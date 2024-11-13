from typing import Dict, Any, List, Optional, Literal
from enum import Enum
from pydantic import BaseModel
from datetime import datetime

class ImageFormat(str, Enum):
    """Supported image formats."""
    PNG = "png"
    JPEG = "jpeg"
    GIF = "gif"
    WEBP = "webp"


class DocumentFormat(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    CSV = "csv"
    DOC = "doc"
    DOCX = "docx"
    XLS = "xls"
    XLSX = "xlsx"
    HTML = "html"
    TXT = "txt"
    MD = "md"


class MessageRole(str, Enum):
    """Message roles for conversation."""
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A conversation Message"""
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class StopReason(str, Enum):
    """Reasons for message stop."""
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    GUARDRAIL_INTERVENED = "guardrail_intervened"
    CONTENT_FILTERED = "content_filtered"


class ImageContent(BaseModel):
    """Image content in messages."""
    format: ImageFormat
    source: bytes


class DocumentContent(BaseModel):
    """Document content in messages."""
    format: DocumentFormat
    name: str
    source: bytes


class ToolUse(BaseModel):
    """Tool use in messages."""
    tool_use_id: str
    name: str
    input: Any


class ToolResult(BaseModel):
    """Tool result in messages."""
    tool_use_id: str
    content: List[Dict[str, Any]]
    status: Literal["success", "error"]


class InferenceConfig(BaseModel):
    max_tokens: int
    temperature: float
    top_p: float


class BedrockConfig(BaseModel):
    """Bedrock configuration."""
    model_id: str
    inference_config: InferenceConfig
    tool_config: Optional[Dict[str, Any]] = None
    guardrail_config: Optional[Dict[str, Any]] = None


class StreamResponse(BaseModel):
    """Streaming response from Bedrock."""
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
    stop_reason: Optional[StopReason] = None
    error: Optional[Dict[str, Any]] = None


class Role(Enum):
    """
    Enum for user roles in a conversation
    """
    INTERVIEWER = "interviewer"
    INTERVIEWEE = "interviewee"
    SUPPORT_AGENT = "support_agent"
    CUSTOMER = "customer"
    MEETING_HOST = "meeting_host"
    MEETING_PARTICIPANT = "meeting_participant"


class Document(BaseModel):
    """
    Document model
    """
    content: bytes
    mime_type: DocumentFormat
    # TODO: Add String sanitization
    name: str
    metadata: Dict[str, Any] = {}