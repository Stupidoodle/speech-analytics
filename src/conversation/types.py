from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ContentFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    GIF = "gif"
    WEBP = "webp"


class DocumentFormat(str, Enum):
    PDF = "pdf"
    CSV = "csv"
    DOC = "doc"
    DOCX = "docx"
    XLS = "xls"
    XLSX = "xlsx"
    HTML = "html"
    TXT = "txt"
    MD = "md"


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    GUARDRAIL_INTERVENED = "guardrail_intervened"
    CONTENT_FILTERED = "content_filtered"


@dataclass
class ImageContent:
    format: ContentFormat
    source: bytes


@dataclass
class DocumentContent:
    format: DocumentFormat
    name: str
    source: bytes


@dataclass
class ToolUse:
    tool_use_id: str
    name: str
    input: Any


@dataclass
class ToolResult:
    tool_use_id: str
    status: str
    content: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MessageContent:
    text: Optional[str] = None
    image: Optional[ImageContent] = None
    document: Optional[DocumentContent] = None
    tool_use: Optional[ToolUse] = None
    tool_result: Optional[ToolResult] = None


@dataclass
class Message:
    role: Role
    content: List[MessageContent]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InferenceConfig:
    max_tokens: int
    temperature: float
    top_p: float
    stop_sequences: Optional[List[str]] = None


@dataclass
class StreamMetadata:
    usage: Dict[str, int]
    metrics: Dict[str, int]
    trace: Optional[Dict[str, Any]] = None


@dataclass
class StreamResponse:
    text: Optional[str] = None
    tool_use: Optional[Dict[str, str]] = None
    metadata: Optional[StreamMetadata] = None
    stop_reason: Optional[StopReason] = None
    error: Optional[Dict[str, Any]] = None
