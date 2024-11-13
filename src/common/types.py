"""Common type definitions shared across modules."""
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel


class Role(str, Enum):
    """Unified role definition."""
    INTERVIEWER = "interviewer"
    INTERVIEWEE = "interviewee"
    SUPPORT_AGENT = "support_agent"
    CUSTOMER = "customer"
    MEETING_HOST = "meeting_host"
    MEETING_PARTICIPANT = "meeting_participant"


class DocumentType(str, Enum):
    """Unified document type definition."""
    CV = "cv"
    JOB_DESCRIPTION = "job_description"
    TECHNICAL_SPEC = "technical_spec"
    PRODUCT_MANUAL = "product_manual"
    SUPPORT_GUIDE = "support_guide"
    MEETING_NOTES = "meeting_notes"
    CODE = "code"
    GENERAL = "general"


class DocumentFormat(str, Enum):
    """Unified document format definition."""
    PDF = "pdf"
    CSV = "csv"
    DOC = "doc"
    DOCX = "docx"
    XLS = "xls"
    XLSX = "xlsx"
    HTML = "html"
    TXT = "txt"
    MD = "md"


class BedrockConfig(BaseModel):
    """Unified Bedrock configuration."""
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: Optional[list[str]] = None
    tool_config: Optional[Dict[str, Any]] = None
    guardrail_config: Optional[Dict[str, Any]] = None


class Document(BaseModel):
    """Unified document model."""
    content: bytes
    format: DocumentFormat
    name: str
    doc_type: DocumentType
    metadata: Dict[str, Any] = {}