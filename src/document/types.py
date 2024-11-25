"""Core type definitions for document processing."""

from enum import Enum
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Types of documents the system can process."""

    CV = "cv"
    JOB_DESCRIPTION = "job_description"
    TECHNICAL_SPEC = "technical_spec"
    PRODUCT_MANUAL = "product_manual"
    SUPPORT_GUIDE = "support_guide"
    MEETING_NOTES = "meeting_notes"
    CODE = "code"
    GENERAL = "general"


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


class Document(BaseModel):
    """Base document model."""

    content: bytes
    format: DocumentFormat
    name: str
    doc_type: DocumentType
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessedDocument(BaseModel):
    """Fully processed document with analysis."""

    id: str
    original: Document
    doc_type: DocumentType
    analysis: Dict[str, Any]  # Structured analysis results
    role_specific: Dict[str, Any]  # Role-specific insights
    metadata: Dict[str, Any]
    references: List[str]  # References to other documents
    confidence: float
    processed_at: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


class ProcessingContext(BaseModel):
    """Context for document processing."""

    role: str
    document_type: DocumentType
    purpose: str
    priority: str = "medium"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessingResult(BaseModel):
    """Result of document processing."""

    document_id: str
    content_type: str
    analysis: Dict[str, Any]
    processing_time: float
    context_updates: Dict[str, Any]
    role_specific: Dict[str, Any] = Field(default_factory=dict)
    extracted_at: datetime = Field(default_factory=datetime.now)


class DocumentReference(BaseModel):
    """Reference to a processed document."""

    id: str
    type: DocumentType
    name: str
    summary: str
    key_points: List[str]
    relevance: float = Field(ge=0.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict)
