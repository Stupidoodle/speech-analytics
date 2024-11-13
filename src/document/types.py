"""Document processing types and models."""
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentType(str, Enum):
    """Types of documents that can be processed."""
    CV = "cv"
    JOB_DESCRIPTION = "job_description"
    TECHNICAL_SPEC = "technical_spec"
    PRODUCT_MANUAL = "product_manual"
    SUPPORT_GUIDE = "support_guide"
    MEETING_NOTES = "meeting_notes"
    CODE = "code"
    GENERAL = "general"


class ProcessingPriority(str, Enum):
    """Processing priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProcessingContext(BaseModel):
    """Context for document processing."""
    role: str = Field(..., description="Role performing the processing")
    document_type: DocumentType
    purpose: str = Field(..., description="Purpose of processing")
    priority: ProcessingPriority = Field(
        default=ProcessingPriority.MEDIUM,
        description="Processing priority"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class ProcessingResult(BaseModel):
    """Result of document processing."""
    document_id: str
    content_type: str
    analysis: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float
    context_updates: Dict[str, Any]
    extracted_at: datetime = Field(default_factory=datetime.now)
    role_specific: Dict[str, Any] = Field(
        default_factory=dict,
        description="Role-specific analysis results",

    )


class DocumentReference(BaseModel):
    """Reference to a processed document."""
    id: str
    type: DocumentType
    name: str
    summary: str
    key_points: List[str]
    relevance: float = Field(ge=0.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict)


class ContextUpdate(BaseModel):
    """Context update from document processing."""
    type: str
    content: Dict[str, Any]
    priority: ProcessingPriority
    timestamp: datetime = Field(default_factory=datetime.now)
    requires_action: bool = False
    action_details: Optional[Dict[str, Any]] = None


class ProcessedDocument(BaseModel):
    """Processed document."""
    document_id: str
    document_type: DocumentType
    content_type: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    context_updates: List[ContextUpdate]
    extracted_at: datetime = Field(default_factory=datetime.now)
    role_specific: Dict[str, Any] = Field(default_factory=dict)