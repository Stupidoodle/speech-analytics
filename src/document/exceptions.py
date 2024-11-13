"""Document processing exceptions."""
from typing import Dict, Any, Optional


class DocumentError(Exception):
    """Base exception for document processing."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ProcessingError(DocumentError):
    """Error during document processing."""
    pass


class RoleConfigError(DocumentError):
    """Error in role configuration."""
    pass


class ContextError(DocumentError):
    """Error in processing context."""
    pass


class AIProcessingError(DocumentError):
    """Error during AI-based processing."""

    def __init__(
        self,
        message: str,
        model_error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.model_error = model_error


class InvalidFormatError(DocumentError):
    """Error for unsupported or invalid formats."""
    pass


class ContentExtractionError(DocumentError):
    """Error extracting content from document."""
    pass


class DocumentNotFoundError(DocumentError):
    """Document not found error."""
    pass

class StorageError(DocumentError):
    """Storage error."""
    pass

class DocumentValidationError(DocumentError):
    """Document validation error."""
    pass