"""Custom exceptions for response generation."""
from typing import Optional

class ResponseError(Exception):
    """Base class for response-related errors."""
    def __init__(self, message: str, response_type: Optional[str] = None):
        super().__init__(message)
        self.response_type = response_type

class TemplateError(ResponseError):
    """Raised when there is an error with response templates."""
    def __init__(self, message: str, template_id: Optional[str] = None):
        super().__init__(message, response_type="template")
        self.template_id = template_id

class CandidateSelectionError(ResponseError):
    """Raised when candidate selection fails."""
    def __init__(self, message: str, candidates: Optional[list] = None):
        super().__init__(message, response_type="candidate_selection")
        self.candidates = candidates

class ResponseEnhancementError(ResponseError):
    """Raised when an error occurs during response enhancement."""
    def __init__(self, message: str, enhancement_stage: Optional[str] = None):
        super().__init__(message, response_type="enhancement")
        self.enhancement_stage = enhancement_stage

class StreamError(ResponseError):
    """Raised when streaming fails during response generation."""
    def __init__(self, message: str, stream_id: Optional[str] = None):
        super().__init__(message, response_type="stream")
        self.stream_id = stream_id

class FallbackError(ResponseError):
    """Raised when generating a fallback response fails."""
    def __init__(self, message: str):
        super().__init__(message, response_type="fallback")
