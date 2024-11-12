class ConversationError(Exception):
    """Base exception for conversation errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class StreamError(ConversationError):
    """Exception for streaming errors."""
    pass


class ModelError(ConversationError):
    """Exception for model-related errors."""
    pass


class ValidationError(ConversationError):
    """Exception for validation errors."""
    pass


class ThrottlingError(ConversationError):
    """Exception for rate limiting."""
    pass


class ServiceError(ConversationError):
    """Exception for service availability issues."""
    pass


class ContentFilterError(ConversationError):
    """Exception for content filtering issues."""
    pass


class GuardrailError(ConversationError):
    """Exception for guardrail violations."""
    pass


class DocumentError(ConversationError):
    """Exception for document handling errors."""
    pass
