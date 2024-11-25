from typing import Any, Dict, List, Optional


class ConversationError(Exception):
    """Base exception for conversation errors."""

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.session_id = session_id
        self.details = details or {}


class SessionError(ConversationError):
    """Error in conversation session."""

    def __init__(
        self,
        message: str,
        session_id: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, session_id, details)


class MessageError(ConversationError):
    """Error in message handling."""

    def __init__(
        self,
        message: str,
        session_id: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, session_id, details)


class StreamError(ConversationError):
    """Error in message streaming."""

    def __init__(
        self,
        message: str,
    ):
        super().__init__(message, "0")


class ModelError(ConversationError):
    """Error from AI model."""

    def __init__(
        self,
        message: str,
        service: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, None, details)
        self.service = service
        self.error_code = error_code


class ServiceError(ConversationError):
    """Error in service integration."""

    def __init__(
        self,
        message: str,
        service: str,
        error_code: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, None, details)
        self.service = service
        self.error_code = error_code
        self.status_code = status_code


class ValidationError(ConversationError):
    """Error in message validation."""

    def __init__(
        self,
        message: str,
        validation_errors: List[Dict[str, Any]],
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, None, details)
        self.validation_errors = validation_errors
