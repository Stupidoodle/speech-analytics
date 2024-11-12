class TranscriptionError(Exception):
    """Base exception for transcription errors."""
    pass


class ServiceException(TranscriptionError):
    """Base exception for AWS service errors."""

    def __init__(self, message=None, error_code=None, status_code=None):
        super().__init__(message)
        self.message = message or "Unknown service error"
        self.error_code = error_code or "Unknown"
        self.status_code = status_code or 500


class BadRequestException(ServiceException):
    """Exception raised when request is malformed."""
    pass


class ConflictException(ServiceException):
    """Exception raised when there is a conflict with the current state."""
    pass


class InternalFailureException(ServiceException):
    """Exception raised when service encounters internal error."""
    pass


class LimitExceededException(ServiceException):
    """Exception raised when service limits are exceeded."""
    pass


class ServiceUnavailableException(ServiceException):
    """Exception raised when service is unavailable."""
    pass


class ValidationException(ServiceException):
    """Exception raised when validation fails."""
    pass


class CredentialsException(ServiceException):
    """Exception raised when there are credential issues."""
    pass


class UnknownServiceException(ServiceException):
    """Exception raised for unknown service errors."""
    pass


class TranscriptionConfigError(TranscriptionError):
    """Exception raised for configuration errors."""
    pass


class StreamingError(TranscriptionError):
    """Exception raised for streaming-related errors."""
    pass


class RateLimitError(TranscriptionError):
    """Exception raised when rate limits are hit."""
    pass


class QualityError(TranscriptionError):
    """Exception raised for transcription quality issues."""
    pass


class ConnectionError(TranscriptionError):
    """Exception raised for connection issues."""
    pass
