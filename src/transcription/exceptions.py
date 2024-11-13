from typing import Dict, Any, Optional


class TranscriptionError(Exception):
    """Base exception for transcription errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class StreamingError(TranscriptionError):
    """Error in streaming transcription."""
    pass


class ServiceError(TranscriptionError):
    """AWS Transcribe service error."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.error_code = error_code
        self.status_code = status_code


class BadRequestError(ServiceError):
    """Invalid request error."""
    pass


class ConfigurationError(TranscriptionError):
    """Error in transcription configuration."""
    pass


class AudioFormatError(TranscriptionError):
    """Error in audio format."""
    pass


class BufferError(TranscriptionError):
    """Error in audio buffer handling."""
    pass


class ResultError(TranscriptionError):
    """Error in result handling."""
    pass


class VocabularyError(TranscriptionError):
    """Error in vocabulary handling."""
    pass


class ConnectionError(TranscriptionError):
    """Connection error with AWS service."""
    pass


class ThrottlingError(ServiceError):
    """AWS throttling error."""
    pass


class QuotaError(ServiceError):
    """AWS quota exceeded error."""
    pass
