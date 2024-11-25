"""Audio processing exceptions."""

from typing import Any, Dict, Optional


class AudioError(Exception):
    """Base exception for audio processing errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CaptureError(AudioError):
    """Error in audio capture."""


class DeviceError(AudioError):
    """Error in audio device operations."""


class DeviceNotFoundError(DeviceError):
    """Specified audio device not found."""

    def __init__(
        self,
        device_id: int,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message or f"Device not found: {device_id}", details)
        self.device_id = device_id


class DeviceConfigError(DeviceError):
    """Error in device configuration."""

    def __init__(
        self,
        message: str,
        config: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details)
        self.config = config


class DeviceInUseError(DeviceError):
    """Device already in use."""

    def __init__(
        self,
        device_id: int,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message or f"Device in use: {device_id}", details)
        self.device_id = device_id


class ProcessingError(AudioError):
    """Error in audio processing."""

    def __init__(
        self,
        message: str,
        processing_step: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details)
        self.processing_step = processing_step


class MixerError(AudioError):
    """Error in audio mixing."""

    def __init__(
        self,
        message: str,
        channels: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details)
        self.channels = channels


class BufferError(AudioError):
    """Error in audio buffer operations."""

    def __init__(
        self,
        message: str,
        buffer_stats: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details)
        self.buffer_stats = buffer_stats


class FormatError(AudioError):
    """Error in audio format handling."""

    def __init__(
        self, message: str, format_type: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.format_type = format_type


class CalibrationError(AudioError):
    """Error in audio calibration."""

    def __init__(
        self,
        message: str,
        calibration_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details)
        self.calibration_type = calibration_type
