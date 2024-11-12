class AudioError(Exception):
    """Base exception for all audio-related errors."""
    pass


class DeviceError(AudioError):
    """Raised when there are issues with audio devices."""
    pass


class DeviceNotFoundError(DeviceError):
    """Raised when a specified audio device cannot be found."""
    pass


class DeviceConfigError(DeviceError):
    """Raised when there's an error configuring an audio device."""
    pass


class DeviceInUseError(DeviceError):
    """Raised when attempting to use an audio device that's already in use."""
    pass


class ProcessingError(AudioError):
    """Raised when audio processing operations fail."""
    pass


class CaptureError(AudioError):
    """Raised when audio capture operations fail."""
    pass


class StreamError(AudioError):
    """Raised when audio streaming operations fail."""
    pass


class MixerError(AudioError):
    """Raised when audio mixing operations fail."""
    pass


class InvalidAudioDataError(AudioError):
    """Raised when audio data is invalid or corrupted."""
    pass


class BufferError(AudioError):
    """Raised when there are issues with audio buffers."""
    pass


class FormatError(AudioError):
    """Raised when there are audio format incompatibilities."""
    pass


class CalibrationError(AudioError):
    """Raised when audio calibration fails."""
    pass
