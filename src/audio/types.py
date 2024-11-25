"""Enhanced audio type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Set

from pydantic import BaseModel


class AudioFormat(str, Enum):
    """Supported audio formats."""

    PCM = "pcm"
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"


class DeviceType(str, Enum):
    """Types of audio devices."""

    INPUT = "input"  # Microphone
    OUTPUT = "output"  # Desktop audio
    LOOPBACK = "loopback"


class ProcessingMode(str, Enum):
    """Audio processing modes."""

    REALTIME = "realtime"
    BATCH = "batch"
    ADAPTIVE = "adaptive"


class ChannelConfig(str, Enum):
    """Audio channel configurations."""

    MONO = "mono"
    STEREO = "stereo"


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    # Basic audio settings
    sample_rate: int = 16000  # Required for AWS Transcribe
    channels: int = 2  # Stereo for channel identification
    format: AudioFormat = AudioFormat.PCM
    chunk_duration_ms: int = 100  # 100ms chunks

    # Device configuration
    mic_device_id: Optional[int] = None
    desktop_device_id: Optional[int] = None

    # Channel settings
    channel_config: ChannelConfig = ChannelConfig.STEREO
    left_channel_source: DeviceType = DeviceType.INPUT  # Mic to left
    right_channel_source: DeviceType = DeviceType.OUTPUT  # Desktop to right

    # Processing settings
    enable_noise_reduction: bool = True
    enable_auto_gain: bool = True
    processing_mode: ProcessingMode = ProcessingMode.REALTIME

    @property
    def chunk_size(self) -> int:
        """Calculate chunk size in bytes.

        Returns:
            Chunk size considering stereo PCM format
        """
        bytes_per_sample = 2  # 16-bit audio
        return int(
            (self.chunk_duration_ms / 1000)
            * self.sample_rate
            * self.channels
            * bytes_per_sample
        )


@dataclass
class DeviceInfo:
    """Audio device information."""

    id: int
    name: str
    type: DeviceType
    channels: int
    sample_rate: int
    is_default: bool = False
    is_loopback: bool = False
    supports_input: bool = False
    supports_output: bool = False


@dataclass
class BufferMetrics:
    """Audio buffer metrics."""

    total_bytes_written: int = 0
    total_bytes_read: int = 0
    overflow_count: int = 0
    underrun_count: int = 0
    write_count: int = 0
    read_count: int = 0


class BufferStatus(BaseModel):
    """Audio buffer status."""

    levels: Dict[str, float]  # Buffer fill levels
    latencies: Dict[str, float]  # Buffer latencies
    active_channels: Set[str]  # Active channels
    metrics: BufferMetrics  # Buffer metrics


@dataclass
class AudioMetrics:
    """Audio processing metrics."""

    # Level metrics
    peak_level: float = 0.0
    rms_level: float = 0.0
    noise_level: float = 0.0

    # Quality metrics
    clipping_count: int = 0
    dropout_count: int = 0

    # Performance metrics
    processing_time: float = 0.0
    buffer_stats: Dict[str, float] = field(default_factory=dict)


class ProcessingResult(BaseModel):
    """Result of audio processing."""

    processed_data: bytes  # Processed audio data
    metrics: AudioMetrics  # Processing metrics
    format: AudioFormat  # Audio format
    sample_rate: int  # Sample rate
    channels: int  # Number of channels
    duration: float  # Duration in seconds
    channel_mapping: Dict[str, str] = field(
        default_factory=dict
    )  # Maps channels to sources
