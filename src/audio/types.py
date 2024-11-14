"""Type definitions for audio processing."""
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
    INPUT = "input"
    OUTPUT = "output"
    LOOPBACK = "loopback"


class ProcessingMode(str, Enum):
    """Audio processing modes."""
    REALTIME = "realtime"
    BATCH = "batch"
    ADAPTIVE = "adaptive"


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 44100
    channels: int = 2
    format: AudioFormat = AudioFormat.PCM
    chunk_size: int = 1024
    device_id: Optional[int] = None
    processing_mode: ProcessingMode = ProcessingMode.REALTIME
    enable_noise_reduction: bool = True
    enable_auto_gain: bool = True
    device_type: Optional[DeviceType] = None


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
    levels: Dict[str, float]
    latencies: Dict[str, float]
    active_channels: Set[str]
    metrics: BufferMetrics


@dataclass
class AudioMetrics:
    """Audio processing metrics."""
    peak_level: float = 0.0
    rms_level: float = 0.0
    noise_level: float = 0.0
    clipping_count: int = 0
    dropout_count: int = 0
    processing_time: float = 0.0
    buffer_stats: Dict[str, float] = field(default_factory=dict)


class ProcessingResult(BaseModel):
    """Result of audio processing."""
    processed_data: bytes
    metrics: AudioMetrics
    format: AudioFormat
    sample_rate: int
    channels: int
    duration: float