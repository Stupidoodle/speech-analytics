from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


class TranscriptionState(Enum):
    """States of transcription process."""
    IDLE = "idle"
    STARTING = "starting"
    STREAMING = "streaming"
    STOPPING = "stopping"
    ERROR = "error"


class ResultState(Enum):
    """States of transcription results."""
    PARTIAL = "partial"
    STABLE = "stable"


@dataclass
class TranscriptionConfig:
    """Configuration for transcription service."""
    language_code: str = "en-US"
    sample_rate_hz: int = 16000
    media_encoding: str = "pcm"
    enable_partial_results: bool = True
    max_delay_ms: int = 100  # Max delay for partial results
    vocabulary_name: Optional[str] = None
    vocabulary_filter_name: Optional[str] = None
    enable_speaker_separation: bool = True
    enable_channel_identification: bool = True
    number_of_channels: int = 2
    channel_identification_mode: str = "speakers"  # or "channels"


@dataclass
class Word:
    """Word-level transcription data."""
    content: str
    confidence: float
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    speaker_confidence: Optional[float] = None
    stable: bool = False


@dataclass
class TranscriptionSegment:
    """A segment of transcription."""
    text: str
    words: List[Word]
    speaker: Optional[str]
    channel: Optional[str]
    start_time: float
    end_time: float
    confidence: float
    state: ResultState
    timestamp: datetime


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    segments: List[TranscriptionSegment]
    is_partial: bool
    session_id: str
    request_id: str
    language_code: str
    media_sample_rate_hz: int
    media_encoding: str
    vocabulary_name: Optional[str]
    vocabulary_filter_name: Optional[str]
    timestamp: datetime


@dataclass
class TranscriptionStreamResponse:
    """Response from transcription stream."""
    result: Optional[TranscriptionResult] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
