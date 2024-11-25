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
    start_time: float
    end_time: float
    confidence: float
    speaker: Optional[str] = None
    stable: bool = False


@dataclass
class SpeakerSegment:
    """Speaker-specific segment of transcription."""

    speaker: str
    start_time: float
    end_time: float
    transcript: str
    avg_confidence: float


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

    result_id: str
    transcript: str
    start_time: float
    end_time: float
    words: List[Word]
    speaker_segments: List[SpeakerSegment]
    is_partial: bool
    avg_confidence: float
    timestamp: datetime


@dataclass
class TranscriptionStreamResponse:
    """Response from transcription stream."""

    result: Optional[TranscriptionResult] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
