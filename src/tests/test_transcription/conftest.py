"""Test configuration for transcription tests."""
import pytest
from unittest.mock import AsyncMock, Mock

from src.transcription.types import TranscriptionConfig
from src.transcription.models import TranscriptionStore


@pytest.fixture
def transcription_config():
    """Create test transcription configuration."""
    return TranscriptionConfig(
        language_code="en-US",
        sample_rate_hz=16000,
        media_encoding="pcm",
        enable_speaker_separation=True,
        enable_channel_identification=True,
        number_of_channels=2
    )


@pytest.fixture
def mock_transcript_event():
    """Create mock transcript event."""
    class MockTranscriptEvent:
        def __init__(self):
            self.transcript = Mock()
            self.transcript.results = []
            self.request_id = "test-request"
            self.language_code = "en-US"
            self.media_sample_rate_hz = 16000
            self.media_encoding = "pcm"
            self.vocabulary_name = None
            self.vocabulary_filter_name = None

    return MockTranscriptEvent()


@pytest.fixture
def mock_stream():
    """Create mock AWS stream."""
    stream = AsyncMock()
    stream.input_stream = AsyncMock()
    stream.output_stream = AsyncMock()
    return stream


@pytest.fixture
def mock_transcribe_client(mock_stream):
    """Create mock Transcribe client."""
    client = AsyncMock()
    client.start_stream_transcription.return_value = mock_stream
    return client


@pytest.fixture
def transcription_store():
    """Create transcription store."""
    return TranscriptionStore()


@pytest.fixture
def sample_transcript_result():
    """Create sample transcript result."""
    return {
        "Transcript": {
            "Results": [
                {
                    "Alternatives": [
                        {
                            "Items": [
                                {
                                    "Content": "Hello",
                                    "EndTime": 0.5,
                                    "StartTime": 0.0,
                                    "Type": "pronunciation",
                                    "Confidence": 0.95,
                                    "Speaker": "spk_0"
                                },
                                {
                                    "Content": "world",
                                    "EndTime": 1.0,
                                    "StartTime": 0.6,
                                    "Type": "pronunciation",
                                    "Confidence": 0.98,
                                    "Speaker": "spk_0"
                                }
                            ],
                            "Transcript": "Hello world"
                        }
                    ],
                    "EndTime": 1.0,
                    "IsPartial": False,
                    "ResultId": "test-result",
                    "StartTime": 0.0
                }
            ]
        }
    }
