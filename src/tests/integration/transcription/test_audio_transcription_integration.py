import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.transcription.handlers import TranscriptionHandler
from src.transcription.models import TranscriptionStore
from src.audio.capture import AudioCapture
from src.audio.buffer import AudioBuffer
from src.audio.processor import AudioProcessor
from src.events.bus import EventBus
from src.events.types import EventType
from src.transcription.types import TranscriptionResult, Word, SpeakerSegment
from src.audio.types import (
    AudioConfig,
    DeviceType,
    DeviceInfo,
    ProcessingResult,
    AudioFormat,
    AudioMetrics,
)
from datetime import datetime


@pytest.fixture
def mock_event_bus():
    return AsyncMock(spec=EventBus)


@pytest.fixture
def transcription_store(mock_event_bus):
    transcription_store = TranscriptionStore(event_bus=mock_event_bus)
    transcription_store.create_session("test_session", config={})
    return transcription_store


@pytest.fixture
def transcription_handler(mock_event_bus, transcription_store):
    return TranscriptionHandler(
        event_bus=mock_event_bus,
        output_stream=None,
        store=transcription_store,
        session_id="test_session",
        on_result=AsyncMock(),
        on_error=AsyncMock(),
    )


@pytest.fixture
def audio_config():
    return AudioConfig(sample_rate=16000, channels=2)


@pytest.fixture
def audio_buffer(mock_event_bus, audio_config):
    return AudioBuffer(
        event_bus=mock_event_bus,
        config=audio_config,
    )


@pytest.fixture
def mock_device_manager():
    mock_manager = AsyncMock()
    mock_manager.get_default_stereo_devices = AsyncMock(
        return_value={
            "mic": DeviceInfo(
                id=1,
                name="Test Mic",
                type=DeviceType.INPUT,
                channels=2,
                sample_rate=16000,
                is_default=True,
                is_loopback=False,
                supports_input=True,
                supports_output=False,
            ),
            "desktop": DeviceInfo(
                id=2,
                name="Test Desktop",
                type=DeviceType.OUTPUT,
                channels=2,
                sample_rate=16000,
                is_default=True,
                is_loopback=True,
                supports_input=False,
                supports_output=True,
            ),
        }
    )
    return mock_manager


@pytest.fixture
def audio_processor(mock_event_bus, audio_config):
    processor = AudioProcessor(mock_event_bus, audio_config)
    # Create proper AudioMetrics instance
    metrics = AudioMetrics(
        peak_level=0.8,
        rms_level=0.5,
        noise_level=0.1,
        clipping_count=0,
        dropout_count=0,
        processing_time=0.001,
        buffer_stats={"running_max": 0.8},
    )

    processor.process_chunk = AsyncMock(
        return_value=ProcessingResult(
            processed_data=b"\x00\x01" * 400,
            metrics=metrics,
            format=AudioFormat.PCM,
            sample_rate=16000,
            channels=2,
            duration=0.1,
        )
    )
    return processor


@pytest.fixture
def audio_capture(
    mock_event_bus, audio_config, audio_buffer, audio_processor, mock_device_manager
):
    capture = AudioCapture(mock_event_bus, audio_config)
    capture.device_manager = mock_device_manager
    capture.processor = audio_processor
    capture.buffer = audio_buffer
    return capture


@pytest.mark.asyncio
async def test_integration_audio_to_transcription(audio_capture, transcription_handler):
    # Create mock audio chunk generator
    async def mock_chunk_generator():
        for _ in range(5):  # Generate 5 chunks
            yield b"\x00\x01" * 400
            await asyncio.sleep(0.01)

    # Mock the device streams
    mic_stream = mock_chunk_generator()
    desktop_stream = mock_chunk_generator()

    # Set up device stream mocking
    audio_capture.device_manager.open_stream = MagicMock(
        side_effect=lambda device, _: mic_stream if device.id == 1 else desktop_stream
    )

    try:
        # Start audio capture
        await audio_capture.start_capture()

        # Let the capture run briefly
        await asyncio.sleep(0.2)

        # Verify that audio data is processed and written to buffer
        assert (
            len(audio_capture.buffer._buffers["combined"]) > 0
        ), "Buffer should contain processed audio data"

        # Verify events were published
        assert (
            audio_capture.event_bus.publish.called
        ), "Event bus should have published events"

        # Verify that at least one AUDIO_CHUNK event was published
        audio_chunk_events = [
            call
            for call in audio_capture.event_bus.publish.call_args_list
            if call[0][0].type == EventType.AUDIO_CHUNK
        ]
        assert len(audio_chunk_events) > 0, "Should have published AUDIO_CHUNK events"

    finally:
        # Ensure we stop capture even if test fails
        await audio_capture.stop_capture()
