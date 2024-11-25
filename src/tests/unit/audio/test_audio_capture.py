import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.audio.capture import AudioCapture
from src.audio.types import AudioConfig
from src.events.bus import EventBus


class AsyncStreamMock:
    """Mock class for async streaming"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        if isinstance(chunk, Exception):
            raise chunk
        return chunk


@pytest.mark.asyncio
async def test_chunk_processing():
    """Test processing audio chunks in AudioCapture."""
    event_bus = EventBus()
    config = AudioConfig(sample_rate=16000, channels=2)
    capture = AudioCapture(event_bus, config)

    # Mock dependencies
    capture.device_manager.get_default_stereo_devices = AsyncMock(
        return_value={"mic": MagicMock(id=1), "desktop": MagicMock(id=2)}
    )

    # Create test data
    mic_chunks = [b"\x00\x01" * 100]
    desktop_chunks = [b"\x00\x02" * 100]

    mic_stream = AsyncStreamMock(mic_chunks)
    desktop_stream = AsyncStreamMock(desktop_chunks)

    # Mock the open_stream method to return our stream mocks directly
    capture.device_manager.open_stream = MagicMock(
        side_effect=lambda device, _: mic_stream if device.id == 1 else desktop_stream
    )

    capture.processor.process_chunk = AsyncMock(
        side_effect=lambda x: MagicMock(processed_data=x)
    )
    capture.mixer.mix_streams = AsyncMock(
        side_effect=lambda x, y: MagicMock(processed_data=x + y)
    )
    capture.buffer.write = AsyncMock()

    await capture.start_capture()

    # Allow processing loop to run
    await asyncio.sleep(0.1)
    await capture.stop_capture()

    # Validate stats
    assert (
        capture._stats["chunks_processed"] > 0
    ), f"Chunks processed: {capture._stats['chunks_processed']}"
    assert (
        capture._stats["bytes_processed"] > 0
    ), f"Bytes processed: {capture._stats['bytes_processed']}"

    # Verify mocks were called
    capture.processor.process_chunk.assert_called()
    capture.mixer.mix_streams.assert_called()
    capture.buffer.write.assert_called()


@pytest.mark.asyncio
async def test_stream_interruption():
    """Test handling of stream interruptions during processing."""
    event_bus = EventBus()
    config = AudioConfig(sample_rate=16000, channels=2)
    capture = AudioCapture(event_bus, config)

    # Mock dependencies
    capture.device_manager.get_default_stereo_devices = AsyncMock(
        return_value={"mic": MagicMock(id=1), "desktop": MagicMock(id=2)}
    )

    # Create test data with interruption
    mic_chunks = [b"\x00\x01" * 100, Exception("Stream error")]
    desktop_chunks = [b"\x00\x02" * 100]

    mic_stream = AsyncStreamMock(mic_chunks)
    desktop_stream = AsyncStreamMock(desktop_chunks)

    # Mock the open_stream method to return our stream mocks directly
    capture.device_manager.open_stream = MagicMock(
        side_effect=lambda device, _: mic_stream if device.id == 1 else desktop_stream
    )

    capture.processor.process_chunk = AsyncMock(
        side_effect=lambda x: MagicMock(processed_data=x)
    )
    capture.mixer.mix_streams = AsyncMock(
        side_effect=lambda x, y: MagicMock(processed_data=x + y if y else x)
    )
    capture.buffer.write = AsyncMock()

    await capture.start_capture()

    # Allow processing loop to run
    await asyncio.sleep(0.1)
    await capture.stop_capture()

    # Validate stats
    assert (
        capture._stats["chunks_processed"] > 0
    ), f"Chunks processed: {capture._stats['chunks_processed']}"


@pytest.mark.asyncio
async def test_device_initialization_failure():
    """Test handling of device initialization failure."""
    event_bus = EventBus()
    config = AudioConfig(sample_rate=16000, channels=2)
    capture = AudioCapture(event_bus, config)

    # Mock device initialization failure
    capture.device_manager.get_default_stereo_devices = AsyncMock(
        side_effect=Exception("Device error")
    )

    with pytest.raises(
        Exception, match="Failed to start capture: Failed to initialize devices"
    ):
        await capture.start_capture()
