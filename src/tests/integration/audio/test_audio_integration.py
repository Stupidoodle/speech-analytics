import asyncio
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.audio.capture import AudioCapture
from src.audio.types import AudioConfig, DeviceInfo, DeviceType
from src.events.bus import EventBus
from src.events.types import Event, EventType


class EventCollector:
    """Collects events for testing verification"""

    def __init__(self):
        self.events: List[Event] = []

    async def collect_event(self, event: Event) -> None:
        self.events.append(event)


class MockAudioStream:
    """Simulates an audio stream with configurable behavior"""

    def __init__(self, chunks: List[Any], delay: float = 0.01):
        self.chunks = chunks
        self.delay = delay
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        if isinstance(chunk, Exception):
            raise chunk  # Raise the exception to simulate an error
        await asyncio.sleep(self.delay)  # Simulate real-time audio
        return chunk


@pytest.fixture
async def event_bus_task():
    """Fixture for event bus task management"""
    event_bus = EventBus()  # Create the EventBus instance
    task = asyncio.create_task(event_bus.start())  # Start the EventBus task
    try:
        yield event_bus  # Yield the EventBus instance
    finally:
        task.cancel()  # Cancel the EventBus task
        try:
            await task  # Ensure proper cleanup
        except asyncio.CancelledError:
            pass


@pytest.fixture
async def test_streams():
    """Fixture for creating test audio streams"""
    # Create test audio data
    duration_sec = 0.5  # Short test duration
    sample_rate = 16000
    num_samples = int(duration_sec * sample_rate)

    # Calculate chunk size based on 20ms chunks
    chunk_size = int(sample_rate * 0.02) * 4  # 4 bytes per stereo sample
    num_chunks = num_samples // (chunk_size // 4)

    # Create sine waves for mic and desktop audio
    t = np.linspace(0, duration_sec, num_samples)
    mic_signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    desktop_signal = np.sin(2 * np.pi * 880 * t)  # 880 Hz tone

    # Convert to 16-bit PCM and split into chunks
    mic_chunks = []
    desktop_chunks = []

    for i in range(num_chunks):
        start = i * (chunk_size // 4)
        end = start + (chunk_size // 4)

        mic_chunk = (mic_signal[start:end] * 32767).astype(np.int16).tobytes()
        desktop_chunk = (desktop_signal[start:end] * 32767).astype(np.int16).tobytes()

        mic_chunks.append(mic_chunk)
        desktop_chunks.append(desktop_chunk)

    return mic_chunks, desktop_chunks


@pytest.fixture
async def mock_capture(event_bus_task, test_streams):
    """Fixture for setting up the capture system with mocks"""
    async for event_bus in event_bus_task:
        collector = EventCollector()

        # Subscribe collector to all event types
        for event_type in EventType:
            event_bus.subscribe(event_type, collector.collect_event)

        # Create audio config
        config = AudioConfig(sample_rate=16000, channels=2, chunk_duration_ms=20)

        # Initialize capture system
        capture = AudioCapture(event_bus, config)

        # Await test streams to generate chunks
        mic_chunks, desktop_chunks = await test_streams

        # Create mock streams
        mic_stream = MockAudioStream(mic_chunks)
        desktop_stream = MockAudioStream(desktop_chunks)

        # Mock device manager
        capture.device_manager.get_default_stereo_devices = AsyncMock(
            return_value={
                "mic": DeviceInfo(
                    id=1,
                    name="Test Mic",
                    type=DeviceType.INPUT,
                    channels=2,
                    sample_rate=16000,
                ),
                "desktop": DeviceInfo(
                    id=2,
                    name="Test Desktop",
                    type=DeviceType.OUTPUT,
                    channels=2,
                    sample_rate=16000,
                ),
            }
        )

        capture.device_manager.open_stream = MagicMock(
            side_effect=lambda device, _: (
                mic_stream if device.id == 1 else desktop_stream
            )
        )

        # Yield capture and collector
        yield capture, collector


@pytest.mark.asyncio
async def test_audio_capture_pipeline(mock_capture):
    """Test the complete audio capture pipeline"""
    async for capture, collector in mock_capture:  # Proper async generator consumption
        # Start capture
        await capture.start_capture()

        # Let it run for a short time
        await asyncio.sleep(1.0)

        # Stop capture
        await capture.stop_capture()

        # Give some time for the event bus to process the events
        await asyncio.sleep(0.5)

        # Ensure all events have been processed by the event bus
        await capture.event_bus._queue.join()  # Wait until the queue is empty

        # Verify events
        events_by_type = {}
        for event in collector.events:
            if event.type not in events_by_type:
                events_by_type[event.type] = []
            events_by_type[event.type].append(event)

        # Assertions for events
        assert EventType.AUDIO_CHUNK in events_by_type
        assert (
            len(
                [
                    e
                    for e in collector.events
                    if e.data.get("status") == "capture_started"
                ]
            )
            == 1
        )
        assert (
            len(
                [
                    e
                    for e in collector.events
                    if e.data.get("status") == "capture_stopped"
                ]
            )
            == 1
        )


@pytest.mark.asyncio
async def test_error_handling(mock_capture):
    """Test error handling in the capture pipeline"""
    async for capture, collector in mock_capture:

        # Mock device error
        capture.device_manager.get_default_stereo_devices = AsyncMock(
            side_effect=Exception("Device initialization error")
        )

        # Attempt to start capture
        with pytest.raises(Exception):
            await capture.start_capture()

        # Ensure all events have been processed by the event bus
        await capture.event_bus._queue.join()  # Wait until the queue is empty

        # Verify error events
        error_events = [
            e for e in collector.events if e.data.get("status") == "device_error"
        ]
        assert len(error_events) > 0


@pytest.mark.asyncio
async def test_stream_interruption(mock_capture):
    """Test handling of stream interruptions"""
    async for capture, collector in mock_capture:

        # Create streams with interruption
        mic_chunks = [b"\x00\x01" * 100, Exception("Stream error")]
        desktop_chunks = [b"\x00\x02" * 100]

        mic_stream = MockAudioStream(mic_chunks)
        desktop_stream = MockAudioStream(desktop_chunks)

        # Update mock streams
        capture.device_manager.open_stream = MagicMock(
            side_effect=lambda device, _: (
                mic_stream if device.id == 1 else desktop_stream
            )
        )

        # Start capture
        await capture.start_capture()

        # Let it run until interruption
        await asyncio.sleep(0.5)

        # Stop capture
        await capture.stop_capture()

        # Ensure all events have been processed by the event bus
        await capture.event_bus._queue.join()  # Wait until the queue is empty

        # Verify error handling
        error_events = [
            e for e in collector.events if e.data.get("status") == "capture_error"
        ]
        assert len(error_events) > 0
