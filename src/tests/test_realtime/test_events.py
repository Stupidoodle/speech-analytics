import pytest
from unittest.mock import Mock, AsyncMock
import asyncio
from datetime import datetime

from src.realtime.events import EventEmitter, EventType, Event


@pytest.fixture
def event_emitter():
    """Create a fresh event emitter for each test."""
    emitter = EventEmitter()
    yield emitter


@pytest.mark.asyncio
async def test_subscribe_and_emit(event_emitter):
    """Test basic subscription and event emission."""
    # Create mock callback
    callback = Mock()

    # Subscribe to event
    event_emitter.subscribe(EventType.TRANSCRIPT, callback)

    # Start event processing
    await event_emitter.start_processing()

    # Emit event
    test_event = Event(
        type=EventType.TRANSCRIPT,
        data={"text": "test"},
        timestamp=datetime.now().isoformat()
    )
    await event_emitter.emit(test_event)

    # Give time for event to be processed
    await asyncio.sleep(0.1)

    # Verify callback was called
    callback.assert_called_once_with(test_event)

    # Clean up
    await event_emitter.stop_processing()


@pytest.mark.asyncio
async def test_async_callback(event_emitter):
    """Test async callback handling."""
    # Create async mock callback
    callback = AsyncMock()

    event_emitter.subscribe(EventType.TRANSCRIPT, callback)
    await event_emitter.start_processing()

    test_event = Event(
        type=EventType.TRANSCRIPT,
        data={"text": "test"},
        timestamp=datetime.now().isoformat()
    )
    await event_emitter.emit(test_event)

    await asyncio.sleep(0.1)

    callback.assert_called_once_with(test_event)
    await event_emitter.stop_processing()


@pytest.mark.asyncio
async def test_multiple_subscribers(event_emitter):
    """Test multiple subscribers for same event."""
    callback1 = Mock()
    callback2 = Mock()

    event_emitter.subscribe(EventType.TRANSCRIPT, callback1)
    event_emitter.subscribe(EventType.TRANSCRIPT, callback2)
    await event_emitter.start_processing()

    test_event = Event(
        type=EventType.TRANSCRIPT,
        data={"text": "test"},
        timestamp=datetime.now().isoformat()
    )
    await event_emitter.emit(test_event)

    await asyncio.sleep(0.1)

    callback1.assert_called_once_with(test_event)
    callback2.assert_called_once_with(test_event)
    await event_emitter.stop_processing()


@pytest.mark.asyncio
async def test_unsubscribe(event_emitter):
    """Test unsubscribing from events."""
    callback = Mock()

    event_emitter.subscribe(EventType.TRANSCRIPT, callback)
    event_emitter.unsubscribe(EventType.TRANSCRIPT, callback)
    await event_emitter.start_processing()

    test_event = Event(
        type=EventType.TRANSCRIPT,
        data={"text": "test"},
        timestamp=datetime.now().isoformat()
    )
    await event_emitter.emit(test_event)

    await asyncio.sleep(0.1)

    callback.assert_not_called()
    await event_emitter.stop_processing()


@pytest.mark.asyncio
async def test_error_in_callback(event_emitter):
    """Test handling of errors in callbacks."""
    def failing_callback(event):
        raise Exception("Test error")

    # Add both failing and working callbacks
    event_emitter.subscribe(EventType.TRANSCRIPT, failing_callback)
    working_callback = Mock()
    event_emitter.subscribe(EventType.TRANSCRIPT, working_callback)

    await event_emitter.start_processing()

    test_event = Event(
        type=EventType.TRANSCRIPT,
        data={"text": "test"},
        timestamp=datetime.now().isoformat()
    )
    await event_emitter.emit(test_event)

    await asyncio.sleep(0.1)

    # Working callback should still be called
    working_callback.assert_called_once_with(test_event)
    await event_emitter.stop_processing()


@pytest.mark.asyncio
async def test_queue_clearing(event_emitter):
    """Test clearing of event queue on stop."""
    callback = Mock()
    event_emitter.subscribe(EventType.TRANSCRIPT, callback)
    await event_emitter.start_processing()

    # Add multiple events
    for i in range(5):
        await event_emitter.emit(Event(
            type=EventType.TRANSCRIPT,
            data={"text": f"test{i}"},
            timestamp=datetime.now().isoformat()
        ))

    # Stop immediately
    await event_emitter.stop_processing()

    # Queue should be empty
    assert event_emitter._queue.empty()


@pytest.mark.asyncio
async def test_multiple_event_types(event_emitter):
    """Test handling different event types."""
    transcript_callback = Mock()
    analysis_callback = Mock()

    event_emitter.subscribe(EventType.TRANSCRIPT, transcript_callback)
    event_emitter.subscribe(EventType.ANALYSIS, analysis_callback)
    await event_emitter.start_processing()

    # Emit different types of events
    transcript_event = Event(
        type=EventType.TRANSCRIPT,
        data={"text": "test"},
        timestamp=datetime.now().isoformat()
    )
    analysis_event = Event(
        type=EventType.ANALYSIS,
        data={"result": "test"},
        timestamp=datetime.now().isoformat()
    )

    await event_emitter.emit(transcript_event)
    await event_emitter.emit(analysis_event)

    await asyncio.sleep(0.1)

    # Verify correct callbacks were called
    transcript_callback.assert_called_once_with(transcript_event)
    analysis_callback.assert_called_once_with(analysis_event)

    await event_emitter.stop_processing()
