import pytest
from unittest.mock import AsyncMock
from datetime import datetime

from src.realtime.processor import RealtimeProcessor
from src.events.bus import EventBus, EventType, Event
from src.conversation.manager import ConversationManager
from src.transcription.aws_transcribe import TranscribeManager
from src.conversation.roles import Role


@pytest.fixture
async def mock_transcribe_manager():
    manager = AsyncMock(spec=TranscribeManager)
    manager.process_audio.return_value = "Test transcription"
    return manager


@pytest.fixture
async def mock_conversation_manager():
    manager = AsyncMock(spec=ConversationManager)
    manager.process_realtime.return_value = [
        "Test assistance response"
    ]
    return manager


@pytest.fixture
async def event_bus():
    return EventBus()


@pytest.fixture
async def processor(
    event_bus,
    mock_conversation_manager,
    mock_transcribe_manager
):
    return RealtimeProcessor(
        event_bus,
        mock_conversation_manager,
        mock_transcribe_manager,
        Role.INTERVIEWER
    )


@pytest.mark.asyncio
async def test_audio_chunk_processing(processor):
    """Test processing of audio chunks."""
    # Track published events
    received_events = []

    async def event_handler(event: Event):
        received_events.append(event)

    processor.event_bus.subscribe(
        EventType.TRANSCRIPTION,
        event_handler,
        {Role.INTERVIEWER}
    )

    # Start processor
    processor.is_running = True

    # Process audio chunk
    audio_chunk = b"test audio data"
    await processor._handle_audio_chunk(Event(
        type=EventType.AUDIO_CHUNK,
        data=audio_chunk,
        timestamp=datetime.now(),
        role=Role.INTERVIEWER
    ))

    # Verify transcription event was published
    assert len(received_events) == 1
    assert received_events[0].type == EventType.TRANSCRIPTION
    assert received_events[0].data == "Test transcription"
    assert received_events[0].role == Role.INTERVIEWER


@pytest.mark.asyncio
async def test_transcription_processing(processor):
    """Test processing of transcriptions."""
    received_events = []

    async def event_handler(event: Event):
        received_events.append(event)

    processor.event_bus.subscribe(
        EventType.ASSISTANCE,
        event_handler,
        {Role.INTERVIEWER}
    )

    processor.is_running = True

    # Process transcription
    await processor._handle_transcription(Event(
        type=EventType.TRANSCRIPTION,
        data="Test input",
        timestamp=datetime.now(),
        role=Role.INTERVIEWER
    ))

    assert len(received_events) == 1
    assert received_events[0].type == EventType.ASSISTANCE
    assert received_events[0].data == "Test assistance response"


@pytest.mark.asyncio
async def test_context_update_handling(processor):
    """Test handling of context updates."""
    received_events = []

    async def event_handler(event: Event):
        received_events.append(event)

    processor.event_bus.subscribe(
        EventType.ASSISTANCE,
        event_handler,
        {Role.INTERVIEWER}
    )

    processor.is_running = True

    # Process context update
    await processor._handle_context_update(Event(
        type=EventType.CONTEXT_UPDATE,
        data={"new_context": "test"},
        timestamp=datetime.now(),
        role=Role.INTERVIEWER
    ))

    assert len(received_events) == 1
    assert received_events[0].type == EventType.ASSISTANCE
    assert received_events[0].metadata["trigger"] == "context_update"
