from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.events.bus import EventBus
from src.transcription.handlers import TranscriptionHandler
from src.transcription.models import TranscriptionSession, TranscriptionStore
from src.transcription.types import SpeakerSegment, TranscriptionResult, Word


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


@pytest.mark.asyncio
async def test_handle_partial_result(transcription_handler):
    # Mock a partial result
    mock_result = MagicMock()
    mock_result.result_id = "result_1"
    mock_result.alternatives = [
        MagicMock(items=[MagicMock(content="word1", confidence=0.9)])
    ]
    mock_result.start_time = 0.0
    mock_result.end_time = 1.0
    mock_result.is_partial = True

    await transcription_handler._handle_partial_result(mock_result)

    # Verify that the partial result is stored
    assert "result_1" in transcription_handler._partial_results
    # Verify that the event bus published an event
    transcription_handler.event_bus.publish.assert_called()


@pytest.mark.asyncio
async def test_handle_complete_result(transcription_handler):
    # Mock a complete result
    mock_result = MagicMock()
    mock_result.result_id = "result_1"
    mock_result.alternatives = [
        MagicMock(content="word1", confidence=0.9, transcript="word1")
    ]
    mock_result.start_time = 0.0
    mock_result.end_time = 1.0
    mock_result.is_partial = False

    await transcription_handler._handle_complete_result(mock_result)

    # Verify that the complete result is stored in the store
    session_results = transcription_handler.store.get_session_results("test_session")
    assert any(result["transcript"] == "word1" for result in session_results["results"])
    # Verify that the event bus published an event
    transcription_handler.event_bus.publish.assert_called()


class CorruptedTranscriptEvent:
    @property
    def transcript(self):
        raise RuntimeError("Simulated corruption in transcript")


@pytest.mark.asyncio
async def test_handle_error(transcription_handler):
    # Simulate an error during processing
    error_message = "Test Error"

    # Mock a method to raise an exception
    transcription_handler.some_method = AsyncMock(side_effect=Exception(error_message))

    transcript_event = CorruptedTranscriptEvent()

    # Verify that the exception is raised and handled
    with pytest.raises(
        Exception,
        match="Failed to handle transcript: Simulated corruption in transcript",
    ):
        await transcription_handler.handle_transcript_event(transcript_event)

    # Verify that the on_error callback is called
    transcription_handler.on_error.assert_called()


def test_create_session(transcription_store):
    # Create a new session
    session_id = "test_session"
    transcription_store.create_session(session_id, config={})

    # Verify the session exists
    assert session_id in transcription_store.sessions
    assert isinstance(transcription_store.sessions[session_id], TranscriptionSession)


def test_add_partial_result(transcription_store):
    # Create a new session
    session_id = "test_session"
    transcription_store.create_session(session_id, config={})
    partial_result = {"transcript": "Partial Result", "timestamp": datetime.now()}

    # Add a partial result
    transcription_store.add_partial_result(session_id, "result_1", partial_result)

    # Verify that the partial result is stored
    assert session_id in transcription_store.partial_results
    assert "result_1" in transcription_store.partial_results[session_id]
    assert (
        transcription_store.partial_results[session_id]["result_1"]["transcript"]
        == "Partial Result"
    )


def test_cleanup_session(transcription_store):
    # Create a session and add data
    session_id = "test_session"
    transcription_store.create_session(session_id, config={})
    transcription_store.add_partial_result(
        session_id, "result_1", {"transcript": "Partial"}
    )
    transcription_store.add_result(
        session_id,
        TranscriptionResult(
            session_id,
            "Final",
            0.0,
            1.0,
            [Word("Final", 0.0, 1.0, 0.9)],
            [SpeakerSegment("", 0.0, 1.0, "Final", 0.9)],
            False,
            0.9,
            datetime.now(),
        ),
    )

    # Cleanup the session
    transcription_store.cleanup_session(session_id)

    # Verify that the session data is removed
    assert session_id not in transcription_store.sessions
    assert session_id not in transcription_store.partial_results
    assert session_id not in transcription_store.results
