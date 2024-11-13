"""Transcription event and stream handlers."""
from typing import Dict, Any, Optional, Callable, List
import asyncio
from datetime import datetime

from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from .types import (
    TranscriptionResult,
    TranscriptionSegment,
    Word,
    ResultState,
    TranscriptionStreamResponse
)
from .models import TranscriptionStore
from .exceptions import ResultError
from src.events.types import Event, EventType
from src.events.bus import EventBus


class TranscriptionHandler(TranscriptResultStreamHandler):
    """Enhanced transcription handler with speaker separation."""

    def __init__(
            self,
            event_bus: EventBus,
            output_stream,
            store: TranscriptionStore,
            session_id: str,
            on_result: Optional[Callable] = None,
            on_error: Optional[Callable] = None
    ):
        """Initialize transcription handler.

        Args:
            output_stream: AWS Transcribe output stream
            store: Transcription store for results
            session_id: Active session ID
            on_result: Optional result callback
            on_error: Optional error callback
        """
        super().__init__(output_stream)
        self.event_bus = event_bus
        self.store = store
        self.session_id = session_id
        self.on_result = on_result
        self.on_error = on_error
        self._partial_buffer: Dict[str, Any] = {}

    async def handle_transcript_event(
            self,
            transcript_event: TranscriptEvent
    ) -> None:
        """Process transcript events.

        Args:
            transcript_event: Transcript event from AWS
        """
        try:
            # Process each result in the transcript
            for aws_result in transcript_event.transcript.results:
                for alternative in aws_result.alternatives:
                    segments = await self._process_alternative(
                        alternative,
                        aws_result.is_partial
                    )

                    # Create result object
                    result = TranscriptionResult(
                        segments=segments,
                        is_partial=aws_result.is_partial,
                        session_id=self.session_id,
                        request_id=transcript_event.request_id,
                        language_code=transcript_event.language_code,
                        media_sample_rate_hz=transcript_event.media_sample_rate_hz,
                        media_encoding=transcript_event.media_encoding,
                        vocabulary_name=transcript_event.vocabulary_name,
                        vocabulary_filter_name=transcript_event.vocabulary_filter_name,
                        timestamp=datetime.now()
                    )
                    # TODO: Change the result to actual dict
                    await self.event_bus.publish(Event(
                        type=EventType.TRANSCRIPT,
                        data={
                            "status": "transcript_received",
                            "result": result.__dict__
                        }
                    ))

                    # Store result
                    self.store.add_result(self.session_id, result)

                    # Callback if provided
                    if self.on_result:
                        await self._call_callback(
                            self.on_result,
                            TranscriptionStreamResponse(result=result)
                        )

        except Exception as e:
            if self.on_error:
                await self._call_callback(
                    self.on_error,
                    TranscriptionStreamResponse(
                        error={"message": str(e)}
                    )
                )
            raise ResultError(f"Failed to handle transcript: {e}")

    async def _process_alternative(
            self,
            alternative,
            is_partial: bool
    ) -> List[TranscriptionSegment]:
        """Process transcription alternative.

        Args:
            alternative: AWS transcription alternative
            is_partial: Whether this is a partial result

        Returns:
            List of processed segments
        """
        segments = []
        current_segment = None
        current_words = []

        for item in alternative.items:
            # Create word object
            word = Word(
                content=item.content,
                confidence=item.confidence,
                start_time=item.start_time,
                end_time=item.end_time,
                speaker=item.speaker_label if hasattr(item, 'speaker_label') else None,
                speaker_confidence=(
                    item.speaker_confidence
                    if hasattr(item, 'speaker_confidence')
                    else None
                ),
                stable=item.stable if hasattr(item, 'stable') else True
            )

            # Handle speaker changes or start of new segment
            if not current_segment or (
                    word.speaker and word.speaker != current_segment.speaker
            ):
                if current_segment and current_words:
                    # Finalize current segment
                    current_segment.words = current_words
                    current_segment.text = " ".join(w.content for w in current_words)
                    segments.append(current_segment)

                # Start new segment
                current_segment = TranscriptionSegment(
                    text="",
                    words=[],
                    speaker=word.speaker,
                    channel=None,  # Will be set if channel identification is enabled
                    start_time=word.start_time,
                    end_time=word.end_time,
                    confidence=word.confidence,
                    state=ResultState.PARTIAL if is_partial else ResultState.STABLE,
                    timestamp=datetime.now()
                )
                current_words = []

            current_words.append(word)
            current_segment.end_time = word.end_time
            current_segment.confidence = sum(
                w.confidence for w in current_words
            ) / len(current_words)

        # Add final segment
        if current_segment and current_words:
            current_segment.words = current_words
            current_segment.text = " ".join(w.content for w in current_words)
            segments.append(current_segment)

        return segments

    async def _call_callback(
            self,
            callback: Callable,
            data: Any
    ) -> None:
        """Safely execute callback function.

        Args:
            callback: Callback function
            data: Data to pass to callback
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            if self.on_error:
                await self._call_callback(
                    self.on_error,
                    TranscriptionStreamResponse(
                        error={"message": f"Callback error: {e}"}
                    )
                )
