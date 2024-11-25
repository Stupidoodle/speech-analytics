"""Transcription event and stream handlers."""

import asyncio
from dataclasses import asdict
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta

from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from .types import (
    TranscriptionResult,
    TranscriptionSegment,
    Word,
    ResultState,
    TranscriptionStreamResponse,
    SpeakerSegment,
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
        on_error: Optional[Callable] = None,
    ):
        super().__init__(output_stream)
        self.event_bus = event_bus
        self.store = store
        self.session_id = session_id
        self.on_result = on_result
        self.on_error = on_error

        # Track partial results by ResultId
        self._partial_results: Dict[str, Dict[str, Any]] = {}

        # Track last cleanup time
        self._last_cleanup = datetime.now()
        self._processed_events = 0
        self._cleanup_interval = 100  # Cleanup every 100 events

        # Configuration
        self.min_confidence_change = 0.15  # Significant confidence change threshold
        self.min_transcript_growth = 20  # Characters of growth to trigger event

    async def handle_transcript_event(self, transcript_event: TranscriptEvent) -> None:
        """Process transcript events from AWS Transcribe stream."""
        try:
            for result in transcript_event.transcript.results:
                if result.is_partial:
                    await self._handle_partial_result(result)
                else:
                    await self._handle_complete_result(result)

        except Exception as e:
            if self.on_error:
                await self._call_callback(
                    self.on_error,
                    TranscriptionStreamResponse(error={"message": str(e)}),
                )
            raise ResultError(f"Failed to handle transcript: {e}")

    async def _handle_partial_result(self, result) -> None:
        """Handle partial result updates."""
        result_id = result.result_id
        alternative = result.alternatives[0]

        # Process words with stability and speaker tracking
        words = self._process_words(alternative.items)

        # Calculate segment metrics
        avg_confidence = sum(w.confidence for w in words) / len(words) if words else 0
        speaker_segments = self._identify_speaker_segments(words)

        # Create new result state
        new_result = {
            "transcript": alternative.transcript,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "words": words,
            "is_partial": True,
            "speaker_segments": speaker_segments,
            "avg_confidence": avg_confidence,
            "timestamp": datetime.now(),
        }

        # Check for significant changes
        should_emit = self._is_significant_change(result_id, new_result)

        # Update tracking
        self._partial_results[result_id] = new_result

        # Store result
        self.store.add_partial_result(self.session_id, result_id, new_result)

        # Emit event if change is significant
        if should_emit:
            await self._emit_transcript_event("partial_update", new_result)

        # Periodic cleanup
        self._processed_events += 1
        await self._maybe_cleanup()

    async def _handle_complete_result(self, result) -> None:
        """Handle complete/final result."""
        result_id = result.result_id
        alternative = result.alternatives[0]

        # Process final words
        words = self._process_words(alternative.items, is_final=True)

        # Calculate final metrics
        avg_confidence = sum(w.confidence for w in words) / len(words) if words else 0
        speaker_segments = self._identify_speaker_segments(words)

        # Create final result
        final_result = TranscriptionResult(
            result_id=result_id,
            transcript=alternative.transcript,
            start_time=result.start_time,
            end_time=result.end_time,
            words=words,
            speaker_segments=speaker_segments,
            is_partial=False,  # Maybe change to get it from the result
            avg_confidence=avg_confidence,
            timestamp=datetime.now(),
        )

        # Clean up partial tracking
        self._partial_results.pop(result_id, None)

        # Store final result
        self.store.add_result(self.session_id, final_result)

        # Always emit complete results
        # noinspection PyTypeChecker
        await self._emit_transcript_event("complete", asdict(final_result))

        # If callback provided
        if self.on_result:
            await self._call_callback(
                self.on_result, TranscriptionStreamResponse(result=final_result)
            )

    @staticmethod
    def _process_words(items: List[Any], is_final: bool = False) -> List[Word]:
        """Process word items with stability tracking."""
        words = []
        for item in items:
            word = Word(
                item.content,
                item.start_time,
                item.end_time,
                item.confidence,
                getattr(item, "speaker_label", None),
                is_final or getattr(item, "stable", False),
            )
            words.append(word)
        return words

    @staticmethod
    def _identify_speaker_segments(words: List[Word]) -> List[SpeakerSegment]:
        """Identify speaker segments from word sequence."""
        segments = []
        current_segment = None
        word_confidence = []

        for word in words:
            word_confidence.append(word.confidence)

            speaker = word.speaker
            if not speaker:
                continue

            if not current_segment or current_segment.speaker != speaker:
                # New speaker segment
                if current_segment:
                    segments.append(current_segment)
                current_segment = SpeakerSegment(
                    speaker, word.start_time, word.end_time, word.content, 0
                )
            else:
                # Continue current segment
                current_segment.end_time = word.end_time
                current_segment.transcript += f" {word.content}"

        # Add final segment
        if current_segment:
            segments.append(current_segment)

        # Calculate average confidence for segments
        for segment in segments:
            segment.avg_confidence = sum(word_confidence) / len(word_confidence)

        return segments

    def _is_significant_change(
        self, result_id: str, new_result: Dict[str, Any]
    ) -> bool:
        """Determine if change is significant enough to emit event."""
        if result_id not in self._partial_results:
            return True

        old_result = self._partial_results[result_id]

        # Word count or stability changes
        if len(new_result["words"]) != len(old_result["words"]):
            return True

        old_stables = sum(1 for w in old_result["words"] if w["stable"])
        new_stables = sum(1 for w in new_result["words"] if w["stable"])
        if new_stables > old_stables:
            return True

        # Speaker changes
        old_segments = old_result["speaker_segments"]
        new_segments = new_result["speaker_segments"]
        if len(old_segments) != len(new_segments):
            return True

        # Significant confidence change
        if (
            abs(new_result["avg_confidence"] - old_result["avg_confidence"])
            > self.min_confidence_change
        ):
            return True

        # Significant transcript growth
        if (
            len(new_result["transcript"]) - len(old_result["transcript"])
            > self.min_transcript_growth
        ):
            return True

        return False

    async def _maybe_cleanup(self) -> None:
        """Perform periodic cleanup of old partial results."""
        if self._processed_events >= self._cleanup_interval:
            now = datetime.now()
            # Remove partial results older than 5 minutes
            cutoff = now - timedelta(minutes=5)

            self._partial_results = {
                k: v
                for k, v in self._partial_results.items()
                if v["timestamp"] > cutoff
            }

            self._processed_events = 0
            self._last_cleanup = now

    async def _emit_transcript_event(self, status: str, result: Dict[str, Any]) -> None:
        """Emit transcript event."""
        await self.event_bus.publish(
            Event(
                type=EventType.TRANSCRIPT,
                data={
                    "status": status,
                    "session_id": self.session_id,
                    "result": result,
                },
            )
        )

    async def _call_callback(self, callback: Callable, data: Any) -> None:
        """Safely execute callback function."""
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
                    ),
                )
