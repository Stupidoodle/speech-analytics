from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from .types import (
    TranscriptionResult,
)

from src.events.types import Event, EventType
from src.events.bus import EventBus
import asyncio

@dataclass
class TranscriptionSession:
    """Active transcription session."""
    session_id: str
    start_time: datetime
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    state: Dict[str, Any]


class TranscriptionStore:
    """Manages transcription results and state."""

    def __init__(self, event_bus: EventBus):
        """Initialize transcription store."""
        self.event_bus = event_bus
        self.sessions: Dict[str, TranscriptionSession] = {}
        self.results: Dict[str, List[TranscriptionResult]] = {}
        self.partial_results: Dict[str, Dict[str, Any]] = {}
        self.speaker_profiles: Dict[str, Dict[str, Any]] = {}

    def create_session(
            self,
            session_id: str,
            config: Dict[str, Any]
    ) -> None:
        """Create new transcription session."""
        self.sessions[session_id] = TranscriptionSession(
            session_id=session_id,
            start_time=datetime.now(),
            config=config,
            metrics={
                "total_audio_time": 0.0,
                "processed_chunks": 0,
                "stable_segments": 0,
                "partial_updates": 0
            },
            state={
                "last_sequence": 0,
                "current_speaker": None,
                "speakers": set()
            }
        )
        self.results[session_id] = []
        self.partial_results[session_id] = {}

    def add_result(
            self,
            session_id: str,
            result: TranscriptionResult
    ) -> None:
        """Add transcription result."""
        if not result.is_partial:
            self.results[session_id].append(result)
            asyncio.get_event_loop().run_until_complete(self.event_bus.publish(Event(
                type=EventType.TRANSCRIPT,
                data={
                    "session_id": session_id,
                    "result_id": id(result)
                }
            )))
            session = self.sessions[session_id]
            session.metrics["stable_segments"] += len(result.segments)

            # Update speaker profiles
            for segment in result.segments:
                if segment.speaker:
                    if segment.speaker not in self.speaker_profiles:
                        self.speaker_profiles[segment.speaker] = {
                            "first_seen": datetime.now(),
                            "total_segments": 0,
                            "total_words": 0,
                            "total_duration": 0.0,
                            "average_confidence": 0.0
                        }

                    profile = self.speaker_profiles[segment.speaker]
                    profile["total_segments"] += 1
                    profile["total_words"] += len(segment.words)
                    profile["total_duration"] += (
                            segment.end_time - segment.start_time
                    )
                    profile["average_confidence"] = (
                            (profile["average_confidence"] *
                             (profile["total_segments"] - 1) +
                             segment.confidence) / profile["total_segments"]
                    )
        else:
            self.partial_results[session_id] = {
                "segments": [
                    {
                        "text": s.text,
                        "speaker": s.speaker,
                        "channel": s.channel,
                        "confidence": s.confidence
                    }
                    for s in result.segments
                ]
            }
            self.sessions[session_id].metrics["partial_updates"] += 1

    def get_session_results(
            self,
            session_id: str,
            include_partial: bool = False
    ) -> Dict[str, Any]:
        """Get results for session."""
        if session_id not in self.sessions:
            raise KeyError(f"Session not found: {session_id}")

        session = self.sessions[session_id]
        results = self.results[session_id]

        response = {
            "session_id": session_id,
            "duration": (
                    datetime.now() - session.start_time
            ).total_seconds(),
            "metrics": session.metrics,
            "results": [
                {
                    "segments": [
                        {
                            "text": segment.text,
                            "speaker": segment.speaker,
                            "channel": segment.channel,
                            "start_time": segment.start_time,
                            "end_time": segment.end_time,
                            "confidence": segment.confidence,
                            "words": [
                                {
                                    "content": w.content,
                                    "confidence": w.confidence,
                                    "start_time": w.start_time,
                                    "end_time": w.end_time,
                                    "speaker": w.speaker,
                                    "stable": w.stable
                                }
                                for w in segment.words
                            ]
                        }
                        for segment in result.segments
                    ],
                    "timestamp": result.timestamp.isoformat()
                }
                for result in results
            ]
        }

        if include_partial:
            response["partial"] = self.partial_results.get(
                session_id,
                {}
            )

        if session.state["speakers"]:
            response["speakers"] = [
                {
                    "id": speaker_id,
                    **self.speaker_profiles[speaker_id]
                }
                for speaker_id in session.state["speakers"]
                if speaker_id in self.speaker_profiles
            ]

        return response

    def cleanup_session(self, session_id: str) -> None:
        """Clean up session data."""
        self.sessions.pop(session_id, None)
        self.results.pop(session_id, None)
        self.partial_results.pop(session_id, None)
