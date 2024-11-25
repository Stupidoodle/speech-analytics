from typing import Dict, Any, List
from dataclasses import dataclass, field
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
    metrics: Dict[str, Any] = field(
        default_factory=lambda: {
            "total_audio_time": 0.0,  # Duration of processed audio
            "processed_chunks": 0,  # Number of audio chunks processed
            "stable_segments": 0,  # Number of finalized segments
            "partial_updates": 0,  # Number of partial updates
            "avg_confidence": 0.0,  # Running average confidence
            "speaker_times": {},  # Speaking time per speaker
            "total_words": 0,  # Total word count
            "stable_words": 0,  # Count of stable words
        }
    )
    state: Dict[str, Any] = field(
        default_factory=lambda: {
            "last_sequence": 0,
            "current_speaker": None,
            "speakers": set(),
            "last_update": None,
        }
    )


class TranscriptionStore:
    """Manages transcription results and state."""

    def __init__(self, event_bus: EventBus):
        """Initialize transcription store."""
        self.event_bus = event_bus
        self.sessions: Dict[str, TranscriptionSession] = {}
        self.results: Dict[str, List[TranscriptionResult]] = {}
        self.partial_results: Dict[str, Dict[str, Any]] = {}
        self.speaker_profiles: Dict[str, Dict[str, Any]] = {}

    def create_session(self, session_id: str, config: Dict[str, Any]) -> None:
        """Create new transcription session."""
        self.sessions[session_id] = TranscriptionSession(
            session_id=session_id, start_time=datetime.now(), config=config
        )
        self.results[session_id] = []
        self.partial_results[session_id] = {}

    def add_result(self, session_id: str, result: TranscriptionResult) -> None:
        """Add complete transcription result."""
        if not result.is_partial:
            self.results[session_id].append(result)
            session = self.sessions[session_id]

            # Update metrics
            session.metrics["stable_segments"] += len(result.speaker_segments)
            session.metrics["total_words"] += len(result.words)
            session.metrics["stable_words"] += sum(1 for w in result.words if w.stable)

            # Update speaker profiles
            for segment in result.speaker_segments:
                speaker = segment.speaker
                if speaker not in self.speaker_profiles:
                    self.speaker_profiles[speaker] = {
                        "first_seen": datetime.now(),
                        "total_segments": 0,
                        "total_words": 0,
                        "total_duration": 0.0,
                        "average_confidence": 0.0,
                    }

                profile = self.speaker_profiles[speaker]
                profile["total_segments"] += 1
                profile["total_duration"] += segment.end_time - segment.start_time
                profile["average_confidence"] = (
                    profile["average_confidence"] * (profile["total_segments"] - 1)
                    + segment.avg_confidence
                ) / profile["total_segments"]

                # Update session speaker times
                session.metrics["speaker_times"][speaker] = session.metrics[
                    "speaker_times"
                ].get(speaker, 0.0) + (segment.end_time - segment.start_time)

            # Update state
            session.state["last_sequence"] += 1
            session.state["current_speaker"] = (
                result.speaker_segments[-1].speaker if result.speaker_segments else None
            )
            session.state["speakers"].update(s.speaker for s in result.speaker_segments)
            session.state["last_update"] = datetime.now()

    def add_partial_result(
        self, session_id: str, result_id: str, result: Dict[str, Any]
    ) -> None:
        """Add or update partial result."""
        self.partial_results[session_id][result_id] = result
        session = self.sessions[session_id]
        session.metrics["partial_updates"] += 1

    def get_session_results(
        self, session_id: str, include_partial: bool = False
    ) -> Dict[str, Any]:
        """Get results for session."""
        if session_id not in self.sessions:
            raise KeyError(f"Session not found: {session_id}")

        session = self.sessions[session_id]
        results = self.results[session_id]

        response = {
            "session_id": session_id,
            "duration": (datetime.now() - session.start_time).total_seconds(),
            "metrics": session.metrics,
            "results": [
                {
                    "transcript": result.transcript,
                    "speaker_segments": [
                        {
                            "speaker": segment.speaker,
                            "transcript": segment.transcript,
                            "start_time": segment.start_time,
                            "end_time": segment.end_time,
                            "avg_confidence": segment.avg_confidence,
                        }
                        for segment in result.speaker_segments
                    ],
                    "words": [
                        {
                            "content": word.content,
                            "speaker": word.speaker,
                            "confidence": word.confidence,
                            "start_time": word.start_time,
                            "end_time": word.end_time,
                            "stable": word.stable,
                        }
                        for word in result.words
                    ],
                    "timestamp": result.timestamp.isoformat(),
                }
                for result in results
            ],
        }

        if include_partial:
            response["partial"] = self.partial_results.get(session_id, {})

        if session.state["speakers"]:
            response["speakers"] = [
                {"id": speaker_id, **self.speaker_profiles[speaker_id]}
                for speaker_id in session.state["speakers"]
                if speaker_id in self.speaker_profiles
            ]

        return response

    def cleanup_session(self, session_id: str) -> None:
        """Clean up session data."""
        self.sessions.pop(session_id, None)
        self.results.pop(session_id, None)
        self.partial_results.pop(session_id, None)
