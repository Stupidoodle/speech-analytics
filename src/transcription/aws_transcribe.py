"""AWS Transcribe service integration."""
from typing import Dict, Any, Optional
import asyncio
import uuid

from amazon_transcribe.client import TranscribeStreamingClient

from .types import (
    TranscriptionConfig,
    TranscriptionState,
)
from .models import TranscriptionStore
from .handlers import TranscriptionHandler
from .exceptions import (
    StreamingError,
    ConfigurationError,
    ConnectionError
)
from src.events.bus import EventBus
from src.events.types import Event, EventType


class TranscribeManager:
    """Manages AWS Transcribe streaming sessions."""

    def __init__(
            self,
            event_bus: EventBus,
            region: str,
            config: Optional[TranscriptionConfig] = None
    ):
        """Initialize transcribe manager.

        Args:
            region: AWS region
            config: Optional transcription configuration
        """
        self.event_bus = event_bus
        self.region = region
        self.config = config or TranscriptionConfig()
        self.client: Optional[TranscribeStreamingClient] = None
        self.store = TranscriptionStore(event_bus)
        self.current_stream = None
        self.current_handler = None
        self.state = TranscriptionState.IDLE
        self._error = None

    async def __aenter__(self):
        """Enter async context."""
        try:
            self.client = TranscribeStreamingClient(region=self.region)
            return self
        except Exception as e:
            raise ConnectionError(f"Failed to create client: {e}")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self.state != TranscriptionState.IDLE:
            await self.stop_stream()
        self.client = None

    async def start_stream(
            self,
            session_id: Optional[str] = None,
            on_result: Optional[callable] = None,
            on_error: Optional[callable] = None
    ) -> str:
        """Start transcription stream.

        Args:
            session_id: Optional session ID
            on_result: Optional result callback
            on_error: Optional error callback

        Returns:
            Session ID

        Raises:
            StreamingError: If stream fails to start
            ConfigurationError: If configuration is invalid
        """
        try:
            if not self.client:
                raise ConfigurationError("Client not initialized")

            if self.state != TranscriptionState.IDLE:
                raise StreamingError("Stream already running")

            self.state = TranscriptionState.STARTING
            await self.event_bus.publish(Event(
                type=EventType.TRANSCRIPT,
                data={
                    "status": "stream_started",
                    "session_id": session_id
                }
            ))
            session_id = session_id or str(uuid.uuid4())

            # Create stream configuration
            stream_config = {
                "language_code": self.config.language_code,
                "media_sample_rate_hz": self.config.sample_rate_hz,
                "media_encoding": self.config.media_encoding,
                "vocabulary_name": self.config.vocabulary_name,
                "vocabulary_filter_name": self.config.vocabulary_filter_name,
                "show_speaker_label": self.config.enable_speaker_separation,
                "enable_channel_identification": (
                    self.config.enable_channel_identification
                ),
                "number_of_channels": self.config.number_of_channels
            }

            # Start AWS stream
            self.current_stream = await self.client.start_stream_transcription(
                **stream_config
            )

            # Initialize store and handler
            self.store.create_session(session_id, self.config.__dict__)
            self.current_handler = TranscriptionHandler(
                self.event_bus,
                self.current_stream.output_stream,
                self.store,
                session_id,
                on_result,
                on_error
            )

            # Start handler
            asyncio.create_task(self._run_handler())
            self.state = TranscriptionState.STREAMING

            return session_id

        except Exception as e:
            self.state = TranscriptionState.ERROR
            self._error = str(e)
            if isinstance(e, (StreamingError, ConfigurationError)):
                raise
            raise StreamingError(f"Failed to start stream: {e}")

    async def process_audio(
            self,
            chunk: bytes,
            # session_id: str,
            # channel: Optional[str] = None
    ) -> None:
        """Process audio chunk.

        Args:
            chunk: Audio data
            # session_id: Session ID
            # channel: Optional channel identifier

        Raises:
            StreamingError: If processing fails
        """
        try:
            if self.state != TranscriptionState.STREAMING:
                raise StreamingError("Stream not active")

            if not self.current_stream:
                raise StreamingError("Stream not initialized")

            await self.current_stream.input_stream.send_audio_event(
                audio_chunk=chunk
            )
            await self.event_bus.publish(Event(
                type=EventType.AUDIO_CHUNK,
                data={
                    "status": "audio_chunk_sent",
                    "chunk_size": len(chunk)
                }
            ))

        except Exception as e:
            self.state = TranscriptionState.ERROR
            self._error = str(e)
            raise StreamingError(f"Failed to process audio: {e}")

    async def stop_stream(
            self,
            session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Stop transcription stream.

        Args:
            session_id: Optional session ID to stop

        Returns:
            Final results if available

        Raises:
            StreamingError: If stop fails
        """
        try:
            if self.state not in (
                    TranscriptionState.STREAMING,
                    TranscriptionState.ERROR
            ):
                return None

            self.state = TranscriptionState.STOPPING

            if self.current_stream:
                await self.current_stream.input_stream.end_stream()

            # Get final results if session specified
            results = None
            if session_id:
                results = self.store.get_session_results(
                    session_id,
                    include_partial=False
                )
                self.store.cleanup_session(session_id)

            self.current_stream = None
            self.current_handler = None
            self.state = TranscriptionState.IDLE
            self._error = None

            await self.event_bus.publish(Event(
                type=EventType.TRANSCRIPT,
                data={
                    "status": "stream_stopped",
                    "session_id": session_id
                }
            ))

            return results

        except Exception as e:
            self.state = TranscriptionState.ERROR
            self._error = str(e)
            raise StreamingError(f"Failed to stop stream: {e}")

    async def get_results(
            self,
            session_id: str,
            include_partial: bool = False
    ) -> Dict[str, Any]:
        """Get results for session.

        Args:
            session_id: Session ID
            include_partial: Whether to include partial results

        Returns:
            Session results
        """
        return self.store.get_session_results(
            session_id,
            include_partial
        )

    @property
    def status(self) -> Dict[str, Any]:
        """Get current status.

        Returns:
            Status information
        """
        return {
            "state": self.state.value,
            "error": self._error,
            "sessions": len(self.store.sessions)
        }

    async def _run_handler(self) -> None:
        """Run transcription handler."""
        try:
            if not self.current_handler:
                raise StreamingError("Handler not initialized")

            await self.current_handler.handle_events()

        except Exception as e:
            self.state = TranscriptionState.ERROR
            self._error = str(e)
            # Handler will be cleaned up in stop_stream
