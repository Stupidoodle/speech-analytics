from typing import Dict, Any, Optional
import asyncio
from datetime import datetime
import time

from ..audio.capture import AudioCapture
from ..transcription.aws_transcribe import TranscribeManager
from ..conversation.manager import ConversationManager
from ..assistance.enhanced_assistant import ConversationAssistant, Role
from ..events.bus import EventBus, Event, EventType


class RealtimeProcessor:
    """Manages real-time audio processing, transcription, and assistance."""

    def __init__(
        self,
        event_bus: EventBus,
        conversation_manager: ConversationManager,
        transcribe_manager: TranscribeManager,
        role: Role
    ):
        """Initialize realtime processor.

        Args:
            event_bus: Event bus for real-time processing
            conversation_manager: Conversation manager for real-time processing
            transcribe_manager: Transcribe manager for real-time processing
            role: User's role in conversation
        """
        self.event_bus = event_bus
        self.conversation = conversation_manager
        self.transcribe = transcribe_manager
        self.role = role
        self.is_running = False

        # Subscribe to relevant events
        self.event_bus.subscribe(
            EventType.AUDIO_CHUNK,
            self._handle_audio_chunk,
            {self.role}
        )
        self.event_bus.subscribe(
            EventType.TRANSCRIPTION,
            self._handle_transcription,
            {self.role}
        )
        self.event_bus.subscribe(
            EventType.CONTEXT_UPDATE,
            self._handle_context_update,
            {self.role}
        )

    async def start(
        self,
        region: str,
        mic_device_id: Optional[int] = None,
        desktop_device_id: Optional[int] = None
    ) -> None:
        """Start real-time processing."""
        try:
            # Initialize audio capture
            self.audio_capture = AudioCapture(
                mic_device_id=mic_device_id,
                desktop_device_id=desktop_device_id
            )

            # Initialize managers
            self.transcribe_manager = TranscribeManager(region=region)
            self.conversation_manager = ConversationManager(
                region=region,
                model_id="anthropic.claude-3-sonnet-20240229-v1:0"
            )

            # Initialize assistant
            self.assistant = ConversationAssistant(
                self.conversation_manager,
                role=self.role,
                context_type="interview"
            )

            # Start capture
            await self.audio_capture.start_capture()

            # Start processing
            self.is_running = True
            await self._process_audio_stream()

        except Exception as e:
            if self.on_error:
                await self._call_callback(self.on_error, str(e))
            raise

    async def stop(self) -> None:
        """Stop real-time processing."""
        self.is_running = False

        # Stop audio capture if it exists
        if self.audio_capture:
            try:
                await self.audio_capture.stop_capture()
            finally:
                # Clear references only after stopping
                self.audio_capture = None

        # Clear other components
        self.transcribe_manager = None
        self.conversation_manager = None
        self.assistant = None

    async def add_context(
        self,
        content: Any,
        doc_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add context document for processing."""
        if self.assistant:
            await self.assistant.add_document(content, doc_type, metadata)

    async def _process_audio_stream(self) -> None:
        """Process audio stream in real-time."""
        try:
            while self.is_running:
                if not self.audio_capture:
                    break

                chunk = await self.audio_capture._read_stream()
                if chunk:
                    await self._process_chunk(chunk)

                # Small delay to prevent CPU overload
                await asyncio.sleep(0.001)

        except Exception as e:
            self.is_running = False
            if self.on_error:
                await self._call_callback(self.on_error, str(e))
            raise

    async def _get_audio_chunk(self) -> Optional[bytes]:
        """Get next audio chunk."""
        try:
            # Read from audio capture
            chunk = await self.audio_capture._read_stream('mic')
            if chunk is None:
                return None

            # Get audio levels for speaker detection
            levels = await self.audio_capture.get_audio_levels()
            self._update_speaker(levels)

            return chunk

        except Exception as e:
            if self.on_error:
                self.on_error(f"Error getting audio chunk: {e}")
            return None

    async def _process_chunk(self, chunk: bytes) -> None:
        """Process a single chunk of audio data.

        Args:
            chunk: Raw audio data to process
        """
        try:
            # Process audio through transcription
            transcript = await self.transcribe_manager.process_audio(chunk)

            # Call transcript callback if we have a transcript
            if transcript and self.on_transcript:
                # Ensure we await the callback
                await self._call_callback(self.on_transcript, transcript)

            # Check if we should run analysis
            current_time = time.time()
            should_analyze = (
                self.last_analysis_time is None or
                current_time - self.last_analysis_time >=
                self.analysis_interval
            )

            # Process through conversation if
            # we have a transcript and should analyze
            if transcript and should_analyze:
                await self._process_conversation(transcript)
                self.last_analysis_time = current_time

        except Exception as e:
            if self.on_error:
                await self._call_callback(self.on_error, str(e))

    async def _process_conversation(self, transcript: str) -> None:
        """Process transcript through conversation manager.

        Args:
            transcript: Text to process
        """
        try:
            if self.conversation_manager:
                # Process through conversation manager
                async for response in\
                        self.conversation_manager.process_message(transcript):
                    if self.on_suggestion and response.get('suggestion'):
                        await self._call_callback(
                            self.on_suggestion,
                            response['suggestion']
                        )
                    if self.on_analysis and response.get('analysis'):
                        await self._call_callback(
                            self.on_analysis,
                            response['analysis']
                        )
        except Exception as e:
            if self.on_error:
                await self._call_callback(
                    self.on_error,
                    f"Conversation error: {str(e)}"
                )

    async def _call_callback(self, callback, data):
        """Safely execute a callback function.

        Args:
            callback: Callback function to execute
            data: Data to pass to callback
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                # For non-async callbacks, run in executor to prevent blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, data)
        except Exception as e:
            if self.on_error and callback != self.on_error:
                await self._call_callback(
                    self.on_error,
                    f"Callback error: {str(e)}"
                )

    def _update_speaker(self, levels: Dict[str, float]) -> None:
        """Update current speaker based on audio levels."""
        mic_level = levels.get('mic', float('-inf'))
        desktop_level = levels.get('desktop', float('-inf'))

        # Simple threshold-based speaker detection
        if mic_level > -30 and mic_level > desktop_level + 10:
            self.current_speaker = 'local'
        elif desktop_level > -30 and desktop_level > mic_level + 10:
            self.current_speaker = 'remote'
        else:
            self.current_speaker = None

    def _should_run_analysis(self) -> bool:
        """Determine if we should run analysis now."""
        now = datetime.now()
        if not self.last_analysis_time:
            self.last_analysis_time = now
            return True

        time_diff = (now - self.last_analysis_time).total_seconds()
        if time_diff >= self.analysis_interval:
            self.last_analysis_time = now
            return True

        return False

    async def _run_analysis(self, transcript: str) -> None:
        """Run analysis on current transcript."""
        try:
            analysis = await self.assistant.analyze_response(transcript)

            if self.on_analysis:
                self.on_analysis({
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                })

        except Exception as e:
            if self.on_error:
                self.on_error(f"Analysis error: {e}")

    async def _get_suggestions(self, transcript: str) -> None:
        """Get and send real-time suggestions."""
        try:
            async for suggestion in self.assistant.get_suggestions(transcript):
                if self.on_suggestion:
                    self.on_suggestion({
                        'suggestion': suggestion.text,
                        'type': 'real-time',
                        'timestamp': datetime.now().isoformat()
                    })

        except Exception as e:
            if self.on_error:
                self.on_error(f"Suggestion error: {e}")

    async def _handle_audio_chunk(self, event: Event) -> None:
        """Process audio chunk and generate transcription."""
        if not self.is_running:
            return

        result = await self.transcribe.process_audio(event.data)
        if result:
            await self.event_bus.publish(Event(
                type=EventType.TRANSCRIPTION,
                data=result.text,
                timestamp=datetime.now(),
                role=self.role
            ))

    async def _handle_transcription(self, event: Event) -> None:
        """Process transcription and generate assistance."""
        if not self.is_running:
            return

        try:
            async for response in self.conversation.process_realtime(
                event.data,
                self.role
            ):
                await self.event_bus.publish(Event(
                    type=EventType.ASSISTANCE,
                    data=response,
                    timestamp=datetime.now(),
                    role=self.role
                ))
        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.ERROR,
                data=str(e),
                timestamp=datetime.now(),
                role=self.role,
                metadata={"stage": "assistance"}
            ))

    async def _handle_context_update(self, event: Event) -> None:
        """Handle context updates during conversation."""
        if not self.is_running:
            return

        try:
            # Update conversation context
            await self.conversation.update_context(event.data)

            # Generate new assistance based on updated context
            async for response in self.conversation.process_realtime(
                "",  # Empty string triggers context-only analysis
                self.role
            ):
                await self.event_bus.publish(Event(
                    type=EventType.ASSISTANCE,
                    data=response,
                    timestamp=datetime.now(),
                    role=self.role,
                    metadata={"trigger": "context_update"}
                ))
        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.ERROR,
                data=str(e),
                timestamp=datetime.now(),
                role=self.role,
                metadata={"stage": "context_update"}
            ))
