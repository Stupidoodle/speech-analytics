from typing import Dict, Any, Optional, AsyncIterator, Callable
from datetime import datetime
import asyncio
import logging

from ..conversation.types import (
    Role,
    StreamResponse,
    BedrockConfig,
    DocumentFormat,
    MessageRole
)
from ..conversation.context import ConversationContext
from ..conversation.manager import ConversationManager
from ..transcription.aws_transcribe import TranscribeManager
from ..events.bus import EventBus, Event, EventType
from ..assistance.enhanced_assistant import AssistanceResponse
from ..conversation.exceptions import ConversationError

logger = logging.getLogger(__name__)


class RealtimeProcessor:
    """Manages real-time processing with role-based context and streaming."""

    def __init__(
            self,
            event_bus: EventBus,
            conversation_manager: ConversationManager,
            transcribe_manager: TranscribeManager,
            role: Role,
            **kwargs
    ) -> None:
        """Initialize realtime processor.

        Args:
            event_bus: Event bus for real-time coordination
            conversation_manager: Conversation manager
            transcribe_manager: Transcription manager
            role: User's role
            **kwargs: Optional callback functions:
                     - on_transcript(str)
                     - on_assistance(AssistanceResponse)
                     - on_tool_use(Dict)
                     - on_error(str)
                     - on_metrics(Dict)
        """
        self.event_bus = event_bus
        self.conversation = conversation_manager
        self.transcribe = transcribe_manager
        self.context = ConversationContext(role)
        self.callbacks = kwargs

        # State tracking
        self.is_running = False
        self.current_speaker: Optional[str] = None
        self.last_analysis_time: Optional[float] = None
        self.analysis_interval = 1.0  # seconds
        self.stream_buffer = []

        # Performance metrics
        self.metrics = {
            "processed_chunks": 0,
            "transcripts_generated": 0,
            "responses_generated": 0,
            "tools_used": 0,
            "errors": 0,
            "latency": []
        }

        # Set up error handling
        self.event_bus.add_error_handler(self._handle_error)

    async def start(
            self,
            audio_config: Optional[Dict[str, Any]] = None,
            model_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Start real-time processing.

        Args:
            audio_config: Optional audio configuration
            model_config: Optional model configuration
        """
        try:
            self.is_running = True

            # Initialize transcription
            await self.transcribe.start_stream()

            # Set up event handlers
            self.event_bus.subscribe(
                EventType.AUDIO_CHUNK,
                self._handle_audio_chunk
            )
            self.event_bus.subscribe(
                EventType.TRANSCRIPT,
                self._handle_transcript
            )
            self.event_bus.subscribe(
                EventType.CONTEXT_UPDATE,
                self._handle_context_update
            )

            # Notify start
            await self._emit_event(
                EventType.STATUS,
                {
                    "status": "started",
                    "role": self.context.role.value,
                    "config": {
                        "audio": audio_config,
                        "model": model_config
                    }
                }
            )

        except Exception as e:
            self.metrics["errors"] += 1
            await self._handle_error(Event(
                type=EventType.ERROR,
                data={"message": f"Start error: {str(e)}"},
                timestamp=datetime.now()
            ))
            raise

    async def stop(self) -> None:
        """Stop real-time processing and cleanup."""
        try:
            self.is_running = False

            # Stop transcription
            await self.transcribe.stop_stream()

            # Unsubscribe from events
            self.event_bus.unsubscribe(
                EventType.AUDIO_CHUNK,
                self._handle_audio_chunk
            )
            self.event_bus.unsubscribe(
                EventType.TRANSCRIPT,
                self._handle_transcript
            )
            self.event_bus.unsubscribe(
                EventType.CONTEXT_UPDATE,
                self._handle_context_update
            )

            # Final metrics
            if "on_metrics" in self.callbacks:
                await self._call_callback(
                    "on_metrics",
                    self.metrics
                )

            # Notify stop
            await self._emit_event(
                EventType.STATUS,
                {
                    "status": "stopped",
                    "metrics": self.metrics
                }
            )

        except Exception as e:
            self.metrics["errors"] += 1
            await self._handle_error(Event(
                type=EventType.ERROR,
                data={"message": f"Stop error: {str(e)}"},
                timestamp=datetime.now()
            ))

    async def add_document(
            self,
            path: str,
            doc_type: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add document to conversation context.

        Args:
            path: Path to document
            doc_type: Document type
            metadata: Optional metadata
        """
        try:
            # Add to context
            document = await self.context.add_document(
                path,
                doc_type,
                metadata
            )

            # Notify document added
            await self._emit_event(
                EventType.DOCUMENT_ADDED,
                {
                    "name": document.name,
                    "type": document.mime_type,
                    "metadata": metadata
                }
            )

        except Exception as e:
            self.metrics["errors"] += 1
            await self._handle_error(Event(
                type=EventType.ERROR,
                data={"message": f"Document error: {str(e)}"},
                timestamp=datetime.now()
            ))

    async def _handle_audio_chunk(self, event: Event) -> None:
        """Process audio chunk events.

        Args:
            event: Audio chunk event
        """
        if not self.is_running:
            return

        try:
            start_time = datetime.now().timestamp()
            self.metrics["processed_chunks"] += 1

            # Process audio
            transcript = await self.transcribe.process_audio(
                event.data["audio"]
            )

            if transcript:
                self.metrics["transcripts_generated"] += 1
                self.stream_buffer.append(transcript)

                # Calculate latency
                latency = datetime.now().timestamp() - start_time
                self.metrics["latency"].append(latency)

                # Publish transcript
                await self._emit_event(
                    EventType.TRANSCRIPT,
                    {
                        "text": transcript,
                        "speaker": self.current_speaker,
                        "latency": latency
                    }
                )

                # Callback
                await self._call_callback(
                    "on_transcript",
                    transcript
                )

        except Exception as e:
            self.metrics["errors"] += 1
            await self._handle_error(Event(
                type=EventType.ERROR,
                data={"message": f"Audio error: {str(e)}"},
                timestamp=datetime.now()
            ))

    async def _handle_transcript(self, event: Event) -> None:
        """Process transcript events with role-based responses.

        Args:
            event: Transcript event
        """
        if not self.is_running:
            return

        try:
            current_time = datetime.now().timestamp()
            should_analyze = (
                    self.last_analysis_time is None or
                    current_time - self.last_analysis_time >=
                    self.analysis_interval
            )

            if should_analyze:
                self.last_analysis_time = current_time
                system_messages = self.context.system_prompts

                # Get streaming response
                async for response in self.conversation.send_message(
                        event.data["text"],
                        system_messages=system_messages
                ):
                    self.metrics["responses_generated"] += 1

                    # Handle content
                    if response.content and response.content.text:
                        await self._emit_event(
                            EventType.ASSISTANCE,
                            {
                                "type": "suggestion",
                                "content": response.content.text,
                                "role": self.context.role.value,
                                "context": {
                                    "speaker": event.data.get("speaker"),
                                    "latency": event.data.get("latency")
                                }
                            }
                        )

                        await self._call_callback(
                            "on_assistance",
                            AssistanceResponse(
                                suggestion=response.content.text,
                                confidence=0.9,
                                context={
                                    "role": self.context.role.value,
                                    "speaker": event.data.get("speaker")
                                },
                                timestamp=datetime.now()
                            )
                        )

                    # Handle tool use
                    if response.content and response.content.tool_use:
                        self.metrics["tools_used"] += 1
                        await self._handle_tool_use(
                            response.content.tool_use
                        )

                    # Handle metadata
                    if response.metadata:
                        await self._emit_event(
                            EventType.METRICS,
                            response.metadata.model_dump()
                        )

        except Exception as e:
            self.metrics["errors"] += 1
            await self._handle_error(Event(
                type=EventType.ERROR,
                data={"message": f"Transcript error: {str(e)}"},
                timestamp=datetime.now()
            ))

    async def _handle_tool_use(self, tool_use: Dict[str, Any]) -> None:
        """Handle tool use responses.

        Args:
            tool_use: Tool use data
        """
        await self._emit_event(
            EventType.TOOL_USE,
            tool_use
        )

        await self._call_callback(
            "on_tool_use",
            tool_use
        )

    async def _handle_context_update(self, event: Event) -> None:
        """Handle context update events.

        Args:
            event: Context update event
        """
        try:
            if "document" in event.data:
                await self.add_document(**event.data["document"])
            elif "system_prompt" in event.data:
                await self.context.add_system_prompt(
                    event.data["system_prompt"]
                )

        except Exception as e:
            self.metrics["errors"] += 1
            await self._handle_error(Event(
                type=EventType.ERROR,
                data={"message": f"Context error: {str(e)}"},
                timestamp=datetime.now()
            ))

    async def _handle_error(self, event: Event) -> None:
        """Handle errors with proper logging and notification.

        Args:
            event: Error event
        """
        error_msg = event.data["message"]
        logger.error(error_msg)

        # Callback
        await self._call_callback(
            "on_error",
            error_msg
        )

    async def _emit_event(
            self,
            event_type: EventType,
            data: Dict[str, Any]
    ) -> None:
        """Emit event with proper structure.

        Args:
            event_type: Type of event
            data: Event data
        """
        await self.event_bus.publish(Event(
            type=event_type,
            data=data,
            timestamp=datetime.now(),
            role=self.context.role,
            metadata={"processor_id": id(self)}
        ))

    async def _call_callback(
            self,
            name: str,
            data: Any
    ) -> None:
        """Safely execute callback if it exists.

        Args:
            name: Callback name
            data: Data for callback
        """
        if name in self.callbacks:
            try:
                callback = self.callbacks[name]
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                await self._handle_error(Event(
                    type=EventType.ERROR,
                    data={
                        "message": f"Callback error ({name}): {str(e)}"
                    },
                    timestamp=datetime.now()
                ))