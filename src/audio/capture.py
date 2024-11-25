"""Audio capture and stream management."""

from datetime import datetime
from typing import Any, AsyncIterator, Dict, Optional

from src.events.bus import EventBus
from src.events.types import Event, EventType

from .buffer import AudioBuffer
from .devices import DeviceManager
from .exceptions import CaptureError, DeviceError
from .mixer import AudioMixer
from .processor import AudioProcessor
from .types import AudioConfig, DeviceInfo, ProcessingResult


class AudioCapture:
    """Manages audio capture and processing pipeline."""

    def __init__(
        self,
        event_bus: EventBus,
        config: AudioConfig,
    ) -> None:
        """Initialize audio capture."""
        self.event_bus = event_bus
        self.config = config

        # Initialize core components
        self.device_manager = DeviceManager()
        self.processor = AudioProcessor(event_bus, config)
        self.mixer = AudioMixer(event_bus, config)
        self.buffer = AudioBuffer(event_bus, config)

        # Device information
        self.mic_device: Optional[DeviceInfo] = None
        self.desktop_device: Optional[DeviceInfo] = None

        # State management
        self._running = False
        self._stats: Dict[str, Any] = {
            "bytes_processed": 0,
            "chunks_processed": 0,
            "start_time": None,
            "last_chunk": None,
        }

    async def start_capture(self) -> None:
        """Start audio capture."""
        if self._running:
            raise CaptureError("Capture already running")

        try:
            # Initialize devices
            if not await self._init_devices():
                raise DeviceError("Failed to initialize devices")

            self._running = True
            self._stats["start_time"] = datetime.now()

            # Start processing loop
            await self._publish_capture_event("capture_started", {})
            await self._process_audio()

        except Exception as e:
            await self.stop_capture()
            raise CaptureError(f"Failed to start capture: {e}")

    async def stop_capture(self) -> None:
        """Stop audio capture."""
        if not self._running:
            return

        self._running = False
        self._stats["last_chunk"] = datetime.now()

        await self._publish_capture_event(
            "capture_stopped",
            {
                "duration": (
                    (datetime.now() - self._stats["start_time"]).total_seconds()
                    if self._stats["start_time"]
                    else 0
                )
            },
        )

    async def _init_devices(self) -> bool:
        """Initialize microphone and desktop audio devices."""
        try:
            devices = await self.device_manager.get_default_stereo_devices()
            self.mic_device = devices["mic"]
            self.desktop_device = devices["desktop"]
            return True
        except Exception as e:
            await self._publish_capture_event("device_error", {"error": str(e)})
            return False

    async def _process_audio(self) -> None:
        """Main audio processing loop."""
        try:
            # Open audio streams
            mic_stream = self.device_manager.open_stream(self.mic_device, self.config)
            desktop_stream = self.device_manager.open_stream(
                self.desktop_device, self.config
            )

            print("Processing loop started.")

            async for mic_chunk, desktop_chunk in self._read_chunks(
                mic_stream, desktop_stream
            ):
                try:
                    # Process chunks
                    mic_result = await self.processor.process_chunk(mic_chunk)
                    desktop_result = await self.processor.process_chunk(desktop_chunk)

                    # Mix and write to buffer
                    mixed_result = await self.mixer.mix_streams(
                        mic_result.processed_data, desktop_result.processed_data
                    )
                    await self.buffer.write(mixed_result.processed_data)
                    await self._update_stats(mixed_result)

                    # Publish processing event
                    await self._publish_capture_event(
                        "chunk_processed",
                        {
                            "chunk_size": len(mixed_result.processed_data),
                            "metrics": mixed_result.metrics.__dict__,
                        },
                    )

                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    await self._publish_capture_event(
                        "capture_error", {"error": str(e)}
                    )
                    if not self._running:
                        break

        except Exception as e:
            print(f"Error in processing loop: {e}")
            await self._publish_capture_event("capture_error", {"error": str(e)})

    async def _read_chunks(
        self, mic_stream: AsyncIterator[bytes], desktop_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[tuple[bytes, bytes]]:
        """Read chunks from mic and desktop streams."""
        while self._running:
            try:
                mic_chunk = await anext(mic_stream)
                desktop_chunk = await anext(desktop_stream)

                if mic_chunk is None or desktop_chunk is None:
                    print("One of the streams ended. Breaking the loop.")
                    break

                yield mic_chunk, desktop_chunk

            except StopAsyncIteration:
                print("Stream iteration completed.")
                break
            except Exception as e:
                print(f"Error reading chunk: {e}")
                await self._publish_capture_event("capture_error", {"error": str(e)})
                if not self._running:
                    break

    async def _update_stats(self, result: ProcessingResult) -> None:
        """Update capture statistics."""
        self._stats["bytes_processed"] += len(result.processed_data)
        self._stats["chunks_processed"] += 1
        self._stats["last_chunk"] = datetime.now()

    async def _publish_capture_event(self, status: str, data: Dict[str, Any]) -> None:
        """Publish capture event."""
        await self.event_bus.publish(
            Event(
                type=EventType.AUDIO_CHUNK,
                data={
                    "status": status,
                    "mic_device_id": self.mic_device.id if self.mic_device else None,
                    "desktop_device_id": (
                        self.desktop_device.id if self.desktop_device else None
                    ),
                    "stats": self._stats,
                    **data,
                },
            )
        )
