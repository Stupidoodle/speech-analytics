"""Audio capture and stream management."""
from typing import Optional, Dict, Any, AsyncIterator
from datetime import datetime

import numpy as np
import asyncio

from src.events.bus import EventBus
from src.events.types import Event, EventType
from .devices import DeviceManager
from .mixer import AudioMixer
from .processor import AudioProcessor
from .types import AudioConfig, ProcessingResult, DeviceInfo
from .exceptions import (
    CaptureError,
    DeviceError,
    ProcessingError,
)


class AudioCapture:
    """Manages audio capture and processing pipeline."""

    def __init__(
        self,
        event_bus: EventBus,
        config: AudioConfig,
        device_id: Optional[int] = None,
    ) -> None:
        """Initialize audio capture.

        Args:
            event_bus: Event bus instance
            config: Audio configuration
            device_id: Optional device ID
        """
        self.event_bus = event_bus
        self.config = config
        self.device_id = device_id

        # Initialize components
        self.device_manager = DeviceManager()
        self.mixer = AudioMixer(event_bus, config)
        self.processor = AudioProcessor(event_bus, config)

        # State management
        self._running = False
        self._paused = False
        self._stream: Optional[AsyncIterator[bytes]] = None
        self._stats: Dict[str, Any] = {
            "bytes_processed": 0,
            "chunks_processed": 0,
            "start_time": None,
            "last_chunk": None,
        }

    async def start_capture(self) -> None:
        """Start audio capture.

        Raises:
            CaptureError: If capture fails to start
            DeviceError: If device initialization fails
        """
        if self._running:
            raise CaptureError("Capture already running")

        try:
            # Validate device
            if not await self._init_device():
                raise DeviceError("Failed to initialize device")

            self._running = True
            self._stats["start_time"] = datetime.now()

            # Start processing loop
            asyncio.create_task(self._process_audio())

            await self._publish_capture_event("capture_started", {})

        except Exception as e:
            await self.stop_capture()
            raise CaptureError(f"Failed to start capture: {e}")

    async def stop_capture(self) -> None:
        """Stop audio capture."""
        self._running = False
        self._stream = None
        self._stats["last_chunk"] = datetime.now()

        await self._publish_capture_event(
            "capture_stopped",
            {
                "duration": (
                    datetime.now() - self._stats["start_time"]
                ).total_seconds()
                if self._stats["start_time"]
                else 0
            }
        )

    async def pause_capture(self) -> None:
        """Pause audio capture."""
        self._paused = True
        await self._publish_capture_event("capture_paused", {})

    async def resume_capture(self) -> None:
        """Resume audio capture."""
        self._paused = False
        await self._publish_capture_event("capture_resumed", {})

    async def get_levels(self) -> Dict[str, float]:
        """Get current audio levels.

        Returns:
            Dict of audio levels
        """
        return {
            "peak": self._stats.get("peak_level", 0.0),
            "rms": self._stats.get("rms_level", 0.0),
        }

    async def _init_device(self) -> bool:
        """Initialize audio device.

        Returns:
            Whether initialization succeeded
        """
        try:
            if not self.device_id:
                device_info = await self.device_manager.get_default_device(
                    self.config.device_type
                )
                if not device_info:
                    raise DeviceError("No default device found")
                self.device_id = device_info.id

            # Validate device
            if not await self.device_manager.validate_device(
                self.device_id,
                self.config
            ):
                raise DeviceError(
                    f"Device {self.device_id} not compatible"
                )

            # Open stream
            self._stream = await self.device_manager.open_stream(
                self.device_id,
                self.config
            )
            return True

        except Exception as e:
            await self._publish_capture_event(
                "device_error",
                {"error": str(e)}
            )
            return False

    async def _process_audio(self) -> None:
        """Main audio processing loop."""
        if not self._stream:
            return

        while self._running:
            try:
                if self._paused:
                    await asyncio.sleep(0.1)
                    continue

                # Get next chunk
                async for chunk in self._stream:
                    if not chunk or not self._running:
                        break

                    # Process audio
                    try:
                        result = await self._process_chunk(chunk)
                        await self._update_stats(result)
                        await self._publish_capture_event(
                            "chunk_processed",
                            {
                                "chunk_size": len(chunk),
                                "metrics": result.metrics.dict()
                            }
                        )
                    except ProcessingError as e:
                        await self._publish_capture_event(
                            "processing_error",
                            {"error": str(e)}
                        )

                    await asyncio.sleep(0.001)

            except Exception as e:
                await self._publish_capture_event(
                    "capture_error",
                    {"error": str(e)}
                )
                await asyncio.sleep(0.1)

    async def _process_chunk(self, chunk: bytes) -> ProcessingResult:
        """Process audio chunk.

        Args:
            chunk: Raw audio data

        Returns:
            Processing result

        Raises:
            ProcessingError: If processing fails
        """
        # Mix audio if needed
        if self.config.channels > 1:
            result = await self.mixer.mix_streams(chunk)
            chunk = result.processed_data

        # Process audio
        return await self.processor.process_chunk(
            chunk,
            apply_noise_reduction=self.config.enable_noise_reduction,
            apply_gain=self.config.enable_auto_gain
        )

    async def _update_stats(self, result: ProcessingResult) -> None:
        """Update capture statistics.

        Args:
            result: Processing result
        """
        self._stats.update({
            "bytes_processed": (
                self._stats["bytes_processed"] + len(result.processed_data)
            ),
            "chunks_processed": self._stats["chunks_processed"] + 1,
            "last_chunk": datetime.now(),
            "peak_level": result.metrics.peak_level,
            "rms_level": result.metrics.rms_level
        })

    async def _publish_capture_event(
        self,
        status: str,
        data: Dict[str, Any]
    ) -> None:
        """Publish capture event.

        Args:
            status: Event status
            data: Event data
        """
        await self.event_bus.publish(
            Event(
                type=EventType.AUDIO_CHUNK,
                data={
                    "status": status,
                    "device_id": self.device_id,
                    "stats": self._stats,
                    **data
                }
            )
        )

    async def __aenter__(self) -> 'AudioCapture':
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        await self.stop_capture()