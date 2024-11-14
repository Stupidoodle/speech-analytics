"""Audio buffer management for transcription streaming."""
from collections import deque
from datetime import datetime
from typing import Dict, Optional, AsyncIterator, Set

import asyncio

from src.events.bus import EventBus
from src.events.types import Event, EventType
from .exceptions import BufferError
from .types import AudioConfig, BufferMetrics, BufferStatus


class AudioBuffer:
    """Manages buffering of audio data for transcription streams."""

    def __init__(
            self,
            event_bus: EventBus,
            config: AudioConfig,
            max_size: int = 32768,  # 32KB default
            chunk_size: int = 1024,  # 1KB chunks
    ) -> None:
        """Initialize audio buffer.

        Args:
            event_bus: Event bus instance
            config: Audio configuration
            max_size: Maximum buffer size in bytes
            chunk_size: Size of chunks to return
        """
        self.event_bus = event_bus
        self.config = config
        self.max_size = max_size
        self.chunk_size = chunk_size

        # Initialize channel buffers
        self._buffers: Dict[str, deque] = {
            "main": deque(),
            "channel_1": deque(),
        }
        if config.channels > 1:
            self._buffers["channel_2"] = deque()

        # Track buffer sizes
        self._sizes: Dict[str, int] = {k: 0 for k in self._buffers}

        # Track timestamps
        self._last_write: Dict[str, Optional[datetime]] = {
            k: None for k in self._buffers
        }
        self._last_read: Dict[str, Optional[datetime]] = {
            k: None for k in self._buffers
        }

        # Initialize metrics
        self._metrics = BufferMetrics()

    async def write(
            self,
            data: bytes,
            channel: Optional[str] = None
    ) -> None:
        """Write audio data to buffer.

        Args:
            data: Audio data bytes
            channel: Optional channel identifier

        Raises:
            BufferError: If buffer would overflow or invalid channel
        """
        try:
            if len(data) % 2 != 0:  # Validate 16-bit samples
                raise BufferError("Invalid sample alignment")

            buffer_key = channel or "main"
            if buffer_key not in self._buffers:
                raise BufferError(f"Invalid channel: {buffer_key}")

            buffer = self._buffers[buffer_key]
            current_size = self._sizes[buffer_key]

            # Handle overflow
            if current_size + len(data) > self.max_size:
                self._metrics.overflow_count += 1
                while current_size + len(data) > self.max_size:
                    removed = buffer.popleft()
                    current_size -= len(removed)
                    self._sizes[buffer_key] = current_size

            # Add new data
            buffer.append(data)
            self._sizes[buffer_key] = current_size + len(data)
            self._last_write[buffer_key] = datetime.now()
            self._metrics.total_bytes_written += len(data)

            await self._publish_buffer_event("write_complete", len(data))

        except BufferError:
            raise
        except Exception as e:
            raise BufferError(f"Write failed: {str(e)}")

    async def read(
            self,
            size: Optional[int] = None,
            channel: Optional[str] = None,
            timeout: Optional[float] = None
    ) -> Optional[bytes]:
        """Read audio data from buffer.

        Args:
            size: Number of bytes to read
            channel: Optional channel identifier
            timeout: Optional read timeout in seconds

        Returns:
            Audio data if available, None otherwise

        Raises:
            BufferError: If read fails
        """
        try:
            read_size = size or self.chunk_size
            buffer_key = channel or "main"

            if buffer_key not in self._buffers:
                raise BufferError(f"Invalid channel: {buffer_key}")

            buffer = self._buffers[buffer_key]
            current_size = self._sizes[buffer_key]

            # Wait for data if timeout specified
            if timeout and current_size < read_size:
                start = datetime.now()
                while (
                        current_size < read_size and
                        (datetime.now() - start).total_seconds() < timeout
                ):
                    await asyncio.sleep(0.001)
                    current_size = self._sizes[buffer_key]

            # Check if enough data available
            if current_size < read_size:
                self._metrics.underrun_count += 1
                return None

            # Combine data from buffer
            data = bytes()
            remaining = read_size

            while remaining > 0 and buffer:
                chunk = buffer.popleft()
                if len(chunk) <= remaining:
                    data += chunk
                    remaining -= len(chunk)
                    self._sizes[buffer_key] -= len(chunk)
                else:
                    # Split chunk if needed
                    data += chunk[:remaining]
                    buffer.appendleft(chunk[remaining:])
                    self._sizes[buffer_key] -= remaining
                    remaining = 0

            self._last_read[buffer_key] = datetime.now()
            self._metrics.total_bytes_read += len(data)

            await self._publish_buffer_event("read_complete", len(data))
            return data

        except BufferError:
            raise
        except Exception as e:
            raise BufferError(f"Read failed: {str(e)}")

    async def read_stream(
            self,
            channel: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """Stream audio data from buffer.

        Args:
            channel: Optional channel identifier

        Yields:
            Audio chunks
        """
        while True:
            chunk = await self.read(channel=channel)
            if chunk:
                yield chunk
            else:
                await asyncio.sleep(0.001)

    def get_status(self) -> BufferStatus:
        """Get current buffer status.

        Returns:
            Buffer status information
        """
        active_channels: Set[str] = {
            k for k, v in self._sizes.items() if v > 0
        }

        return BufferStatus(
            levels={
                k: (v / self.max_size) * 100
                for k, v in self._sizes.items()
            },
            latencies={
                k: self._calculate_latency(k)
                for k in self._buffers.keys()
            },
            active_channels=active_channels,
            metrics=self._metrics
        )

    def clear(self, channel: Optional[str] = None) -> None:
        """Clear buffer contents.

        Args:
            channel: Optional channel to clear
        """
        if channel:
            if channel in self._buffers:
                self._buffers[channel].clear()
                self._sizes[channel] = 0
        else:
            for key in self._buffers:
                self._buffers[key].clear()
                self._sizes[key] = 0

    def _calculate_latency(self, channel: str) -> float:
        """Calculate buffer latency in milliseconds.

        Args:
            channel: Channel identifier

        Returns:
            Latency in milliseconds
        """
        samples = self._sizes[channel] // 2  # 16-bit samples
        return (samples / self.config.sample_rate) * 1000

    async def _publish_buffer_event(
            self,
            status: str,
            bytes_processed: int
    ) -> None:
        """Publish buffer event.

        Args:
            status: Event status
            bytes_processed: Number of bytes processed
        """
        await self.event_bus.publish(
            Event(
                type=EventType.AUDIO_CHUNK,
                data={
                    "status": status,
                    "bytes_processed": bytes_processed,
                    # Changed .dict() to .model_dump()
                    "buffer_status": self.get_status().model_dump()
                }
            )
        )