"""Audio buffer management for transcription streaming."""
from typing import Optional, Dict, Any, AsyncIterator
import asyncio
from collections import deque
from datetime import datetime

from .exceptions import BufferError
from .types import TranscriptionConfig

from src.events.types import Event, EventType
from src.events.bus import EventBus


class AudioBuffer:
    """Manages audio buffering for transcription streams."""

    def __init__(
            self,
            event_bus: EventBus,
            config: TranscriptionConfig,
            max_size: int = 32768,  # 32KB default
            chunk_size: int = 1024  # 1KB chunks
    ):
        """Initialize audio buffer.

        Args:
            config: Transcription configuration
            max_size: Maximum buffer size in bytes
            chunk_size: Size of chunks to return in bytes
        """
        self.event_bus = event_bus
        self.config = config
        self.max_size = max_size
        self.chunk_size = chunk_size

        # Initialize buffers for each channel
        self._buffers: Dict[str, deque] = {
            "combined": deque(),
            "channel_1": deque(),
            "channel_2": deque() if config.number_of_channels > 1 else None
        }

        self._buffer_sizes = {
            key: 0 for key in self._buffers.keys() if self._buffers[key] is not None
        }

        # Timing management
        self._last_write: dict[str, Optional[datetime]] = {
            key: None for key in self._buffers.keys() if self._buffers[key] is not None
        }
        self._last_read: dict[str, Optional[datetime]] = {
            key: None for key in self._buffers.keys() if self._buffers[key] is not None
        }

        # Statistics
        self.stats = {
            "total_bytes_written": 0,
            "total_bytes_read": 0,
            "overflow_count": 0,
            "underrun_count": 0
        }

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
            BufferError: If buffer would overflow
        """
        try:
            # Validate data length
            if len(data) % 2 != 0:  # 16-bit samples
                raise BufferError("Invalid data length")

            # Determine target buffer
            buffer_key = channel or "combined"
            if buffer_key not in self._buffers or self._buffers[buffer_key] is None:
                raise BufferError(f"Invalid buffer: {buffer_key}")

            buffer = self._buffers[buffer_key]
            current_size = self._buffer_sizes[buffer_key]

            # Check for overflow
            if current_size + len(data) > self.max_size:
                self.stats["overflow_count"] += 1
                # Remove oldest data to make room
                while current_size + len(data) > self.max_size:
                    removed = buffer.popleft()
                    current_size -= len(removed)
                    self._buffer_sizes[buffer_key] = current_size

            # Add new data
            buffer.append(data)
            self._buffer_sizes[buffer_key] = current_size + len(data)
            self._last_write[buffer_key] = datetime.now()
            self.stats["total_bytes_written"] += len(data)

            await self.event_bus.publish(Event(
                type=EventType.AUDIO_CHUNK,
                data={
                    "status": "audio_buffered",
                    "bytes_written": len(data)
                }
            ))
        except BufferError:
            raise
        except Exception as e:
            raise BufferError(f"Write failed: {e}")

    async def read(
            self,
            size: Optional[int] = None,
            channel: Optional[str] = None,
            timeout: Optional[float] = None
    ) -> Optional[bytes]:
        """Read audio data from buffer.

        Args:
            size: Number of bytes to read (defaults to chunk_size)
            channel: Optional channel identifier
            timeout: Optional read timeout in seconds

        Returns:
            Audio data or None if no data available

        Raises:
            BufferError: If read fails
        """
        try:
            read_size = size or self.chunk_size
            buffer_key = channel or "combined"

            if buffer_key not in self._buffers or self._buffers[buffer_key] is None:
                raise BufferError(f"Invalid buffer: {buffer_key}")

            buffer = self._buffers[buffer_key]
            current_size = self._buffer_sizes[buffer_key]

            # Wait for data if timeout specified
            if timeout and current_size < read_size:
                start_time = datetime.now()
                while (
                        current_size < read_size and
                        (datetime.now() - start_time).total_seconds() < timeout
                ):
                    await asyncio.sleep(0.001)
                    current_size = self._buffer_sizes[buffer_key]

            # Check if enough data available
            if current_size < read_size:
                self.stats["underrun_count"] += 1
                return None

            # Combine data from buffer
            data = bytes()
            remaining = read_size

            while remaining > 0 and buffer:
                chunk = buffer.popleft()
                if len(chunk) <= remaining:
                    data += chunk
                    remaining -= len(chunk)
                    self._buffer_sizes[buffer_key] -= len(chunk)
                else:
                    # Split chunk if needed
                    data += chunk[:remaining]
                    buffer.appendleft(chunk[remaining:])
                    self._buffer_sizes[buffer_key] -= remaining
                    remaining = 0

            self._last_read[buffer_key] = datetime.now()
            self.stats["total_bytes_read"] += len(data)
            return data

        except BufferError:
            raise
        except Exception as e:
            raise BufferError(f"Read failed: {e}")

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

    def get_level(
            self,
            channel: Optional[str] = None
    ) -> float:
        """Get current buffer level as percentage.

        Args:
            channel: Optional channel identifier

        Returns:
            Buffer level percentage (0-100)
        """
        buffer_key = channel or "combined"
        if buffer_key not in self._buffer_sizes:
            return 0.0
        return (self._buffer_sizes[buffer_key] / self.max_size) * 100

    def get_latency(
            self,
            channel: Optional[str] = None
    ) -> float:
        """Get current buffer latency in milliseconds.

        Args:
            channel: Optional channel identifier

        Returns:
            Buffer latency in ms
        """
        buffer_key = channel or "combined"
        samples = self._buffer_sizes[buffer_key] // 2  # 16-bit samples
        return (samples / self.config.sample_rate_hz) * 1000

    def clear(
            self,
            channel: Optional[str] = None
    ) -> None:
        """Clear buffer contents.

        Args:
            channel: Optional channel identifier to clear
        """
        if channel:
            if channel in self._buffers and self._buffers[channel]:
                self._buffers[channel].clear()
                self._buffer_sizes[channel] = 0
        else:
            for key in self._buffers:
                if self._buffers[key]:
                    self._buffers[key].clear()
                    self._buffer_sizes[key] = 0

    @property
    def status(self) -> Dict[str, Any]:
        """Get buffer status.

        Returns:
            Buffer status information
        """
        return {
            "levels": {
                key: self.get_level(key)
                for key in self._buffers.keys()
                if self._buffers[key] is not None
            },
            "latencies": {
                key: self.get_latency(key)
                for key in self._buffers.keys()
                if self._buffers[key] is not None
            },
            "stats": self.stats
        }
