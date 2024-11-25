"""Audio buffer management for stereo transcription streaming."""

import asyncio
from collections import deque
from datetime import datetime
from typing import AsyncIterator, Dict, Optional

from src.events.bus import EventBus
from src.events.types import Event, EventType

from .exceptions import BufferError
from .types import AudioConfig, BufferMetrics, BufferStatus, ChannelConfig


class AudioBuffer:
    """Manages buffering of stereo audio data for transcription."""

    def __init__(
        self,
        event_bus: EventBus,
        config: AudioConfig,
        max_size: int = 32768,  # 32KB default
    ):
        """Initialize audio buffer.

        Args:
            event_bus: Event bus instance
            config: Audio configuration
            max_size: Maximum buffer size in bytes
        """
        self.event_bus = event_bus
        self.config = config
        self.max_size = max_size
        self.chunk_size = config.chunk_size

        # Initialize channel buffers
        self._buffers: Dict[str, deque] = {
            # Combined stereo buffer
            "combined": deque(),
            # Individual channel buffers
            "ch_0": deque(),  # Left channel (mic)
            "ch_1": deque(),  # Right channel (desktop)
        }

        # Track buffer sizes
        self._sizes = {k: 0 for k in self._buffers}

        # Track timestamps
        self._last_write = {k: None for k in self._buffers}
        self._last_read = {k: None for k in self._buffers}

        # Initialize metrics
        self._metrics = BufferMetrics()

    async def write(self, data: bytes, channel: Optional[str] = None) -> None:
        """Write audio data to buffer.

        Args:
            data: Audio data bytes
            channel: Optional channel identifier

        Raises:
            BufferError: If buffer would overflow
        """
        try:
            # Validate data alignment for stereo
            if len(data) % (2 * self.config.channels) != 0:
                raise BufferError("Invalid stereo sample alignment")

            buffer_key = channel or "combined"
            if buffer_key not in self._buffers:
                raise BufferError(f"Invalid channel: {buffer_key}")

            buffer = self._buffers[buffer_key]
            current_size = self._sizes[buffer_key]

            # Handle overflow
            if current_size + len(data) > self.max_size:
                bytes_to_remove = current_size + len(data) - self.max_size
                bytes_removed = 0

                while bytes_removed < bytes_to_remove:
                    try:
                        removed = buffer.popleft()
                        bytes_removed += len(removed)
                        current_size -= len(removed)
                    except IndexError:  # Handle underflow during overflow handling
                        break

                self._metrics.overflow_count += 1

            # Add new data
            buffer.append(data)
            new_size = current_size + len(data)
            self._sizes[buffer_key] = min(new_size, self.max_size)

            self._last_write[buffer_key] = datetime.now()
            self._metrics.total_bytes_written += len(data)

            # If writing combined stereo, split into channels
            if (
                buffer_key == "combined"
                and self.config.channel_config == ChannelConfig.STEREO
            ):
                await self._split_stereo_channels(data)

            await self._publish_buffer_event("write_complete", len(data))

        except BufferError:
            raise
        except Exception as e:
            raise BufferError(f"Write failed: {str(e)}")

    async def _split_stereo_channels(self, data: bytes) -> None:
        """Split stereo data into separate channel buffers.

        Args:
            data: Stereo audio data
        """
        # Calculate sample size (2 bytes for 16-bit audio)
        sample_size = 2
        frame_size = sample_size * self.config.channels

        # Split the stereo data
        for i in range(0, len(data), frame_size):
            frame = data[i : i + frame_size]
            if len(frame) == frame_size:
                # Left channel (first 2 bytes)
                self._buffers["ch_0"].append(frame[:sample_size])
                self._sizes["ch_0"] += sample_size
                # Right channel (second 2 bytes)
                self._buffers["ch_1"].append(frame[sample_size:])
                self._sizes["ch_1"] += sample_size

    async def read(
        self,
        size: Optional[int] = None,
        channel: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Optional[bytes]:
        """Read audio data from buffer.

        Args:
            size: Number of bytes to read
            channel: Optional channel identifier
            timeout: Optional read timeout in seconds

        Returns:
            Audio data if available, None otherwise
        """
        try:
            read_size = size or self.chunk_size
            buffer_key = channel or "combined"

            if buffer_key not in self._buffers:
                raise BufferError(f"Invalid channel: {buffer_key}")

            buffer = self._buffers[buffer_key]
            current_size = self._sizes[buffer_key]

            # Ensure proper size alignment for stereo
            if buffer_key == "combined":
                read_size = (read_size // (2 * self.config.channels)) * (
                    2 * self.config.channels
                )
            else:
                read_size = (read_size // 2) * 2

            # Wait for data if timeout specified
            if timeout and current_size < read_size:
                start = datetime.now()
                while (
                    current_size < read_size
                    and (datetime.now() - start).total_seconds() < timeout
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

    async def read_stream(self, channel: Optional[str] = None) -> AsyncIterator[bytes]:
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
        return BufferStatus(
            levels={k: (v / self.max_size) * 100 for k, v in self._sizes.items()},
            latencies={k: self._calculate_latency(k) for k in self._buffers.keys()},
            active_channels={k for k, v in self._sizes.items() if v > 0},
            metrics=self._metrics,
        )

    def _calculate_latency(self, channel: str) -> float:
        """Calculate buffer latency in milliseconds.

        Args:
            channel: Channel identifier

        Returns:
            Latency in milliseconds
        """
        bytes_per_frame = 2 * (2 if channel == "combined" else 1)  # 2 bytes per sample
        frames = self._sizes[channel] // bytes_per_frame
        return (frames / self.config.sample_rate) * 1000

    async def _publish_buffer_event(self, status: str, bytes_processed: int) -> None:
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
                    "buffer_status": self.get_status().model_dump(),
                },
            )
        )
