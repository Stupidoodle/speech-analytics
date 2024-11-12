from collections import deque
from typing import Optional, Deque, Tuple
import time
from .exceptions import (
    BufferOverflowError,
    BufferUnderrunError,
    InvalidTranscriptionDataError
)


class AudioBuffer:
    """
    Manages audio buffering for transcription with proper timing and chunking.
    Ensures audio data is correctly formatted and timed for AWS Transcribe.
    """

    def __init__(self,
                 max_size: int = 32768,
                 chunk_size: int = 8192,
                 sample_rate: int = 16000):
        """
        Initialize the audio buffer.

        Args:
            max_size: Maximum buffer size in bytes
            chunk_size: Size of chunks to return in bytes
            sample_rate: Audio sample rate
        """
        self.max_size = max_size
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

        # Calculate bytes per sample (2 channels * 2 bytes per sample)
        self.bytes_per_sample = 4

        # Initialize buffer
        self._buffer: Deque[bytes] = deque()
        self._buffer_size = 0

        # Timing management
        self._last_write_time = time.time()
        self._last_read_time = time.time()

    def write(self, data: bytes) -> None:
        """
        Write audio data to the buffer.

        Args:
            data: Raw audio bytes to add to buffer

        Raises:
            BufferOverflowError: If buffer would exceed max size
            InvalidTranscriptionDataError: If data format is invalid
        """
        # Validate data
        if len(data) % self.bytes_per_sample != 0:
            raise InvalidTranscriptionDataError(
                "Data length must be a multiple of 4 bytes"
            )

        # Check for overflow
        if self._buffer_size + len(data) > self.max_size:
            raise BufferOverflowError(
                f"Buffer overflow. Current: {self._buffer_size}, "
                f"Adding: {len(data)}, Max: {self.max_size}"
            )

        self._buffer.append(data)
        self._buffer_size += len(data)
        self._last_write_time = time.time()

    def read(self, size: Optional[int] = None) -> bytes:
        """
        Read audio data from the buffer.

        Args:
            size: Number of bytes to read (defaults to chunk_size)

        Returns:
            Raw audio bytes

        Raises:
            BufferUnderrunError: If not enough data available
        """
        read_size = size if size is not None else self.chunk_size

        # Check for underrun
        if self._buffer_size < read_size:
            raise BufferUnderrunError(
                f"Buffer underrun. Available: {self._buffer_size}, "
                f"Requested: {read_size}"
            )

        # Combine data from buffer
        data = bytes()
        remaining = read_size

        while remaining > 0 and self._buffer:
            chunk = self._buffer.popleft()
            if len(chunk) <= remaining:
                data += chunk
                remaining -= len(chunk)
                self._buffer_size -= len(chunk)
            else:
                # Split chunk if needed
                data += chunk[:remaining]
                self._buffer.appendleft(chunk[remaining:])
                self._buffer_size -= remaining
                remaining = 0

        self._last_read_time = time.time()
        return data

    def get_available_bytes(self) -> int:
        """Get number of bytes available in buffer."""
        return self._buffer_size

    def get_buffer_level(self) -> float:
        """Get buffer level as percentage."""
        return (self._buffer_size / self.max_size) * 100

    def get_latency(self) -> float:
        """Get current buffer latency in milliseconds."""
        return (self._buffer_size / self.bytes_per_sample /
                self.sample_rate * 1000)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._buffer_size = 0

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._buffer_size == 0

    def get_timing_info(self) -> Tuple[float, float, float]:
        """
        Get timing information about the buffer.

        Returns:
            Tuple of (write_age, read_age, total_latency) in seconds
        """
        now = time.time()
        write_age = now - self._last_write_time
        read_age = now - self._last_read_time
        total_latency = self.get_latency() / 1000  # Convert ms to seconds

        return write_age, read_age, total_latency


class StreamBuffer:
    """
    Manages streaming buffer for real-time transcription with automatic
    rate limiting and chunk optimization.
    """

    def __init__(self,
                 target_latency: float = 100.0,  # ms
                 max_latency: float = 500.0,     # ms
                 chunk_duration: float = 100.0,   # ms
                 sample_rate: int = 16000):
        """
        Initialize the streaming buffer.

        Args:
            target_latency: Target buffer latency in milliseconds
            max_latency: Maximum allowable latency in milliseconds
            chunk_duration: Duration of each chunk in milliseconds
            sample_rate: Audio sample rate
        """
        self.target_latency = target_latency
        self.max_latency = max_latency
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate

        # Calculate sizes
        bytes_per_chunk = int(
            chunk_duration / 1000 * sample_rate * 4  # 4 bytes per sample
        )
        max_size = int(
            max_latency / 1000 * sample_rate * 4
        )

        self.buffer = AudioBuffer(
            max_size=max_size,
            chunk_size=bytes_per_chunk,
            sample_rate=sample_rate
        )

        # Rate limiting state
        self._last_chunk_time = time.time()

    async def write(self, data: bytes) -> None:
        """
        Write audio data to the streaming buffer.

        Args:
            data: Raw audio bytes
        """
        try:
            self.buffer.write(data)
        except BufferOverflowError:
            # If buffer is full, drop oldest data
            while (self.buffer.get_buffer_level() > 80 and
                   self.buffer.get_available_bytes() + len(data) >
                   self.buffer.max_size):
                try:
                    self.buffer.read()
                except BufferUnderrunError:
                    break
            self.buffer.write(data)

    async def read(self) -> Optional[bytes]:
        """
        Read a chunk from the streaming buffer with rate limiting.

        Returns:
            Audio chunk or None if not enough time has elapsed
        """
        # Check if enough time has elapsed since last chunk
        elapsed = (time.time() - self._last_chunk_time) * 1000
        if elapsed < self.chunk_duration:
            return None

        # Check if we have enough data
        if self.buffer.get_available_bytes() < self.buffer.chunk_size:
            return None

        try:
            chunk = self.buffer.read()
            self._last_chunk_time = time.time()
            return chunk
        except BufferUnderrunError:
            return None

    def get_status(self) -> dict:
        """Get current buffer status."""
        write_age, read_age, latency = self.buffer.get_timing_info()
        return {
            'buffer_level': self.buffer.get_buffer_level(),
            'latency_ms': self.buffer.get_latency(),
            'write_age_ms': write_age * 1000,
            'read_age_ms': read_age * 1000,
            'is_healthy': (
                self.buffer.get_latency() <= self.max_latency and
                not self.buffer.is_empty()
            )
        }
