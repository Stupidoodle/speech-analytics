"""Audio stream mixing and channel management."""

from dataclasses import asdict
from typing import Optional

import numpy as np

from src.events.bus import EventBus
from src.events.types import Event, EventType

from .exceptions import MixerError
from .types import AudioConfig, AudioMetrics, ProcessingResult


class AudioMixer:
    """Handles mixing of audio streams into stereo format."""

    def __init__(self, event_bus: EventBus, config: AudioConfig) -> None:
        """Initialize audio mixer.

        Args:
            event_bus (EventBus): Event bus instance.
            config (AudioConfig): Audio configuration containing sample rate and channels.
        """
        self.event_bus = event_bus
        self.config = config

    async def mix_streams(
        self, primary: bytes, secondary: Optional[bytes] = None
    ) -> ProcessingResult:
        """Mix two audio streams into a stereo PCM format.

        Args:
            primary (bytes): Primary audio stream (left channel).
            secondary (Optional[bytes]): Secondary audio stream (right channel).

        Returns:
            ProcessingResult: The mixed audio as a stereo PCM stream.

        Raises:
            MixerError: If mixing fails.
        """
        try:
            # Convert audio bytes to float arrays
            primary_arr = self._to_float_array(primary)
            secondary_arr = (
                self._to_float_array(secondary)
                if secondary
                else np.zeros_like(primary_arr)
            )

            # Ensure both streams are of the same length
            min_length = min(len(primary_arr), len(secondary_arr))
            primary_arr = primary_arr[:min_length]
            secondary_arr = secondary_arr[:min_length]

            # Interleave streams into stereo format
            stereo_audio = np.empty((min_length * 2,), dtype=np.float32)
            stereo_audio[0::2] = primary_arr  # Left channel
            stereo_audio[1::2] = secondary_arr  # Right channel

            # Clip values to prevent overflow
            stereo_audio = np.clip(stereo_audio, -1.0, 1.0)

            # Convert back to PCM bytes
            stereo_bytes = (stereo_audio * 32767).astype(np.int16).tobytes()

            # Calculate metrics for the mixed audio
            metrics = self._calculate_metrics(stereo_audio)

            result = ProcessingResult(
                processed_data=stereo_bytes,
                metrics=metrics,
                format=self.config.format,
                sample_rate=self.config.sample_rate,
                channels=2,
                duration=len(stereo_audio) / (self.config.sample_rate * 2),
            )

            # Publish event
            await self._publish_mixer_event("mix_complete", result)
            return result

        except Exception as e:
            raise MixerError(f"Failed to mix streams: {e}")

    @staticmethod
    def _to_float_array(audio_bytes: bytes) -> np.ndarray:
        """Convert PCM audio bytes to normalized float array.

        Args:
            audio_bytes (bytes): PCM audio data.

        Returns:
            np.ndarray: Normalized float array.
        """
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    @staticmethod
    def _calculate_metrics(audio_data: np.ndarray) -> AudioMetrics:
        """Calculate audio metrics for the mixed stream.

        Args:
            audio_data (np.ndarray): Mixed audio data array.

        Returns:
            AudioMetrics: Calculated audio metrics.
        """
        return AudioMetrics(
            peak_level=float(np.max(np.abs(audio_data))),
            rms_level=float(np.sqrt(np.mean(audio_data**2))),
            noise_level=float(np.percentile(np.abs(audio_data), 10)),
            clipping_count=int(np.sum(np.abs(audio_data) > 0.99)),
            dropout_count=int(np.sum(np.abs(audio_data) < 0.01)),
            processing_time=0.0,  # Populated during processing
            buffer_stats={},  # Placeholder for additional stats
        )

    async def _publish_mixer_event(self, status: str, result: ProcessingResult) -> None:
        """Publish an event about the mixing process.

        Args:
            status (str): Status of the event.
            result (ProcessingResult): Mixed audio data and metrics.
        """
        # noinspection PyTypeChecker
        await self.event_bus.publish(
            Event(
                type=EventType.AUDIO_CHUNK,
                data={
                    "status": status,
                    "metrics": asdict(result.metrics),
                    "sample_rate": result.sample_rate,
                    "channels": result.channels,
                    "duration": result.duration,
                },
            )
        )
