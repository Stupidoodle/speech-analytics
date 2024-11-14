"""Audio stream mixing and channel management."""
from typing import Dict, Optional, Tuple

import numpy as np
import librosa

from src.events.bus import EventBus
from src.events.types import Event, EventType
from .exceptions import MixerError
from .types import AudioConfig, ProcessingResult, AudioMetrics


class AudioMixer:
    """Handles mixing and processing of audio streams."""

    def __init__(
        self,
        event_bus: EventBus,
        config: AudioConfig,
    ) -> None:
        """Initialize audio mixer.

        Args:
            event_bus: Event bus instance
            config: Audio configuration
        """
        self.event_bus = event_bus
        self.config = config
        self.target_rate = 16000  # Target sample rate for transcription

    async def mix_streams(
        self,
        primary: bytes,
        secondary: Optional[bytes] = None,
        primary_gain: float = 1.0,
        secondary_gain: float = 0.7
    ) -> ProcessingResult:
        """Mix two audio streams with gain control.

        Args:
            primary: Primary audio stream
            secondary: Optional secondary stream
            primary_gain: Gain for primary stream
            secondary_gain: Gain for secondary stream

        Returns:
            Mixed audio result

        Raises:
            MixerError: If mixing fails
        """
        try:
            # Convert to float arrays
            primary_arr = self._to_float_array(primary)
            if secondary is not None:
                secondary_arr = self._to_float_array(secondary)
                # Ensure same length
                length = min(len(primary_arr), len(secondary_arr))
                primary_arr = primary_arr[:length]
                secondary_arr = secondary_arr[:length]
            else:
                secondary_arr = None

            # Apply gain and mix
            primary_arr *= primary_gain
            if secondary_arr is not None:
                secondary_arr *= secondary_gain
                mixed = primary_arr + secondary_arr
            else:
                mixed = primary_arr

            # Clip to prevent overflow
            mixed = np.clip(mixed, -1.0, 1.0)

            # Convert back to bytes
            mixed_bytes = (mixed * 32767).astype(np.int16).tobytes()

            metrics = self._calculate_metrics(mixed)

            # Create result
            result = ProcessingResult(
                processed_data=mixed_bytes,
                metrics=metrics,
                format=self.config.format,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                duration=len(mixed) / self.config.sample_rate
            )

            await self._publish_mixer_event("mix_complete", result)
            return result

        except Exception as e:
            raise MixerError(f"Failed to mix streams: {e}")

    async def prepare_for_transcription(
        self,
        audio_data: bytes,
    ) -> Tuple[bytes, AudioMetrics]:
        """Prepare audio for transcription.

        Args:
            audio_data: Raw audio data

        Returns:
            Processed audio and metrics

        Raises:
            MixerError: If processing fails
        """
        try:
            # Convert to float array
            float_data = self._to_float_array(audio_data)

            # Resample if needed
            if self.config.sample_rate != self.target_rate:
                float_data = librosa.resample(
                    float_data,
                    orig_sr=self.config.sample_rate,
                    target_sr=self.target_rate
                )

            # Convert to mono if needed
            if len(float_data.shape) > 1:
                float_data = float_data.mean(axis=1)

            # Normalize
            float_data = librosa.util.normalize(float_data)

            # Convert back to bytes
            processed = (float_data * 32767).astype(np.int16).tobytes()

            metrics = self._calculate_metrics(float_data)

            await self._publish_mixer_event(
                "transcription_prepared",
                ProcessingResult(
                    processed_data=processed,
                    metrics=metrics,
                    format=self.config.format,
                    sample_rate=self.target_rate,
                    channels=1,
                    duration=len(float_data) / self.target_rate
                )
            )

            return processed, metrics

        except Exception as e:
            raise MixerError(f"Failed to prepare for transcription: {e}")

    def _to_float_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to float array.

        Args:
            audio_bytes: Raw audio data

        Returns:
            Normalized float array
        """
        return np.frombuffer(
            audio_bytes, dtype=np.int16
        ).astype(np.float32) / 32768.0

    def _calculate_metrics(self, audio_data: np.ndarray) -> AudioMetrics:
        """Calculate audio metrics.

        Args:
            audio_data: Audio data array

        Returns:
            Calculated metrics
        """
        return AudioMetrics(
            peak_level=float(np.max(np.abs(audio_data))),
            rms_level=float(np.sqrt(np.mean(audio_data**2))),
            noise_level=float(np.percentile(np.abs(audio_data), 10)),
            clipping_count=int(np.sum(np.abs(audio_data) > 0.99)),
            dropout_count=int(np.sum(np.abs(audio_data) < 0.01)),
            processing_time=0.0,  # Populated by caller
            buffer_stats={}  # Populated by caller
        )

    async def _publish_mixer_event(
        self,
        status: str,
        result: ProcessingResult
    ) -> None:
        """Publish mixer event.

        Args:
            status: Event status
            result: Processing result
        """
        await self.event_bus.publish(
            Event(
                type=EventType.AUDIO_CHUNK,
                data={
                    "status": status,
                    # FIXME
                    "metrics": result.metrics.dict(),
                    "sample_rate": result.sample_rate,
                    "channels": result.channels,
                    "duration": result.duration
                }
            )
        )