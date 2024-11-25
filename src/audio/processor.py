"""Audio signal processing and enhancement with librosa Resampling."""

import time
from typing import Dict, Optional

import librosa
import numpy as np

from src.events.bus import EventBus
from src.events.types import Event, EventType

from .exceptions import ProcessingError
from .types import AudioConfig, AudioMetrics, ProcessingResult


class AudioProcessor:
    """Handles audio processing and enhancement."""

    def __init__(
        self,
        event_bus: EventBus,
        config: AudioConfig,
        noise_threshold: float = 0.01,
        gain_target: float = 0.7,
    ) -> None:
        """Initialize audio processor.

        Args:
            event_bus: Event bus instance
            config: Audio configuration
            noise_threshold: Noise gate threshold
            gain_target: Target gain level
        """
        self.event_bus = event_bus
        self.config = config
        self.noise_threshold = noise_threshold
        self.gain_target = gain_target

        # Processing state
        self.noise_profile: Optional[np.ndarray] = None
        self._running_max = 0.0
        self._calibrated = False

    async def process_chunk(
        self,
        audio_data: bytes,
        apply_noise_reduction: bool = True,
        apply_gain: bool = True,
    ) -> ProcessingResult:
        """Process audio chunk.

        Args:
            audio_data: Raw audio data
            apply_noise_reduction: Whether to apply noise reduction
            apply_gain: Whether to apply automatic gain

        Returns:
            Processing result

        Raises:
            ProcessingError: If processing fails
        """
        try:
            start_time = time.time()

            # Convert to float array
            float_data = self._to_float_array(audio_data)

            # Resample if needed
            if self.config.sample_rate != 16000:
                float_data = await self._resample(float_data)

            # Convert to mono if there are multiple channels
            if self.config.channels > 1:
                float_data = float_data.mean(axis=0).astype(float_data.dtype)
                self.config.channels = 1

            # Apply processing steps
            processed = float_data
            if apply_noise_reduction and self._calibrated:
                processed = await self._reduce_noise(processed)
            if apply_gain:
                processed = await self._apply_gain(processed)

            # Calculate metrics
            metrics = self._calculate_metrics(processed, time.time() - start_time)

            # Convert back to bytes
            processed_bytes = (processed * 32767).astype(np.int16).tobytes()

            result = ProcessingResult(
                processed_data=processed_bytes,
                metrics=metrics,
                format=self.config.format,
                sample_rate=16000,
                channels=self.config.channels,
                duration=len(processed_bytes) / (16000 * 2),
            )

            await self._publish_processor_event(
                "processing_complete", result.model_dump()
            )
            return result

        except Exception as e:
            raise ProcessingError(f"Processing failed: {e}")

    async def calibrate_noise(self, audio_data: bytes) -> None:
        """Calibrate noise profile from audio sample.

        Args:
            audio_data: Audio sample for calibration

        Raises:
            ProcessingError: If calibration fails
        """
        try:
            # Convert to float array
            float_data = self._to_float_array(audio_data)

            # Calculate noise profile
            self.noise_profile = np.mean(np.abs(float_data))
            self._calibrated = True

            await self._publish_processor_event(
                "calibration_complete",
                {
                    "noise_profile": float(self.noise_profile),
                    "threshold": self.noise_threshold,
                },
            )

        except Exception as e:
            raise ProcessingError(f"Calibration failed: {e}")

    async def _resample(self, audio_data: np.ndarray) -> np.ndarray:
        """Resample audio data to 16,000 Hz using librosa.

        Args:
            audio_data: Audio data array

        Returns:
            Resampled audio data
        """
        resampled_audio = librosa.resample(
            audio_data, orig_sr=self.config.sample_rate, target_sr=16000
        )
        return resampled_audio

    async def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise reduction.

        Args:
            audio_data: Audio data array

        Returns:
            Noise reduced audio
        """
        if self.noise_profile is None:
            return audio_data

        # Apply spectral gating
        mask = np.abs(audio_data) > (self.noise_profile * self.noise_threshold)
        return audio_data * mask

    async def _apply_gain(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply automatic gain control.

        Args:
            audio_data: Audio data array

        Returns:
            Gain adjusted audio
        """
        # Update running maximum
        current_max = np.max(np.abs(audio_data))
        self._running_max = max(current_max, self._running_max * 0.95)

        if self._running_max > 0.01:
            # Calculate gain adjustment
            gain = self.gain_target / self._running_max
            audio_data *= gain

        return np.clip(audio_data, -1.0, 1.0)

    @staticmethod
    def _to_float_array(audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to float array.

        Args:
            audio_bytes: Raw audio data

        Returns:
            Normalized float array
        """
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def _calculate_metrics(
        self, audio_data: np.ndarray, processing_time: float
    ) -> AudioMetrics:
        """Calculate audio metrics.

        Args:
            audio_data: Audio data array
            processing_time: Processing duration

        Returns:
            Audio metrics
        """
        return AudioMetrics(
            peak_level=float(np.max(np.abs(audio_data))),
            rms_level=float(np.sqrt(np.mean(audio_data**2))),
            noise_level=float(self.noise_profile if self._calibrated else 0.0),
            clipping_count=int(np.sum(np.abs(audio_data) > 0.99)),
            dropout_count=int(np.sum(np.abs(audio_data) < 0.01)),
            processing_time=processing_time,
            buffer_stats={"running_max": float(self._running_max)},
        )

    async def _publish_processor_event(self, status: str, data: Dict) -> None:
        """Publish processor event.

        Args:
            status: Event status
            data: Event data
        """
        await self.event_bus.publish(
            Event(
                type=EventType.AUDIO_CHUNK,
                data={"status": status, "processor_data": data},
            )
        )
