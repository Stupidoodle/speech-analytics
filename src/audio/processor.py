# src/audio/processor.py
import numpy as np
from typing import Optional, Tuple
from .exceptions import ProcessingError


class AudioProcessor:
    """
    Handles audio processing tasks like noise reduction, normalization,
    and gain control.
    """

    def __init__(self,
                 noise_threshold: float = 0.01,
                 gain: float = 1.0,
                 calibration_duration: float = 1.0,
                 sample_rate: int = 16000):
        """
        Initialize the audio processor.

        Args:
            noise_threshold: Threshold for noise gate (0.0 to 1.0)
            gain: Audio gain multiplier
            calibration_duration: Duration in seconds for noise profile
            sample_rate: Audio sample rate
        """
        self.noise_threshold = noise_threshold
        self.gain = gain
        self.calibration_duration = calibration_duration
        self.sample_rate = sample_rate

        # State variables
        self.noise_profile: Optional[np.ndarray] = None
        self.is_calibrated = False
        self._running_max = 0.0
        self._silence_counter = 0

    def calibrate_noise(self, audio: np.ndarray) -> None:
        """
        Calibrate noise profile from audio sample.

        Args:
            audio: numpy array of audio data
        """
        try:
            # Convert to float32 for processing
            audio_float = audio.astype(np.float32) / 32768.0

            # Calculate noise profile using mean and std
            self.noise_profile = np.mean(np.abs(audio_float))
            self.is_calibrated = True

        except Exception as e:
            raise ProcessingError(f"Failed to calibrate noise profile: {e}")

    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction using spectral gating.

        Args:
            audio: numpy array of audio data

        Returns:
            Noise-reduced audio data
        """
        try:
            # Convert to float32 for processing
            audio_float = audio.astype(np.float32) / 32768.0

            if self.is_calibrated and self.noise_profile is not None:
                # Apply noise gate
                mask = np.abs(audio_float) > (self.noise_profile *
                                              self.noise_threshold)
                audio_float *= mask

            # Convert back to int16
            return (audio_float * 32767).astype(np.int16)

        except Exception as e:
            raise ProcessingError(f"Failed to reduce noise: {e}")

    def normalize(self,
                  audio: np.ndarray,
                  target_peak: float = 0.95
                  ) -> Tuple[np.ndarray, float]:
        """
        Normalize audio to target peak amplitude.

        Args:
            audio: numpy array of audio data
            target_peak: Target peak amplitude (0.0 to 1.0)

        Returns:
            Tuple of (normalized audio, peak amplitude)
        """
        try:
            # Convert to float32 for processing
            audio_float = audio.astype(np.float32) / 32768.0

            # Calculate current peak
            current_peak = np.max(np.abs(audio_float))

            # Update running maximum with decay
            self._running_max = max(current_peak,
                                    self._running_max * 0.95)

            if self._running_max > 0:
                # Calculate normalization factor
                norm_factor = target_peak / self._running_max
                # Apply normalization
                audio_float *= norm_factor

            # Clip to prevent overflow
            audio_float = np.clip(audio_float, -1.0, 1.0)

            # Convert back to int16
            return ((audio_float * 32767).astype(np.int16),
                    self._running_max)

        except Exception as e:
            raise ProcessingError(f"Failed to normalize audio: {e}")

    def apply_gain(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply gain to audio data.

        Args:
            audio: numpy array of audio data

        Returns:
            Gain-adjusted audio data
        """
        try:
            # Convert to float32 for processing
            audio_float = audio.astype(np.float32) / 32768.0

            # Apply gain
            audio_float *= self.gain

            # Clip to prevent overflow
            audio_float = np.clip(audio_float, -1.0, 1.0)

            # Convert back to int16
            return (audio_float * 32767).astype(np.int16)

        except Exception as e:
            raise ProcessingError(f"Failed to apply gain: {e}")

    def detect_silence(self,
                       audio: np.ndarray,
                       threshold: float = 0.01,
                       min_duration: float = 0.5
                       ) -> bool:
        """
        Detect if audio chunk is silence.

        Args:
            audio: numpy array of audio data
            threshold: Amplitude threshold for silence detection
            min_duration: Minimum duration of silence (seconds)

        Returns:
            True if audio is considered silence
        """
        try:
            # Convert to float32 for processing
            audio_float = audio.astype(np.float32) / 32768.0

            # Calculate RMS amplitude
            rms = np.sqrt(np.mean(np.square(audio_float)))

            # Update silence counter
            if rms < threshold:
                self._silence_counter += len(audio) / self.sample_rate
            else:
                self._silence_counter = 0

            return self._silence_counter >= min_duration

        except Exception as e:
            raise ProcessingError(f"Failed to detect silence: {e}")

    def process_chunk(self,
                      audio: np.ndarray,
                      apply_noise_reduction: bool = True,
                      apply_normalization: bool = True,
                      apply_gain_control: bool = True
                      ) -> Tuple[np.ndarray, dict]:
        """
        Process an audio chunk with selected processors.

        Args:
            audio: numpy array of audio data
            apply_noise_reduction: Whether to apply noise reduction
            apply_normalization: Whether to apply normalization
            apply_gain_control: Whether to apply gain control

        Returns:
            Tuple of (processed audio, processing info)
        """
        try:
            processed = audio.copy()
            info = {
                'peak_amplitude': 0.0,
                'is_silence': False,
                'applied_gain': self.gain
            }

            if apply_noise_reduction and self.is_calibrated:
                processed = self.reduce_noise(processed)

            if apply_normalization:
                processed, peak = self.normalize(processed)
                info['peak_amplitude'] = peak

            if apply_gain_control:
                processed = self.apply_gain(processed)

            info['is_silence'] = self.detect_silence(processed)

            return processed, info

        except Exception as e:
            raise ProcessingError(f"Failed to process audio chunk: {e}")
