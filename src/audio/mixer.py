import numpy as np
from typing import Optional, Dict
import librosa
from src.events.types import EventType, Event
from src.events.bus import EventBus
import asyncio

class AudioMixer:
    """
    Handles mixing and processing of audio streams while maintaining channel
    separation for transcription purposes.
    """

    def __init__(self, event_bus: EventBus, sample_rate: int = 44100, chunk_size: int = 1024):
        self.event_bus = event_bus
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.target_sample_rate = 16000  # AWS Transcribe preferred rate

    def _resample(self,
                  audio_data: np.ndarray,
                  original_rate: int
                  ) -> np.ndarray:
        """Resample audio to target sample rate for transcription"""
        if original_rate == self.target_sample_rate:
            return audio_data

        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        resampled_data = librosa.resample(audio_data,
                                          orig_sr=original_rate,
                                          target_sr=self.target_sample_rate
                                          )

        resampled_data = np.clip(resampled_data,
                                 -32768,
                                 32767
                                 )
        resampled_data = resampled_data.astype(np.int16)

        return resampled_data

    def prepare_for_transcription(
        self,
        mic_data: Optional[np.ndarray],
        desktop_data: Optional[np.ndarray],
        mic_rate: int,
        desktop_rate: int
    ) -> Dict[str, np.ndarray]:
        """
        Prepares audio data for transcription by:
        1. Resampling to 16kHz if needed
        2. Converting to correct format (16-bit PCM)
        3. Organizing channels properly

        Returns a dict with:
        - 'combined': Mixed audio for recording/monitoring
        - 'ch_0': Microphone audio (resampled)
        - 'ch_1': Desktop audio (resampled)
        """
        # Handle none cases
        if mic_data is None and desktop_data is None:
            return {
                'combined': np.zeros(0, dtype=np.int16),
                'ch_0': np.zeros(0, dtype=np.int16),
                'ch_1': np.zeros(0, dtype=np.int16)
            }

        # Process microphone audio
        if mic_data is not None:
            mic_processed = self._resample(mic_data, mic_rate)
            # Ensure it's mono
            if len(mic_processed.shape) > 1:
                mic_processed = mic_processed.mean(axis=1)
        else:
            mic_processed = np.zeros(self.chunk_size, dtype=np.int16)

        # Process desktop audio
        if desktop_data is not None:
            desktop_processed = self._resample(desktop_data, desktop_rate)
            # Ensure it's mono
            if len(desktop_processed.shape) > 1:
                desktop_processed = desktop_processed.mean(axis=1)
        else:
            desktop_processed = np.zeros(self.chunk_size, dtype=np.int16)

        # Make sure both channels have the same length
        target_length = max(len(mic_processed), len(desktop_processed))
        if len(mic_processed) < target_length:
            mic_processed = np.pad(mic_processed,
                                   (0,
                                    target_length - len(mic_processed)
                                    )
                                   )
        if len(desktop_processed) < target_length:
            desktop_processed = np.pad(desktop_processed,
                                       (0,
                                        target_length - len(desktop_processed)
                                        )
                                       )

        # Create mixed version for recording/monitoring
        mixed = (mic_processed.astype(np.float32) +
                 desktop_processed.astype(np.float32)
                 ) / 2
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

        # Publish an event after preparation for transcription
        asyncio.get_event_loop().run_until_complete(self.event_bus.publish(Event(
            type=EventType.AUDIO_CHUNK,
            data={
                "status": "ready_for_transcription",
                "channels": {
                    'mic': mic_processed,
                    'desktop': desktop_processed
                }
            }
        )))

        return {
            'combined': mixed,
            'ch_0': mic_processed.astype(np.int16),
            'ch_1': desktop_processed.astype(np.int16)
        }

    def create_transcription_chunk(self,
                                   channels: Dict[str, np.ndarray]
                                   ) -> bytes:
        """
        Creates a properly formatted audio chunk for AWS Transcribe.
        For dual-channel PCM, samples are interleaved: LRLRLR...
        """
        # Ensure we have both channels
        ch0 = channels.get('ch_0', np.zeros(0, dtype=np.int16))
        ch1 = channels.get('ch_1', np.zeros(0, dtype=np.int16))

        # Interleave channels
        chunk_length = max(len(ch0), len(ch1))
        if len(ch0) < chunk_length:
            ch0 = np.pad(ch0, (0, chunk_length - len(ch0)))
        if len(ch1) < chunk_length:
            ch1 = np.pad(ch1, (0, chunk_length - len(ch1)))

        # Stack and reshape to interleave
        interleaved = np.column_stack((ch0, ch1)).ravel()

        # Convert to bytes
        return interleaved.tobytes()

    def get_chunk_duration(self, chunk: bytes) -> float:
        """Calculate the duration of an audio chunk in milliseconds"""
        # For 16-bit dual-channel audio,
        # each sample is 4 bytes (2 bytes per channel)
        num_samples = len(chunk) // 4
        return (num_samples / self.target_sample_rate) * 1000  # Convert to ms
