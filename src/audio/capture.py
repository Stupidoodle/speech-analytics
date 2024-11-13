# src/audio/capture.py
import numpy as np
import asyncio
from typing import Optional, Dict, Any, Callable
from .devices import DeviceManager
from .mixer import AudioMixer
from .processor import AudioProcessor
from .exceptions import (
    CaptureError,
    DeviceError,
)
from src.events.types import EventType, Event
from src.events.bus import EventBus


class AudioCapture:
    """
    Main audio capture class that coordinates device management,
    mixing, and processing.
    """

    def __init__(self,
                 event_bus: EventBus,
                 mic_device_id: Optional[int] = None,
                 desktop_device_id: Optional[int] = None,
                 sample_rate: int = 44100,
                 chunk_size: int = 1024,
                 channels: int = 1,
                 ):
        """
        Initialize audio capture.

        Args:
            mic_device_id: ID of microphone device
            desktop_device_id: ID of desktop audio device
            sample_rate: Sample rate for capture
            chunk_size: Size of audio chunks
            channels: Number of channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.event_bus = event_bus

        # Initialize components
        self.device_manager = DeviceManager()
        self.mixer = AudioMixer(sample_rate=sample_rate,
                                chunk_size=chunk_size)
        self.processor = AudioProcessor(sample_rate=sample_rate)

        # Validate and set devices
        if mic_device_id and not self.device_manager.validate_device(
                mic_device_id):
            raise DeviceError(f"Invalid microphone device ID: {mic_device_id}")
        if desktop_device_id and not self.device_manager.validate_device(
                desktop_device_id):
            raise DeviceError(
                f"Invalid desktop device ID: {desktop_device_id}")

        self.mic_device_id = mic_device_id
        self.desktop_device_id = desktop_device_id

        # State management
        self.is_running = False
        self.is_paused = False
        self._streams: Dict[str, Any] = {}
        self._callback: Optional[Callable] = None

    async def start_capture(self,
                            callback: Optional[Callable] = None) -> None:
        """
        Start capturing audio from configured devices.

        Args:
            callback: Optional callback for processed audio data
        """
        if self.is_running:
            raise CaptureError("Capture already running")

        try:
            # Initialize audio streams
            if self.mic_device_id is not None:
                mic_stream = await self._init_mic_stream()
                self._streams['mic'] = mic_stream

            if self.desktop_device_id is not None:
                desktop_stream = await self._init_desktop_stream()
                self._streams['desktop'] = desktop_stream

            self.is_running = True
            self._callback = callback

            # Public event on start
            await self.event_bus.publish(Event(
                type=EventType.AUDIO_CHUNK,
                data={
                    "status": "capture_started",
                }
            ))

            # Start processing loop
            asyncio.create_task(self._process_audio())

        except Exception as e:
            await self.stop_capture()
            raise CaptureError(f"Failed to start capture: {e}")

    async def stop_capture(self) -> None:
        """Stop capturing audio and clean up resources."""
        self.is_running = False
        await self.event_bus.publish(Event(
            type=EventType.AUDIO_CHUNK,
            data={
                "status": "capture_stopped",
            }
        ))
        # Clean up streams
        for stream in self._streams.values():
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")

        self._streams.clear()
        self._callback = None

    async def pause_capture(self) -> None:
        """Pause audio capture."""
        self.is_paused = True

    async def resume_capture(self) -> None:
        """Resume audio capture."""
        self.is_paused = False

    async def _init_mic_stream(self) -> Any:
        """Initialize microphone stream."""
        try:
            mic_stream = await self.device_manager.open_input_stream(
                device_id=self.mic_device_id,
                sample_rate=self.sample_rate,
                channels=self.channels,
                chunk_size=self.chunk_size
            )
            return mic_stream
        except Exception as e:
            raise DeviceError(f"Failed to initialize microphone: {e}")

    async def _init_desktop_stream(self) -> Any:
        """Initialize desktop audio stream."""
        try:
            desktop_stream = await self.device_manager.open_input_stream(
                device_id=self.desktop_device_id,
                sample_rate=self.sample_rate,
                channels=self.channels,
                chunk_size=self.chunk_size,
                as_loopback=True
            )
            return desktop_stream
        except Exception as e:
            raise DeviceError(f"Failed to initialize desktop audio: {e}")

    async def _process_audio(self) -> None:
        """Main audio processing loop."""
        while self.is_running:
            if self.is_paused:
                await asyncio.sleep(0.1)
                continue

            try:
                # Read from streams
                mic_data = await self._read_stream('mic')
                desktop_data = await self._read_stream('desktop')

                # Process audio through mixer
                mixed = self.mixer.prepare_for_transcription(
                    mic_data,
                    desktop_data,
                    self.sample_rate,
                    self.sample_rate
                )

                # Process audio
                processed_combined, info = self.processor.process_chunk(
                    mixed['combined']
                )

                # Create result dictionary
                result = {
                    'combined': processed_combined,
                    'ch_0': mixed['ch_0'],
                    'ch_1': mixed['ch_1'],
                    'info': info
                }

                # Call callback if set
                if self._callback:
                    await self._callback(result)

                # Publish an event for each processed chunk
                await self.event_bus.publish(Event(
                    type=EventType.AUDIO_CHUNK,
                    data={
                        "status": "processed_chunk",
                        "result": result
                    }
                ))

                # Small delay to prevent CPU overload
                await asyncio.sleep(0.001)

            except Exception as e:
                print(f"Error in audio processing: {e}")
                await asyncio.sleep(0.1)

    async def _read_stream(self,
                           stream_name: str) -> Optional[np.ndarray]:
        """Read data from specified stream."""
        if stream_name not in self._streams:
            return None

        try:
            stream = self._streams[stream_name]
            data = await stream.read(self.chunk_size)
            return np.frombuffer(data, dtype=np.int16)
        except Exception as e:
            print(f"Error reading from {stream_name}: {e}")
            return None

    async def get_audio_levels(self) -> Dict[str, float]:
        """Get current audio levels for monitoring."""
        levels = {
            'mic': -float('inf'),
            'desktop': -float('inf')
        }

        for stream_name in ['mic', 'desktop']:
            data = await self._read_stream(stream_name)
            if data is not None:
                data_float = data.astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(np.square(data_float)))
                levels[stream_name] = 20 * np.log10(rms + 1e-10)

        return levels

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.stop_capture())
