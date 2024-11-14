"""Audio device management and configuration."""
from typing import Dict, List, Optional, AsyncIterator
import asyncio
import pyaudiowpatch as pyaudio

from .exceptions import DeviceError, DeviceNotFoundError
from .types import DeviceInfo, DeviceType, AudioConfig


class DeviceManager:
    """Manages audio device discovery and configuration."""

    def __init__(self) -> None:
        """Initialize device manager."""
        self._audio = pyaudio.PyAudio()
        try:
            self._wasapi_info = self._audio.get_host_api_info_by_type(
                pyaudio.paWASAPI
            )
        except Exception as e:
            raise DeviceError(f"Failed to initialize WASAPI: {e}")

    async def list_devices(self) -> List[DeviceInfo]:
        """List available audio devices.

        Returns:
            List of available devices

        Raises:
            DeviceError: If device listing fails
        """
        try:
            devices = []
            for info in self._audio.get_device_info_generator():
                # Skip invalid devices
                if not info or info.get('maxInputChannels', 0) <= 0:
                    continue

                device_type = self._determine_device_type(info)
                devices.append(
                    DeviceInfo(
                        id=info['index'],
                        name=info['name'],
                        type=device_type,
                        channels=info['maxInputChannels'],
                        sample_rate=int(info['defaultSampleRate']),
                        is_default=self._is_default_device(info),
                        is_loopback='Loopback' in info['name'],
                        supports_input=info['maxInputChannels'] > 0,
                        supports_output=info['maxOutputChannels'] > 0
                    )
                )
            return devices

        except Exception as e:
            raise DeviceError(f"Failed to list devices: {e}")

    async def get_default_device(
        self,
        device_type: DeviceType
    ) -> Optional[DeviceInfo]:
        """Get default device of specified type.

        Args:
            device_type: Type of device to get

        Returns:
            Default device if found

        Raises:
            DeviceError: If device query fails
        """
        try:
            if device_type == DeviceType.INPUT:
                return await self._get_default_input()
            elif device_type == DeviceType.OUTPUT:
                return await self._get_default_output()
            else:
                return await self._get_default_loopback()

        except Exception as e:
            raise DeviceError(f"Failed to get default device: {e}")

    async def open_stream(
        self,
        device_id: int,
        config: AudioConfig
    ) -> AsyncIterator[bytes]:
        """Open audio stream for device.

        Args:
            device_id: Device identifier
            config: Stream configuration

        Yields:
            Audio data chunks

        Raises:
            DeviceError: If stream creation fails
            DeviceNotFoundError: If device not found
        """
        try:
            # Validate device
            device_info = self._audio.get_device_info_by_index(device_id)
            if not device_info:
                raise DeviceNotFoundError(
                    device_id,
                    "Device not found"
                )

            # Configure stream
            stream = self._audio.open(
                format=pyaudio.paInt16,
                channels=config.channels,
                rate=config.sample_rate,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=config.chunk_size,
                stream_callback=None
            )

            try:
                while stream.is_active():
                    data = await self._read_stream(stream, config.chunk_size)
                    if data:
                        yield data
                    await asyncio.sleep(0.001)
            finally:
                stream.stop_stream()
                stream.close()

        except DeviceNotFoundError:
            raise
        except Exception as e:
            raise DeviceError(f"Stream creation failed: {e}")

    async def validate_device(
        self,
        device_id: int,
        config: AudioConfig
    ) -> bool:
        """Validate device compatibility.

        Args:
            device_id: Device identifier
            config: Audio configuration

        Returns:
            Whether device is compatible

        Raises:
            DeviceNotFoundError: If device not found
        """
        try:
            device_info = self._audio.get_device_info_by_index(device_id)
            if not device_info:
                raise DeviceNotFoundError(
                    device_id,
                    "Device not found"
                )

            return (
                device_info['maxInputChannels'] >= config.channels and
                self._supports_sample_rate(device_info, config.sample_rate)
            )

        except DeviceNotFoundError:
            raise
        except Exception:
            return False

    def _determine_device_type(self, info: Dict) -> DeviceType:
        """Determine device type from info.

        Args:
            info: Device information

        Returns:
            Determined device type
        """
        if 'Loopback' in info['name']:
            return DeviceType.LOOPBACK
        elif info['maxInputChannels'] > 0:
            return DeviceType.INPUT
        else:
            return DeviceType.OUTPUT

    def _is_default_device(self, info: Dict) -> bool:
        """Check if device is a default device.

        Args:
            info: Device information

        Returns:
            Whether device is default
        """
        try:
            default_input = self._audio.get_default_input_device_info()
            default_output = self._audio.get_default_output_device_info()
            return (
                info['index'] == default_input['index'] or
                info['index'] == default_output['index']
            )
        except Exception:
            return False

    async def _get_default_input(self) -> Optional[DeviceInfo]:
        """Get default input device.

        Returns:
            Default input device if found
        """
        try:
            info = self._audio.get_default_wasapi_device(d_in=True)
            if info and info['maxInputChannels'] > 0:
                return DeviceInfo(
                    id=info['index'],
                    name=info['name'],
                    type=DeviceType.INPUT,
                    channels=info['maxInputChannels'],
                    sample_rate=int(info['defaultSampleRate']),
                    is_default=True
                )
            return None
        except Exception:
            return None

    async def _get_default_output(self) -> Optional[DeviceInfo]:
        """Get default output device.

        Returns:
            Default output device if found
        """
        try:
            info = self._audio.get_default_wasapi_device(d_out=True)
            if info:
                return DeviceInfo(
                    id=info['index'],
                    name=info['name'],
                    type=DeviceType.OUTPUT,
                    channels=info['maxOutputChannels'],
                    sample_rate=int(info['defaultSampleRate']),
                    is_default=True
                )
            return None
        except Exception:
            return None

    async def _get_default_loopback(self) -> Optional[DeviceInfo]:
        """Get default loopback device.

        Returns:
            Default loopback device if found
        """
        try:
            for info in self._audio.get_loopback_device_info_generator():
                return DeviceInfo(
                    id=info['index'],
                    name=info['name'],
                    type=DeviceType.LOOPBACK,
                    channels=info['maxInputChannels'],
                    sample_rate=int(info['defaultSampleRate']),
                    is_default=True,
                    is_loopback=True
                )
            return None
        except Exception:
            return None

    async def _read_stream(
        self,
        stream: pyaudio.Stream,
        chunk_size: int
    ) -> Optional[bytes]:
        """Read from audio stream.

        Args:
            stream: Audio stream
            chunk_size: Size of chunks to read

        Returns:
            Audio data if available
        """
        try:
            return bytes(stream.read(chunk_size))
        except Exception:
            return None

    def _supports_sample_rate(
        self,
        device_info: Dict,
        sample_rate: int
    ) -> bool:
        """Check if device supports sample rate.

        Args:
            device_info: Device information
            sample_rate: Sample rate to check

        Returns:
            Whether rate is supported
        """
        try:
            return abs(
                device_info['defaultSampleRate'] - sample_rate
            ) < 100
        except Exception:
            return False

    async def __aenter__(self) -> 'DeviceManager':
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        self._audio.terminate()