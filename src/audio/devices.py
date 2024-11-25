"""Enhanced Device Manager with pyaudiowpatch support."""

import asyncio
from typing import AsyncIterator, Dict, List, Optional

import pyaudiowpatch as pyaudio

from .exceptions import DeviceError
from .types import AudioConfig, DeviceInfo, DeviceType


class DeviceManager:
    """Manages audio device discovery, validation, and streaming with pyaudiowpatch."""

    def __init__(self) -> None:
        """Initializes the device manager with WASAPI support.

        Raises:
            DeviceError: If WASAPI initialization fails.
        """
        try:
            with pyaudio.PyAudio() as audio_manager:
                self._wasapi_info = audio_manager.get_host_api_info_by_type(
                    pyaudio.paWASAPI
                )
        except Exception as e:
            raise DeviceError(
                "Failed to initialize WASAPI. Ensure it is supported.",
                {"error": str(e)},
            )

    async def list_devices(self) -> List[DeviceInfo]:
        """Lists all available audio devices, including loopback devices.

        Returns:
            List[DeviceInfo]: A list of available audio devices.

        Raises:
            DeviceError: If device listing fails.
        """
        try:
            devices = []
            with pyaudio.PyAudio() as audio_manager:
                print("Debug: PyAudio instance created")
                for info in audio_manager.get_device_info_generator():
                    if info:
                        devices.append(self._parse_device_info(info))
            return devices
        except Exception as e:
            raise DeviceError("Failed to list devices.", {"error": str(e)})

    async def get_default_stereo_devices(self) -> Dict[str, DeviceInfo]:
        """Retrieves the default microphone and desktop loopback devices.

        Returns:
            Dict[str, DeviceInfo]: A dictionary containing the default microphone
            and desktop loopback devices.

        Raises:
            DeviceError: If default devices cannot be retrieved or validated.
        """
        try:
            with pyaudio.PyAudio() as audio_manager:
                mic_device = self._get_default_device(audio_manager, DeviceType.INPUT)
                desktop_device = self._get_default_device(
                    audio_manager, DeviceType.OUTPUT
                )

                if not desktop_device.get("isLoopbackDevice", False):
                    desktop_device = self._find_loopback_device(
                        audio_manager, desktop_device
                    )

                if not mic_device or not desktop_device:
                    raise DeviceError("Failed to locate default stereo devices.")

                return {
                    "mic": self._parse_device_info(mic_device),
                    "desktop": self._parse_device_info(desktop_device),
                }
        except Exception as e:
            raise DeviceError(
                "Failed to retrieve default stereo devices.", {"error": str(e)}
            )

    @staticmethod
    async def open_stream(
        device: DeviceInfo, config: AudioConfig
    ) -> AsyncIterator[bytes]:
        """Opens an audio stream for the specified device.

        Args:
            device (DeviceInfo): The device to open a stream for.
            config (AudioConfig): Configuration for the audio stream.

        Yields:
            AsyncIterator[bytes]: The audio stream as chunks of bytes.

        Raises:
            DeviceError: If the stream cannot be opened.
        """
        try:
            with pyaudio.PyAudio() as audio_manager:
                stream = None
                try:
                    stream = audio_manager.open(
                        format=pyaudio.paInt16,
                        channels=config.channels,
                        rate=config.sample_rate,
                        input=True,
                        input_device_index=device.id,
                        frames_per_buffer=config.chunk_size,
                    )
                    while stream.is_active():
                        yield await asyncio.to_thread(stream.read, config.chunk_size)
                finally:
                    if stream:
                        stream.close()
        except Exception as e:
            raise DeviceError(
                f"Failed to open stream for device {device.name}.", {"error": str(e)}
            )

    @staticmethod
    def _get_default_device(
        audio_manager: pyaudio.PyAudio, device_type: DeviceType
    ) -> Optional[Dict]:
        """Retrieves the default device of the specified type.

        Args:
            audio_manager (pyaudio.PyAudio): The PyAudio instance.
            device_type (DeviceType): The type of device (INPUT, OUTPUT, LOOPBACK).

        Returns:
            Optional[Dict]: Information about the default device, or None if not found.
        """
        try:
            if device_type == DeviceType.INPUT:
                return audio_manager.get_default_wasapi_device(d_in=True)
            elif device_type == DeviceType.OUTPUT:
                return audio_manager.get_default_wasapi_device(d_out=True)
            return None
        except Exception:
            return None

    @staticmethod
    def _find_loopback_device(
        audio_manager: pyaudio.PyAudio, device_info: Dict
    ) -> Optional[Dict]:
        """Finds the loopback device corresponding to the default output device.

        Args:
            audio_manager (pyaudio.PyAudio): The PyAudio instance.
            device_info (Dict): Information about the default output device.

        Returns:
            Optional[Dict]: The loopback device, or None if not found.
        """
        for loopback in audio_manager.get_loopback_device_info_generator():
            if device_info["name"] in loopback["name"]:
                return loopback
        return None

    def _parse_device_info(self, info: Dict) -> DeviceInfo:
        """Parses raw device information into a structured DeviceInfo object.

        Args:
            info (Dict): Raw device information.

        Returns:
            DeviceInfo: Parsed device information.
        """
        return DeviceInfo(
            id=info["index"],
            name=info["name"],
            type=self._determine_device_type(info),
            channels=info["maxInputChannels"] if "maxInputChannels" in info else 0,
            sample_rate=int(info["defaultSampleRate"]),
            is_default=info.get("isDefault", False),
            is_loopback=info.get("isLoopbackDevice", False),
            supports_input=info.get("maxInputChannels", 0) > 0,
            supports_output=info.get("maxOutputChannels", 0) > 0,
        )

    @staticmethod
    def _determine_device_type(info: Dict) -> DeviceType:
        """Determines the type of device based on its properties.

        Args:
            info (Dict): Raw device information.

        Returns:
            DeviceType: The determined device type (INPUT, OUTPUT, LOOPBACK).
        """
        if info.get("isLoopbackDevice", False):
            return DeviceType.LOOPBACK
        elif info.get("maxInputChannels", 0) > 0:
            return DeviceType.INPUT
        else:
            return DeviceType.OUTPUT
