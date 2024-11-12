from typing import Dict, List, Optional
import pyaudiowpatch as pyaudio
from .exceptions import DeviceError


class DeviceManager:
    """Manages audio devices and their properties"""

    def __init__(self):
        self._audio = pyaudio.PyAudio()
        try:
            self._wasapi_info = \
                self._audio.get_host_api_info_by_type(pyaudio.paWASAPI)
        except Exception as e:
            raise DeviceError(f"Failed to initialize WASAPI: {e}")

    def list_devices(self) -> List[Dict]:
        """List all available audio devices"""
        devices = []
        try:
            for device_info in self._audio.get_device_info_generator():
                if device_info['maxInputChannels'] > 0:
                    devices.append({
                        'id': device_info['index'],
                        'name': device_info['name'],
                        'input_channels': device_info['maxInputChannels'],
                        'sample_rate': int(device_info['defaultSampleRate']),
                        'is_wasapi': device_info['hostApi'] ==
                        self._wasapi_info['index'],
                        'is_loopback': 'Loopback' in device_info['name']
                    })
            return devices
        except Exception as e:
            raise DeviceError(f"Failed to list audio devices: {e}")

    def get_default_input_device(self) -> Optional[Dict]:
        """Get the default input device using WASAPI helper"""
        try:
            device_info = self._audio.get_default_wasapi_device(d_in=True)
            if device_info and device_info['maxInputChannels'] > 0:
                return {
                    'id': device_info['index'],
                    'name': device_info['name'],
                    'input_channels': device_info['maxInputChannels'],
                    'sample_rate': int(device_info['defaultSampleRate'])
                }
        except Exception as e:
            print(f"Warning: Could not get default input device: {e}")
        return None

    def get_default_output_device(self) -> Optional[Dict]:
        """
        Get the default output device
        using WASAPI helper with loopback validation
        """
        try:
            device_info = self._audio.get_default_wasapi_device(d_out=True)
            if device_info:
                # We can safely assume that if it is a loopback device it has
                # input channels probably
                if not device_info["isLoopbackDevice"]:
                    for loopback in\
                            self._audio.get_loopback_device_info_generator():
                        if device_info["name"] in loopback["name"]:
                            return {
                                "id": loopback["index"],
                                "name": loopback["name"],
                                "input_channels": loopback["maxInputChannels"],
                                "sample_rate": int(
                                    loopback["defaultSampleRate"]
                                    ),
                            }
                else:
                    return {
                        "id": device_info["index"],
                        "name": device_info["name"],
                        "input_channels": device_info["maxInputChannels"],
                        "sample_rate": int(device_info["defaultSampleRate"]),
                    }
        except Exception as e:
            print(f"Warning: Could not get default output device: {e}")
        return None

    def validate_device(self, device_id: int) -> bool:
        """Validate if a device ID is valid and available"""
        try:
            device_info = self._audio.get_device_info_by_index(device_id)
            return device_info['maxInputChannels'] > 0
        except Exception:
            return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._audio.terminate()
