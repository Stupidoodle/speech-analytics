from unittest.mock import MagicMock, patch

import pytest

from src.audio.devices import DeviceManager
from src.audio.exceptions import DeviceError
from src.audio.types import AudioConfig, DeviceInfo, DeviceType


@pytest.fixture
def mock_pyaudio():
    """Mock pyaudiowpatch module."""
    with patch("src.audio.devices.pyaudio") as mock_pyaudio:
        yield mock_pyaudio


@pytest.fixture
def device_manager(mock_pyaudio):
    """Initialize DeviceManager with mocked PyAudio."""
    mock_pyaudio.PyAudio.return_value.get_host_api_info_by_type.return_value = {
        "defaultOutputDevice": 0
    }
    return DeviceManager()


@pytest.fixture
def audio_config():
    """Provide a sample AudioConfig."""
    return AudioConfig(
        sample_rate=16000,
        channels=2,
        chunk_duration_ms=100,
    )


def test_initialization_success(mock_pyaudio):
    """Test successful initialization of the DeviceManager."""
    mock_pyaudio.PyAudio.return_value.get_host_api_info_by_type.return_value = {
        "defaultOutputDevice": 0
    }
    device_manager = DeviceManager()
    assert device_manager._wasapi_info is not None


def test_initialization_failure(mock_pyaudio):
    """Test initialization failure when WASAPI is unavailable."""
    mock_pyaudio.PyAudio.return_value.__enter__.return_value.get_host_api_info_by_type.side_effect = (
        DeviceError
    )
    with pytest.raises(
        DeviceError, match="Failed to initialize WASAPI. Ensure it is supported."
    ):
        DeviceManager()


@pytest.mark.asyncio
async def test_list_devices_success(mock_pyaudio, device_manager):
    """Test successful listing of devices."""
    mock_device_info = {
        "index": 0,
        "name": "Mock Device",
        "maxInputChannels": 2,
        "defaultSampleRate": 16000,
    }
    mock_pyaudio.PyAudio.return_value.__enter__.return_value.get_device_info_generator.return_value = [
        mock_device_info
    ]

    devices = await device_manager.list_devices()
    assert len(devices) == 1
    assert devices[0].name == "Mock Device"
    assert devices[0].channels == 2


@pytest.mark.asyncio
async def test_get_default_stereo_devices_success(mock_pyaudio, device_manager):
    """Test retrieving default stereo devices."""
    mock_input_device = {
        "index": 1,
        "name": "Default Mic",
        "maxInputChannels": 2,
        "defaultSampleRate": 16000,
    }
    mock_output_device = {
        "index": 2,
        "name": "Default Speakers",
        "maxInputChannels": 2,
        "defaultSampleRate": 16000,
        "isLoopbackDevice": True,
    }

    mock_pyaudio.PyAudio.return_value.__enter__.return_value.get_default_wasapi_device.side_effect = [
        mock_input_device,
        mock_output_device,
    ]

    devices = await device_manager.get_default_stereo_devices()
    assert devices["mic"].name == "Default Mic"
    assert devices["desktop"].name == "Default Speakers"


@pytest.mark.asyncio
async def test_open_stream_success(mock_pyaudio, device_manager, audio_config):
    """Test opening a stream successfully."""
    mock_device_info = DeviceInfo(
        id=1,
        name="Test Device",
        type=DeviceType.INPUT,
        channels=2,
        sample_rate=16000,
        is_default=True,
    )

    mock_pyaudio.PyAudio.return_value.__enter__.return_value.open.return_value = (
        MagicMock()
    )
    stream = device_manager.open_stream(mock_device_info, audio_config)
    assert stream is not None


@pytest.mark.asyncio
async def test_open_stream_failure(mock_pyaudio, device_manager, audio_config):
    """Test failure to open a stream."""
    device_info = DeviceInfo(
        id=1,
        name="Test Device",
        type=DeviceType.INPUT,
        channels=2,
        sample_rate=16000,
        is_default=True,
        is_loopback=False,  # Adding missing field
        supports_input=True,  # Adding missing field
        supports_output=False,  # Adding missing field
    )

    mock_pyaudio.PyAudio.return_value.__enter__.return_value.open.side_effect = OSError

    with pytest.raises(
        DeviceError, match=f"Failed to open stream for device {device_info.name}."
    ):
        async for _ in device_manager.open_stream(device_info, audio_config):
            pass
