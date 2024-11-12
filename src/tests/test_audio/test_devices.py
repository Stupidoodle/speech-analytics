import pytest
from unittest.mock import patch
from src.audio.devices import DeviceManager
from src.audio.exceptions import DeviceError


# TODO: Rewrite missing tests to test error handling
# NOTE: Also test for isLoopbackDevice = True and missing loopback device etc.


@pytest.fixture
def mock_pyaudio():
    with patch('src.audio.devices.pyaudio.PyAudio') as mock_pyaudio:
        yield mock_pyaudio


def test_init(mock_pyaudio):
    mock_pyaudio_instance = mock_pyaudio.return_value
    mock_pyaudio_instance.get_host_api_info_by_type.return_value = {'index': 0}

    manager = DeviceManager()

    assert manager._wasapi_info == {'index': 0}


def test_init_failure(mock_pyaudio):
    mock_pyaudio_instance = mock_pyaudio.return_value
    mock_pyaudio_instance.get_host_api_info_by_type.side_effect = Exception(
        "WASAPI error"
        )

    with pytest.raises(DeviceError,
                       match="Failed to initialize WASAPI: WASAPI error"
                       ):
        DeviceManager()


def test_list_devices(mock_pyaudio):
    mock_pyaudio_instance = mock_pyaudio.return_value
    mock_pyaudio_instance.get_device_info_generator.return_value = [
        {
            'index': 0,
            'name': 'Device 1',
            'maxInputChannels': 2,
            'defaultSampleRate': 44100,
            'hostApi': 0
        },
        {
            'index': 1,
            'name': 'Device 2',
            'maxInputChannels': 0,
            'defaultSampleRate': 44100,
            'hostApi': 0
        },
    ]
    mock_pyaudio_instance.get_host_api_info_by_type.return_value = {'index': 0}

    manager = DeviceManager()
    devices = manager.list_devices()

    assert len(devices) == 1
    assert devices[0]['name'] == 'Device 1'


def test_get_default_input_device(mock_pyaudio):
    mock_pyaudio_instance = mock_pyaudio.return_value
    mock_pyaudio_instance.get_default_wasapi_device.return_value = {
        'index': 0,
        'name': 'Default Device',
        'maxInputChannels': 2,
        'defaultSampleRate': 44100
    }

    manager = DeviceManager()
    device = manager.get_default_input_device()

    assert device['name'] == 'Default Device'


def test_get_default_output_device(mock_pyaudio):
    mock_pyaudio_instance = mock_pyaudio.return_value
    mock_pyaudio_instance.get_default_wasapi_device.return_value = {
        'index': 0,
        'name': 'Default Output Device',
        'maxInputChannels': 2, 'defaultSampleRate': 44100,
        'isLoopbackDevice': False
    }
    mock_pyaudio_instance.get_loopback_device_info_generator.return_value = [
        {'index': 1,
         'name': 'Default Output Device [Loopback]',
         'maxInputChannels': 2,
         'defaultSampleRate': 44100}
    ]

    manager = DeviceManager()
    device = manager.get_default_output_device()

    assert device['name'] == 'Default Output Device [Loopback]'


def test_validate_device(mock_pyaudio):
    mock_pyaudio_instance = mock_pyaudio.return_value
    mock_pyaudio_instance.get_device_info_by_index.return_value = {
        'maxInputChannels': 2
        }

    manager = DeviceManager()
    is_valid = manager.validate_device(0)

    assert is_valid


def test_validate_device_invalid(mock_pyaudio):
    mock_pyaudio_instance = mock_pyaudio.return_value
    mock_pyaudio_instance.get_device_info_by_index.side_effect = Exception(
        "Invalid device"
        )

    manager = DeviceManager()
    is_valid = manager.validate_device(0)

    assert not is_valid
