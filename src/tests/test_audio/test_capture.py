import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from src.audio.capture import AudioCapture
from src.audio.exceptions import CaptureError, DeviceError


@pytest.fixture
def mock_device_manager():
    """Create a mock device manager."""
    with patch('src.audio.capture.DeviceManager') as mock:
        manager = mock.return_value
        # Setup device validation
        manager.validate_device.return_value = True
        # Setup device info
        manager.get_device_info.return_value = {
            'name': 'Test Device',
            'max_input_channels': 2,
            'default_sample_rate': 44100
        }
        # Setup stream opening
        manager.open_input_stream = AsyncMock()
        yield manager


@pytest.fixture
def mock_mixer():
    """Create a mock audio mixer."""
    with patch('src.audio.capture.AudioMixer') as mock:
        mixer = mock.return_value
        mixer.prepare_for_transcription.return_value = {
            'combined': np.zeros(1024, dtype=np.int16),
            'ch_0': np.zeros(1024, dtype=np.int16),
            'ch_1': np.zeros(1024, dtype=np.int16)
        }
        yield mixer


@pytest.fixture
def mock_processor():
    """Create a mock audio processor."""
    with patch('src.audio.capture.AudioProcessor') as mock:
        processor = mock.return_value
        processor.process_chunk.return_value = (
            np.zeros(1024, dtype=np.int16),
            {'peak_amplitude': 0.0, 'is_silence': True}
        )
        yield processor


@pytest.fixture
def mock_stream():
    """Create a mock audio stream."""
    stream = AsyncMock()
    stream.read = AsyncMock(
        return_value=np.zeros(1024, dtype=np.int16).tobytes()
    )
    stream.stop_stream = Mock()
    stream.close = Mock()
    return stream


@pytest.fixture
async def audio_capture(mock_device_manager, mock_mixer, mock_processor):
    """Create an AudioCapture instance with mocked components."""
    async def cleanup(capture):
        await capture.stop_capture()

    with patch.multiple(
        'src.audio.capture',
        DeviceManager=Mock(return_value=mock_device_manager),
        AudioMixer=Mock(return_value=mock_mixer),
        AudioProcessor=Mock(return_value=mock_processor)
    ):
        capture = AudioCapture(
            mic_device_id=1,
            desktop_device_id=2,
            sample_rate=44100,
            chunk_size=1024
        )
        try:
            yield capture
        finally:
            await cleanup(capture)


@pytest.mark.asyncio
async def test_initialization(audio_capture):
    """Test AudioCapture initialization."""
    capture = await anext(audio_capture)
    assert capture.sample_rate == 44100
    assert capture.chunk_size == 1024
    assert capture.mic_device_id == 1
    assert capture.desktop_device_id == 2
    assert not capture.is_running
    assert not capture.is_paused


@pytest.mark.asyncio
async def test_invalid_device_ids():
    """Test initialization with invalid device IDs."""
    with patch('src.audio.capture.DeviceManager') as mock:
        manager = mock.return_value
        manager.validate_device.return_value = False

        with pytest.raises(DeviceError):
            AudioCapture(mic_device_id=999)

        with pytest.raises(DeviceError):
            AudioCapture(desktop_device_id=999)


@pytest.mark.asyncio
async def test_start_capture(audio_capture, mock_stream):
    """Test starting audio capture."""
    capture = await anext(audio_capture)
    capture.device_manager.open_input_stream.return_value = mock_stream
    callback_data = []

    async def test_callback(data):
        callback_data.append(data)

    await capture.start_capture(callback=test_callback)
    assert capture.is_running

    await asyncio.sleep(0.1)
    await capture.stop_capture()

    assert len(callback_data) > 0
    assert 'combined' in callback_data[0]
    assert 'ch_0' in callback_data[0]
    assert 'ch_1' in callback_data[0]


@pytest.mark.asyncio
async def test_pause_resume(audio_capture, mock_stream):
    """Test pausing and resuming capture."""
    capture = await anext(audio_capture)
    capture.device_manager.open_input_stream.return_value = mock_stream

    await capture.start_capture()
    assert capture.is_running
    assert not capture.is_paused

    await capture.pause_capture()
    assert capture.is_paused

    await capture.resume_capture()
    assert not capture.is_paused

    await capture.stop_capture()
    assert not capture.is_running


@pytest.mark.asyncio
async def test_audio_levels(audio_capture, mock_stream):
    """Test audio level monitoring."""
    capture = await anext(audio_capture)
    test_audio = (np.sin(np.linspace(0, 2*np.pi, 1024)) * 32767).astype(
        np.int16
    )
    mock_stream.read.return_value = test_audio.tobytes()
    capture.device_manager.open_input_stream.return_value = mock_stream

    await capture.start_capture()
    levels = await capture.get_audio_levels()

    assert 'mic' in levels
    assert 'desktop' in levels
    assert isinstance(levels['mic'], (float, np.float32))
    assert isinstance(levels['desktop'], (float, np.float32))

    await capture.stop_capture()


@pytest.mark.asyncio
async def test_error_handling(audio_capture, mock_stream):
    """Test error handling during capture."""
    capture = await anext(audio_capture)
    mock_stream.read = AsyncMock(side_effect=Exception("Test error"))
    capture.device_manager.open_input_stream.return_value = mock_stream

    errors = []

    async def error_callback(data):
        errors.append(data)

    await capture.start_capture(callback=error_callback)
    await asyncio.sleep(0.1)

    assert capture.is_running
    await capture.stop_capture()


@pytest.mark.asyncio
async def test_cleanup(audio_capture, mock_stream):
    """Test proper resource cleanup."""
    capture = await anext(audio_capture)
    capture.device_manager.open_input_stream.return_value = mock_stream

    await capture.start_capture()
    await capture.stop_capture()

    mock_stream.stop_stream.assert_called()
    mock_stream.close.assert_called()
    assert not capture._streams


@pytest.mark.asyncio
async def test_device_error_handling(audio_capture):
    """Test handling of device initialization errors."""
    capture = await anext(audio_capture)
    capture.device_manager.open_input_stream = AsyncMock(
        side_effect=Exception("Device error")
    )

    with pytest.raises(CaptureError):
        await capture.start_capture()

    assert not capture.is_running


@pytest.mark.asyncio
async def test_multiple_start_attempts(audio_capture, mock_stream):
    """Test attempting to start capture multiple times."""
    capture = await anext(audio_capture)
    capture.device_manager.open_input_stream.return_value = mock_stream

    await capture.start_capture()
    with pytest.raises(CaptureError):
        await capture.start_capture()

    await capture.stop_capture()
