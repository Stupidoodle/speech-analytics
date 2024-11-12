import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.transcription.aws_transcribe import TranscribeManager, \
    TranscriptionConfig
from src.transcription.exceptions import (
    TranscriptionError,
    ServiceUnavailableException,
    RateLimitError,
    ConnectionError
)


@pytest.fixture
def config():
    """Create a test configuration."""
    return TranscriptionConfig(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm",
        enable_partial_results_stabilization=True
    )


@pytest.fixture
def mock_client():
    """Create a mock AWS Transcribe client."""
    with patch('src.transcription.aws_transcribe.TranscribeStreamingClient'
               ) as m:
        client = Mock()
        m.return_value = client
        client.start_stream_transcription = AsyncMock()
        yield client


@pytest.fixture
def mock_stream():
    """Create a mock stream."""
    stream = Mock()
    stream.input_stream = AsyncMock()
    stream.output_stream = AsyncMock()
    return stream


@pytest.fixture
async def transcribe_manager(config, mock_client):
    """Create a TranscribeManager instance with mocked components."""
    manager = TranscribeManager(
        region="us-west-2",
        config=config,
        max_retries=2,
        retry_delay=0.1
    )
    manager.client = mock_client
    yield manager


@pytest.mark.asyncio
async def test_create_client_success(transcribe_manager):
    """Test successful client creation."""
    manager = await anext(transcribe_manager)
    manager.client = None
    await manager._create_client()
    assert manager.client is not None


@pytest.mark.asyncio
async def test_create_client_retry_failure(transcribe_manager):
    """Test client creation with retry failure."""
    with patch(
        'src.transcription.aws_transcribe.TranscribeStreamingClient',
        side_effect=Exception("Connection error")
    ):
        manager = await anext(transcribe_manager)
        manager.client = None
        with pytest.raises(ConnectionError):
            await manager._create_client()


@pytest.mark.asyncio
async def test_start_stream_success(transcribe_manager, mock_stream):
    """Test successful stream start."""
    manager = await anext(transcribe_manager)
    manager.client.start_stream_transcription.return_value = (
        mock_stream
    )
    await manager.start_stream()
    assert manager.stream == mock_stream


@pytest.mark.asyncio
async def test_start_stream_failure(transcribe_manager):
    """Test stream start failure."""
    manager = await anext(transcribe_manager)
    manager.client.start_stream_transcription.side_effect = (
        Exception("Stream error")
    )
    with pytest.raises(TranscriptionError):
        await manager.start_stream()


@pytest.mark.asyncio
async def test_process_audio_success(transcribe_manager, mock_stream):
    """Test successful audio processing."""
    manager = await anext(transcribe_manager)
    manager.stream = mock_stream
    await manager.process_audio(b"test audio")
    mock_stream.input_stream.send_audio_event.assert_called_once_with(
        audio_chunk=b"test audio"
    )


@pytest.mark.asyncio
async def test_process_audio_rate_limit(transcribe_manager, mock_stream):
    """Test rate limit handling."""
    manager = await anext(transcribe_manager)
    manager.stream = mock_stream
    mock_stream.input_stream.send_audio_event.side_effect = Exception(
        "ThrottlingException"
    )
    with pytest.raises(RateLimitError):
        await manager.process_audio(b"test audio")


@pytest.mark.asyncio
async def test_process_audio_service_unavailable(
    transcribe_manager,
    mock_stream
):
    """Test service unavailable handling."""
    manager = await anext(transcribe_manager)
    manager.stream = mock_stream
    mock_stream.input_stream.send_audio_event.side_effect = Exception(
        "ServiceUnavailable"
    )
    with pytest.raises(ServiceUnavailableException):
        await manager.process_audio(b"test audio")


@pytest.mark.asyncio
async def test_stop_stream_success(transcribe_manager, mock_stream):
    """Test successful stream stop."""
    manager = await anext(transcribe_manager)
    manager.stream = mock_stream
    result = await manager.stop_stream()
    mock_stream.input_stream.end_stream.assert_called_once()
    assert isinstance(result, dict)
    assert 'combined' in result
    assert 'channels' in result


@pytest.mark.asyncio
async def test_stop_stream_failure(transcribe_manager, mock_stream):
    """Test stream stop failure."""
    manager = await anext(transcribe_manager)
    manager.stream = mock_stream
    mock_stream.input_stream.end_stream.side_effect = Exception("Stop error")
    with pytest.raises(TranscriptionError):
        await manager.stop_stream()


@pytest.mark.asyncio
async def test_context_manager(transcribe_manager, mock_stream):
    """Test async context manager."""
    manager = await anext(transcribe_manager)
    manager.client.start_stream_transcription.return_value = (
        mock_stream
    )
    async with manager as async_manager:
        assert async_manager.stream == mock_stream
        await async_manager.process_audio(b"test audio")
    mock_stream.input_stream.end_stream.assert_called_once()


@pytest.mark.asyncio
async def test_transcription_config():
    """Test transcription configuration."""
    config = TranscriptionConfig(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm",
        vocabulary_name="custom-vocab",
        show_speaker_label=True
    )
    config_dict = config.to_dict()

    assert config_dict["language_code"] == "en-US"
    assert config_dict["media_sample_rate_hz"] == 16000
    assert config_dict["media_encoding"] == "pcm"
    assert config_dict["vocabulary_name"] == "custom-vocab"
    assert config_dict["show_speaker_label"] is True

    # Ensure None values are excluded
    assert "session_id" not in config_dict
    assert "vocab_filter_name" not in config_dict


@pytest.mark.asyncio
async def test_retry_logic(transcribe_manager, mock_stream):
    """Test retry logic for audio processing."""
    manager = await anext(transcribe_manager)
    manager.stream = mock_stream
    mock_stream.input_stream.send_audio_event.side_effect = [
        Exception("Temporary error"),
        None  # Success on second attempt
    ]

    await manager.process_audio(b"test audio")
    assert mock_stream.input_stream.send_audio_event.call_count == 2
