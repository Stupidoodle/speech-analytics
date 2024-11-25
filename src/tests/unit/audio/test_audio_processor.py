import numpy as np
import pytest

from src.audio.processor import AudioProcessor
from src.audio.types import AudioConfig
from src.events.bus import EventBus


@pytest.mark.asyncio
async def test_process_chunk_resampling():
    """Test resampling functionality in AudioProcessor."""
    event_bus = EventBus()
    config = AudioConfig(sample_rate=16000, channels=1)
    processor = AudioProcessor(event_bus, config)

    # Generate mock audio data (48 kHz, mono)
    audio_data = (np.random.rand(48000) * 32767).astype(np.int16).tobytes()

    # Process the chunk
    result = await processor.process_chunk(audio_data)

    assert result.sample_rate == 16000
    assert len(result.processed_data) > 0
    assert result.channels == 1


@pytest.mark.asyncio
async def test_process_chunk_noise_reduction():
    """Test noise reduction in AudioProcessor."""
    event_bus = EventBus()
    config = AudioConfig(sample_rate=16000, channels=1)
    processor = AudioProcessor(event_bus, config)

    # Generate mock audio data with random noise
    audio_data = (np.random.rand(16000) * 32767).astype(np.int16).tobytes()

    # Process the chunk
    result = await processor.process_chunk(audio_data, apply_noise_reduction=True)

    assert len(result.processed_data) > 0
    assert result.channels == 1


@pytest.mark.asyncio
async def test_process_chunk_mono_conversion():
    """Test mono conversion for stereo input in AudioProcessor."""
    event_bus = EventBus()
    config = AudioConfig(sample_rate=16000, channels=2)
    processor = AudioProcessor(event_bus, config)

    # Generate stereo audio data
    audio_data = (np.random.rand(16000 * 2) * 32767).astype(np.int16).tobytes()

    # Process the chunk
    result = await processor.process_chunk(audio_data)

    assert result.channels == 1
    assert len(result.processed_data) > 0
