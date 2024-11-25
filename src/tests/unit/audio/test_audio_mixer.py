import pytest

from src.audio.mixer import AudioMixer
from src.audio.types import AudioConfig
from src.events.bus import EventBus


@pytest.mark.asyncio
async def test_mix_streams_basic():
    """Test basic mixing of two audio streams."""
    event_bus = EventBus()
    config = AudioConfig(sample_rate=16000, channels=2)
    mixer = AudioMixer(event_bus, config)

    primary = b"\x00\x01" * 100  # Fake PCM data
    secondary = b"\x02\x03" * 100  # Fake PCM data

    result = await mixer.mix_streams(primary, secondary)

    assert result.channels == 2
    assert len(result.processed_data) > 0


@pytest.mark.asyncio
async def test_mix_streams_with_clipping():
    """Test mixing with potential clipping."""
    event_bus = EventBus()
    config = AudioConfig(sample_rate=16000, channels=2)
    mixer = AudioMixer(event_bus, config)

    primary = b"\xFF\x7F" * 100  # Simulate high amplitude
    secondary = b"\xFF\x7F" * 100  # Simulate high amplitude

    result = await mixer.mix_streams(primary, secondary)

    assert result.channels == 2
    assert result.metrics.clipping_count > 0
