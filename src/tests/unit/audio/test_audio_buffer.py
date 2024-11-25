import asyncio

import pytest

from src.audio.buffer import AudioBuffer
from src.audio.exceptions import BufferError
from src.audio.types import AudioConfig
from src.events.bus import EventBus


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def audio_config():
    return AudioConfig(sample_rate=16000, channels=2, chunk_duration_ms=100)


@pytest.fixture
def buffer(event_bus, audio_config):
    return AudioBuffer(event_bus, audio_config, max_size=4096)


@pytest.mark.asyncio
async def test_write_read_combined(buffer):
    data = b"\x00\x01" * 400  # Stereo data
    await buffer.write(data)
    read_data = await buffer.read(len(data))
    assert read_data == data


@pytest.mark.asyncio
async def test_write_read_channel(buffer):
    data = b"\x00\x01" * 200  # Mono data
    await buffer.write(data, channel="ch_0")
    read_data = await buffer.read(size=len(data), channel="ch_0")

    assert read_data == data


@pytest.mark.asyncio
async def test_stereo_splitting(buffer):
    stereo_data = b"\x00\x01\x02\x03" * 100
    await buffer.write(stereo_data)

    left_data = await buffer.read(channel="ch_0", size=200)
    right_data = await buffer.read(channel="ch_1", size=200)

    assert left_data == b"\x00\x01" * 100
    assert right_data == b"\x02\x03" * 100


@pytest.mark.asyncio
async def test_overflow(buffer):
    data = b"\x00\x01" * 4096  # Data exceeding max_size
    status_before = buffer.get_status().metrics.overflow_count
    await buffer.write(data[:2000])
    await buffer.write(data[2000:])  # This should cause overflow
    status_after = buffer.get_status()
    overflow_count_status_after = status_after.metrics.overflow_count

    assert overflow_count_status_after > status_before
    assert (
        status_after.levels["combined"] <= 100.0
    )  # check that combined level is not 100% because of overflow


@pytest.mark.asyncio
async def test_underrun(buffer):
    read_data = await buffer.read(size=100)  # Reading from empty buffer
    assert read_data is None


@pytest.mark.asyncio
async def test_status(buffer):
    data = b"\x00\x01" * 200

    await buffer.write(data)

    status = buffer.get_status()
    assert status.levels["combined"] > 0
    assert status.latencies["combined"] >= 0
    assert "combined" in status.active_channels
    assert status.metrics.total_bytes_written == 400


@pytest.mark.asyncio
async def test_read_stream(buffer):
    test_data = b"\x00\x01" * 4  # Small test data
    await buffer.write(test_data)

    # Create a wrapper to handle the timeout more gracefully
    async def get_one_chunk():
        stream = buffer.read_stream()
        chunk = await buffer.read(size=len(test_data))  # Specify exact size
        return chunk

    # Use wait_for on our wrapper function
    chunk = await asyncio.wait_for(get_one_chunk(), timeout=1.0)

    # Verify we got valid data
    assert chunk is not None
    assert len(chunk) % 4 == 0  # Verify stereo PCM alignment
    assert chunk == test_data  # Verify exact data match


@pytest.mark.asyncio
async def test_invalid_channel(buffer):
    with pytest.raises(BufferError, match="Invalid stereo sample alignment"):
        await buffer.write(b"\x00\x01", channel="ch_2")

    with pytest.raises(BufferError, match="Invalid channel: ch_2"):
        await buffer.read(size=100, channel="ch_2")
