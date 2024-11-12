import pytest
import numpy as np
from src.audio.mixer import AudioMixer


@pytest.fixture
def mixer():
    return AudioMixer(sample_rate=44100, chunk_size=1024)


@pytest.fixture
def sample_audio():
    # Create more realistic test data with actual audio-like characteristics
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(44100 * duration))

    # Create a 440Hz sine wave for mic (A4 note)
    mic_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.float32)
    # Create a 880Hz sine wave for desktop (A5 note)
    desktop_data = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.float32)

    return mic_data, desktop_data


def test_initialization():
    """Test mixer initialization with different parameters"""
    # Test default initialization
    default_mixer = AudioMixer()
    assert default_mixer.sample_rate == 44100
    assert default_mixer.chunk_size == 1024
    assert default_mixer.target_sample_rate == 16000

    # Test custom initialization
    custom_mixer = AudioMixer(sample_rate=48000, chunk_size=2048)
    assert custom_mixer.sample_rate == 48000
    assert custom_mixer.chunk_size == 2048


def test_resample(mixer, sample_audio):
    mic_data, _ = sample_audio

    # Test downsampling
    resampled_data = mixer._resample(mic_data, 44100)
    expected_length = int(len(mic_data) * (16000 / 44100))
    assert len(resampled_data) == expected_length

    # Test that resampled data is within int16 range
    assert np.max(np.abs(resampled_data)) <= 32767

    # Test basic signal properties
    assert not np.all(resampled_data == 0), "Resampled data is all zeros"
    assert np.mean(np.abs(resampled_data)) > 0, "No signal in resampled data"

    # Test upsampling
    resampled_up = mixer._resample(mic_data[:1000], 8000)
    expected_up_length = int(1000 * (16000 / 8000))
    assert len(resampled_up) == expected_up_length

    # Test same rate (should return same data)
    same_rate_data = np.array([1, 2, 3], dtype=np.float32)
    same_rate = mixer._resample(same_rate_data, 16000)
    assert np.array_equal(same_rate, same_rate_data)


def test_prepare_for_transcription(mixer, sample_audio):
    mic_data, desktop_data = sample_audio

    result = mixer.prepare_for_transcription(
        mic_data, desktop_data, 44100, 44100
    )

    # Check keys exist
    assert all(key in result for key in ['combined', 'ch_0', 'ch_1'])

    # Check data types
    assert all(result[key].dtype == np.int16 for key in result)

    # Check valid ranges
    for key in result:
        assert np.max(np.abs(result[key])) <= 32767, f"Clipping in {key}"
        assert not np.all(result[key] == 0), f"No signal in {key}"

    # Check lengths match
    assert len(result['ch_0']) == len(result['ch_1'])
    assert len(result['combined']) == len(result['ch_0'])


def test_prepare_for_transcription_stereo_input(mixer):
    # Create stereo test data
    stereo_data = np.random.randint(-32768, 32767, (1000, 2)).astype(np.int16)

    result = mixer.prepare_for_transcription(
        stereo_data, None, 44100, 44100
    )

    # Check mono conversion
    assert len(result['ch_0'].shape) == 1
    assert result['ch_0'].dtype == np.int16
    assert len(result['ch_0']) > 0


def test_edge_case(mixer):
    """
    Test handling of different length channels in prepare_for_transcription
    """
    # Create test data with different lengths
    # 1 second at 44100 Hz
    mic_data = np.random.rand(44100).astype(np.float32) * 32767
    # 0.5 second at 44100 Hz
    desktop_data = np.random.rand(22050).astype(np.float32) * 32767

    result = mixer.prepare_for_transcription(
        mic_data, desktop_data, 44100, 44100
    )

    # Calculate the expected length after resampling to 16000 Hz
    expected_length = int(max(len(mic_data),
                              len(desktop_data)) * (16000 / 44100)
                          )

    # Check that the output arrays are the same
    # length and match the expected resampled length
    assert len(result['ch_0']) == expected_length
    assert len(result['ch_1']) == expected_length

    # Verify that combined audio matches the expected resampled length
    assert len(result['combined']) == expected_length

    # Ensure no clipping in combined output
    assert np.max(np.abs(result['combined'])) <= 32767


def test_create_transcription_chunk(mixer, sample_audio):
    mic_data, desktop_data = sample_audio
    channels = mixer.prepare_for_transcription(
        mic_data, desktop_data, 44100, 44100
    )
    chunk = mixer.create_transcription_chunk(channels)

    # Basic chunk validation
    assert isinstance(chunk, bytes)
    assert len(chunk) == len(channels['ch_0']) * 4  # 2 channels * 2 bytes
    assert len(chunk) % 4 == 0, "Chunk not aligned to 4 bytes"

    # Verify data can be read back
    interleaved = np.frombuffer(chunk, dtype=np.int16)
    assert len(interleaved) == len(channels['ch_0']) * 2

    # Check channel separation
    ch0_recovered = interleaved[::2]
    ch1_recovered = interleaved[1::2]
    assert len(ch0_recovered) == len(channels['ch_0'])
    assert len(ch1_recovered) == len(channels['ch_1'])


def test_create_transcription_chunk_validation(mixer):
    """Test chunk creation with invalid inputs"""
    # Test with empty channels
    empty_chunk = mixer.create_transcription_chunk({})
    assert len(empty_chunk) == 0

    # Test with single channel
    single_channel = {'ch_0': np.array([1, 2, 3], dtype=np.int16)}
    chunk = mixer.create_transcription_chunk(single_channel)
    assert len(chunk) == 12  # 3 samples * 2 channels * 2 bytes


def test_audio_levels(mixer, sample_audio):
    """Test audio level preservation and clipping prevention"""
    mic_data, desktop_data = sample_audio

    # Test with high amplitude signals
    loud_mic = mic_data * 2
    loud_desktop = desktop_data * 2

    result = mixer.prepare_for_transcription(
        loud_mic, loud_desktop, 44100, 44100
    )

    # Check for clipping prevention
    assert np.max(np.abs(result['combined'])) <= 32767
    assert np.max(np.abs(result['ch_0'])) <= 32767
    assert np.max(np.abs(result['ch_1'])) <= 32767


def test_prepare_for_transcription_with_silence(mixer):
    """Test handling of silent audio"""
    silence = np.zeros(44100, dtype=np.float32)
    result = mixer.prepare_for_transcription(
        silence, silence, 44100, 44100
    )

    assert np.all(result['combined'] == 0)
    assert np.all(result['ch_0'] == 0)
    assert np.all(result['ch_1'] == 0)


def test_chunk_byte_alignment(mixer, sample_audio):
    """Test proper byte alignment of output chunks"""
    mic_data, desktop_data = sample_audio
    channels = mixer.prepare_for_transcription(
        mic_data, desktop_data, 44100, 44100
    )
    chunk = mixer.create_transcription_chunk(channels)

    # Test chunk size is multiple of 4 (2 bytes/sample * 2 channels)
    assert len(chunk) % 4 == 0

    # Test sample alignment
    samples = np.frombuffer(chunk, dtype=np.int16)
    assert len(samples) % 2 == 0  # Even number of samples for stereo


def test_get_chunk_duration(mixer):
    # Create a chunk of known duration (100ms)
    sample_rate = 16000
    duration_ms = 100
    num_samples = int(sample_rate * duration_ms / 1000)

    # Create dummy audio data
    audio_data = np.zeros(num_samples, dtype=np.int16)
    chunk = mixer.create_transcription_chunk({
        'ch_0': audio_data,
        'ch_1': audio_data
    })

    duration = mixer.get_chunk_duration(chunk)
    assert abs(duration - duration_ms) < 1  # Within 1ms


@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100, 48000])
def test_different_sample_rates(mixer, sample_rate):
    """Test handling of different input sample rates"""
    duration = 1.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)
    test_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.float32)

    result = mixer.prepare_for_transcription(
        test_data, None, sample_rate, sample_rate
    )

    # Check if output is at target sample rate (16kHz)
    expected_length = int(len(test_data) * (16000 / sample_rate))
    assert len(result['ch_0']) == expected_length


def test_prepare_for_transcription_none_data(mixer):
    """Test handling of None inputs"""
    result = mixer.prepare_for_transcription(None, None, 44100, 44100)

    assert result['combined'].size == 0
    assert result['ch_0'].size == 0
    assert result['ch_1'].size == 0
    assert all(result[key].dtype == np.int16 for key in result)


def test_prepare_for_transcription_performance(mixer, sample_audio, benchmark):
    """Test performance of audio processing"""
    mic_data, desktop_data = sample_audio

    def process():
        mixer.prepare_for_transcription(mic_data, desktop_data, 44100, 44100)

    # Run the benchmark
    benchmark(process)
