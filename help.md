# Step 1: Initial Project Setup

## 1. Create Basic Directory Structure
```bash
mkdir speech-analytics
cd speech-analytics

# Create basic directory structure
mkdir -p src/audio src/transcription src/summarization tests docs
```

## 2. Create Initial Files

### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "speech-analytics"
version = "0.1.0"
description = "Real-time speech analytics and summarization tool"
authors = [{name = "Your Name", email = "your.email@example.com"}]
```

### requirements.txt
```
pyaudiowpatch
numpy
sounddevice
boto3
```

### src/audio/capture.py
```python
import pyaudiowpatch as pyaudio
import numpy as np
import sounddevice as sd

class AudioCapture:
    def __init__(self, mic_device_id=None, desktop_device_id=None):
        self.mic_device_id = mic_device_id
        self.desktop_device_id = desktop_device_id
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        
    @staticmethod
    def list_devices():
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        devices = []
        for i in range(num_devices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            devices.append({
                'id': i,
                'name': device_info['name'],
                'input_channels': device_info['maxInputChannels'],
                'output_channels': device_info['maxOutputChannels'],
                'sample_rate': device_info['defaultSampleRate']
            })
        
        p.terminate()
        return devices
    
    def start_capture(self):
        """Start capturing audio from selected devices"""
        raise NotImplementedError("This is your first task to implement!")
```

### src/transcription/aws_transcribe.py
```python
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

class TranscriptionHandler(TranscriptResultStreamHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transcript = []

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # Implement basic transcript handling
        results = transcript_event.transcript.results
        for result in results:
            if not result.is_partial:
                for alt in result.alternatives:
                    self.transcript.append(alt.transcript)
```

### src/summarization/aws_bedrock.py
```python
import boto3
from botocore.exceptions import ClientError

class Summarizer:
    def __init__(self, region="us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = "amazon.titan-text-premier-v1:0"
    
    def summarize(self, text):
        # Implement basic summarization
        raise NotImplementedError("This will be implemented later!")
```

### tests/test_audio.py
```python
import pytest
from src.audio.capture import AudioCapture

def test_list_devices():
    devices = AudioCapture.list_devices()
    assert isinstance(devices, list)
    for device in devices:
        assert 'id' in device
        assert 'name' in device
```

### README.md
```markdown
# Speech Analytics

Real-time speech analytics and summarization tool.

## Features (Planned)
- Real-time audio capture from multiple sources
- Live transcription using AWS Transcribe
- Automated summarization using AWS Bedrock
- Support for multiple languages

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```python
from src.audio.capture import AudioCapture

# List available audio devices
devices = AudioCapture.list_devices()
for device in devices:
    print(f"Device {device['id']}: {device['name']}")
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the AGPL-3.0 License.
```