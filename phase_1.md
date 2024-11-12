# Phase 1: Core Audio Implementation

## 1. src/audio/exceptions.py
```python
class AudioError(Exception):
    """Base exception for audio-related errors"""
    pass

class DeviceError(AudioError):
    """Exception for device-related errors"""
    pass

class CaptureError(AudioError):
    """Exception for capture-related errors"""
    pass

class ProcessingError(AudioError):
    """Exception for audio processing errors"""
    pass
```

## 2. src/audio/devices.py
```python
from typing import Dict, List, Optional
import pyaudiowpatch as pyaudio
from .exceptions import DeviceError

class DeviceManager:
    """Manages audio devices and their properties"""
    
    def __init__(self):
        self._audio = pyaudio.PyAudio()
        self._wasapi_info = self._audio.get_host_api_info_by_type(pyaudio.paWASAPI)
    
    def list_devices(self) -> List[Dict]:
        """List all available audio devices"""
        devices = []
        
        try:
            for i in range(self._audio.get_device_count()):
                try:
                    device_info = self._audio.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:
                        devices.append({
                            'id': i,
                            'name': device_info['name'],
                            'input_channels': device_info['maxInputChannels'],
                            'sample_rate': int(device_info['defaultSampleRate']),
                            'is_wasapi': device_info['hostApi'] == self._wasapi_info['index'],
                            'is_loopback': 'Loopback' in device_info['name']
                        })
                except Exception as e:
                    print(f"Warning: Could not get info for device {i}: {e}")
            
            return devices
        except Exception as e:
            raise DeviceError(f"Failed to list audio devices: {e}")
    
    def get_default_input_device(self) -> Optional[Dict]:
        """Get the default input device"""
        try:
            device_id = self._wasapi_info.get("defaultInputDevice")
            if device_id is not None:
                device_info = self._audio.get_device_info_by_index(device_id)
                return {
                    'id': device_id,
                    'name': device_info['name'],
                    'input_channels': device_info['maxInputChannels'],
                    'sample_rate': int(device_info['defaultSampleRate'])
                }
        except Exception as e:
            print(f"Warning: Could not get default input device: {e}")
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
```

## 3. src/audio/mixer.py
```python
import numpy as np
from typing import Optional, Tuple
from .exceptions import ProcessingError

class AudioMixer:
    """Handles mixing of multiple audio streams"""
    
    def __init__(self, target_sample_rate: int = 44100):
        self.target_sample_rate = target_sample_rate
    
    def mix_streams(self, 
                   mic_data: Optional[np.ndarray], 
                   desktop_data: Optional[np.ndarray],
                   mic_weight: float = 1.0,
                   desktop_weight: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Mix microphone and desktop audio streams with weights
        Returns: (mixed_audio, peak_amplitude)
        """
        try:
            if mic_data is None and desktop_data is None:
                raise ProcessingError("No audio data provided for mixing")
            
            # Convert to float32 for mixing
            if mic_data is not None:
                mic_float = mic_data.astype(np.float32) / 32768.0 * mic_weight
            else:
                mic_float = np.zeros(0, dtype=np.float32)
            
            if desktop_data is not None:
                desktop_float = desktop_data.astype(np.float32) / 32768.0 * desktop_weight
            else:
                desktop_float = np.zeros(0, dtype=np.float32)
            
            # Ensure same length
            length = max(len(mic_float), len(desktop_float))
            if len(mic_float) < length:
                mic_float = np.pad(mic_float, (0, length - len(mic_float)))
            if len(desktop_float) < length:
                desktop_float = np.pad(desktop_float, (0, length - len(desktop_float)))
            
            # Mix streams
            mixed = (mic_float + desktop_float) / (mic_weight + desktop_weight)
            
            # Calculate peak amplitude
            peak_amplitude = np.max(np.abs(mixed))
            
            # Prevent clipping
            if peak_amplitude > 1.0:
                mixed = mixed / peak_amplitude
                peak_amplitude = 1.0
            
            # Convert back to int16
            return (mixed * 32767).astype(np.int16), peak_amplitude
            
        except Exception as e:
            raise ProcessingError(f"Error mixing audio streams: {e}")
```

## 4. src/audio/processor.py
```python
import numpy as np
from typing import Optional, Tuple
from .exceptions import ProcessingError

class AudioProcessor:
    """Handles audio processing tasks like noise reduction and normalization"""
    
    def __init__(self):
        self.noise_floor = None
        self.silence_threshold = 0.01  # -40 dB
        
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to use full dynamic range"""
        try:
            if len(audio) == 0:
                return audio
                
            peak = np.max(np.abs(audio))
            if peak > 0:
                return (audio * (32767 / peak)).astype(np.int16)
            return audio
        except Exception as e:
            raise ProcessingError(f"Error normalizing audio: {e}")
    
    def reduce_noise(self, 
                    audio: np.ndarray, 
                    calibrate: bool = False) -> np.ndarray:
        """Simple noise reduction using spectral gating"""
        try:
            if len(audio) == 0:
                return audio
                
            # Convert to float
            audio_float = audio.astype(np.float32) / 32768.0
            
            # Calibrate noise floor if requested
            if calibrate:
                self.noise_floor = np.mean(np.abs(audio_float))
                return audio
                
            # Apply noise gate
            if self.noise_floor is not None:
                mask = np.abs(audio_float) > (self.noise_floor * 2)
                audio_float *= mask
                
            return (audio_float * 32767).astype(np.int16)
            
        except Exception as e:
            raise ProcessingError(f"Error reducing noise: {e}")
```

## 5. src/audio/capture.py
```python
from typing import Optional, Dict, Tuple
import numpy as np
import wave
import time
from .devices import DeviceManager
from .mixer import AudioMixer
from .processor import AudioProcessor
from .exceptions import CaptureError

class AudioCapture:
    """Main audio capture class that coordinates devices, mixing, and processing"""
    
    def __init__(self,
                 mic_device_id: Optional[int] = None,
                 desktop_device_id: Optional[int] = None,
                 chunk_size: int = 1024,
                 sample_rate: int = 44100):
        
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.running = False
        
        # Initialize components
        self.device_manager = DeviceManager()
        self.mixer = AudioMixer(target_sample_rate=sample_rate)
        self.processor = AudioProcessor()
        
        # Validate devices
        if mic_device_id and not self.device_manager.validate_device(mic_device_id):
            raise CaptureError(f"Invalid microphone device ID: {mic_device_id}")
        if desktop_device_id and not self.device_manager.validate_device(desktop_device_id):
            raise CaptureError(f"Invalid desktop device ID: {desktop_device_id}")
        
        self.mic_device_id = mic_device_id
        self.desktop_device_id = desktop_device_id
        
        # Will be initialized in start_capture()
        self._mic_stream = None
        self._desktop_stream = None
        
    def start_capture(self) -> Dict:
        """Start capturing audio from configured devices"""
        if self.running:
            raise CaptureError("Audio capture already running")
            
        try:
            # Initialize streams using device manager
            # ... (similar to previous implementation but using DeviceManager)
            pass
            
    def read_processed_chunk(self) -> Tuple[np.ndarray, float]:
        """Read and process a chunk of audio from all active streams"""
        try:
            # Read raw audio
            mic_data, desktop_data = self.read_chunk()
            
            # Mix streams
            mixed_audio, peak = self.mixer.mix_streams(mic_data, desktop_data)
            
            # Process audio
            processed_audio = self.processor.reduce_noise(mixed_audio)
            processed_audio = self.processor.normalize_audio(processed_audio)
            
            return processed_audio, peak
            
        except Exception as e:
            raise CaptureError(f"Error reading and processing audio chunk: {e}")
```