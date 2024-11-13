# Detailed Module Hierarchy

## 1. Base Layer - Types and Exceptions
```
src/conversation/types.py
└── Core Data Types
    ├── Role (Enum)
    │   ├── Defines all possible roles (INTERVIEWER, INTERVIEWEE, etc.)
    │   └── Used throughout system for role-based behavior
    ├── ContentType (Enum)
    │   ├── TEXT, AUDIO, DOCUMENT, TRANSCRIPT
    │   └── Used for content classification
    ├── DocumentFormat (Enum)
    │   ├── PDF, CSV, DOC, DOCX, etc.
    │   └── Used for document handling
    ├── MessageRole (Enum)
    │   ├── USER, ASSISTANT
    │   └── Used for Bedrock conversation
    ├── BedrockConfig (Pydantic Model)
    │   ├── model_id, inference_config, tool_config
    │   └── Used for AWS Bedrock configuration
    ├── Message (Pydantic Model)
    │   ├── role, content, timestamp
    │   └── Base message structure
    └── StreamResponse (Pydantic Model)
        ├── content, metadata, stop_reason
        └── Used for streaming responses

src/conversation/exceptions.py
└── Base Exception Classes
    ├── SpeechAnalyticsError
    │   └── Base exception class
    ├── ConfigurationError
    │   └── Configuration issues
    ├── ResourceError
    │   ├── ResourceNotFoundError
    │   └── ResourceBusyError
    ├── ServiceError
    │   ├── ServiceConnectionError
    │   ├── ServiceTimeoutError
    │   └── ServiceQuotaError
    └── ProcessingError
        ├── AudioError
        ├── TranscriptionError
        └── DocumentError
```

## 2. Audio Processing Stack
```
src/audio/devices.py
└── Device Management
    ├── Class: DeviceManager
    │   ├── __init__(self)
    │   ├── list_devices(self) -> List[Dict]
    │   ├── get_default_input_device(self) -> Optional[Dict]
    │   ├── get_default_output_device(self) -> Optional[Dict]
    │   └── validate_device(self, device_id: int) -> bool
    └── Dependencies
        ├── pyaudiowpatch
        └── audio.exceptions

src/audio/mixer.py
└── Audio Mixing
    ├── Class: AudioMixer
    │   ├── __init__(self, sample_rate: int, chunk_size: int)
    │   ├── _resample(self, audio_data: np.ndarray, original_rate: int) -> np.ndarray
    │   ├── prepare_for_transcription(self, mic_data, desktop_data, rates) -> Dict
    │   └── create_transcription_chunk(self, channels: Dict) -> bytes
    └── Dependencies
        ├── numpy
        ├── librosa
        └── audio.exceptions

src/audio/processor.py
└── Audio Processing
    ├── Class: AudioProcessor
    │   ├── __init__(self, noise_threshold: float, gain: float)
    │   ├── calibrate_noise(self, audio: np.ndarray) -> None
    │   ├── reduce_noise(self, audio: np.ndarray) -> np.ndarray
    │   ├── normalize(self, audio: np.ndarray) -> Tuple[np.ndarray, float]
    │   └── process_chunk(self, audio: np.ndarray) -> Tuple[np.ndarray, dict]
    └── Dependencies
        ├── numpy
        └── audio.exceptions

src/audio/capture.py
└── Audio Capture
    ├── Class: AudioCapture
    │   ├── __init__(self, mic_device_id: Optional[int], desktop_device_id: Optional[int])
    │   ├── start_capture(self, callback: Optional[Callable]) -> None
    │   ├── stop_capture(self) -> None
    │   ├── _init_mic_stream(self) -> Any
    │   └── _process_audio(self) -> None
    └── Dependencies
        ├── audio.devices
        ├── audio.mixer
        ├── audio.processor
        └── audio.exceptions
```

## 3. Transcription Stack
```
src/transcription/aws_transcribe.py
└── AWS Transcribe Integration
    ├── Class: TranscriptionConfig
    │   ├── language_code: str
    │   ├── media_sample_rate_hz: int
    │   └── media_encoding: str
    ├── Class: TranscribeManager
    │   ├── __init__(self, region: str, config: Optional[TranscriptionConfig])
    │   ├── _create_client(self) -> None
    │   ├── start_stream(self) -> None
    │   ├── process_audio(self, audio_chunk: bytes) -> Optional[str]
    │   └── stop_stream(self) -> Dict[str, List[Dict[str, Any]]]
    └── Dependencies
        ├── aioboto3
        └── transcription.exceptions

src/transcription/buffer.py
└── Audio Buffer Management
    ├── Class: AudioBuffer
    │   ├── __init__(self, max_size: int, chunk_size: int)
    │   ├── write(self, data: bytes) -> None
    │   ├── read(self, size: Optional[int]) -> bytes
    │   └── get_latency(self) -> float
    ├── Class: StreamBuffer
    │   ├── __init__(self, target_latency: float, max_latency: float)
    │   ├── write(self, data: bytes) -> None
    │   └── read(self) -> Optional[bytes]
    └── Dependencies
        └── transcription.exceptions

src/transcription/handlers.py
└── Transcription Result Handling
    ├── Class: TranscriptHandler
    │   ├── __init__(self)
    │   ├── add_callback(self, callback: Callable) -> None
    │   ├── process_transcript(self, data: Dict) -> Dict
    │   └── clear(self) -> None
    └── Dependencies
        └── None
```

## 4. Document Processing Stack
```
src/document/preprocessor.py
└── Document Preprocessing
    ├── Class: PreprocessedDocument
    │   ├── content: Dict[str, Any]
    │   ├── metadata: Dict[str, Any]
    │   └── keywords: List[str]
    ├── Class: DocumentPreprocessor
    │   ├── __init__(self)
    │   ├── preprocess(self, content: str, doc_type: str) -> PreprocessedDocument
    │   └── _extract_sections(self, content: str) -> Dict[str, str]
    └── Dependencies
        └── document.exceptions

src/document/processor.py
└── Document Processing
    ├── Class: DocumentProcessor
    │   ├── __init__(self, conversation_manager: ConversationManager)
    │   ├── process_document(self, content: str, doc_type: str) -> Dict
    │   └── update_context(self, doc_type: str, new_content: str) -> Dict
    └── Dependencies
        ├── conversation.manager
        └── document.exceptions
```

## 5. Event System
```
src/events/bus.py
└── Event Management
    ├── Class: Event
    │   ├── type: EventType
    │   ├── data: Any
    │   └── timestamp: datetime
    ├── Class: EventBus
    │   ├── __init__(self)
    │   ├── subscribe(self, event_type: EventType, callback: Callable) -> None
    │   └── publish(self, event: Event) -> None
    └── Dependencies
        └── None
```

## 6. Core Conversation Stack
```
src/conversation/manager.py
└── Conversation Management
    ├── Class: ConversationManager
    │   ├── __init__(self, region: str, model_id: str)
    │   ├── send_message(self, text: str, files: Optional[List]) -> AsyncIterator
    │   ├── add_document(self, content: bytes, name: str, format: str) -> None
    │   └── export_conversation(self) -> Dict[str, Any]
    └── Dependencies
        ├── conversation.types
        ├── conversation.exceptions
        └── conversation.context

src/conversation/context.py
└── Context Management
    ├── Class: ConversationContext
    │   ├── __init__(self, role: Role)
    │   ├── add_document(self, path: str, doc_type: str) -> Document
    │   ├── _add_role_prompt(self) -> None
    │   └── get_system_messages(self) -> List[Dict]
    └── Dependencies
        ├── conversation.types
        └── conversation.exceptions
```

## 7. Assistance Layer
```
src/assistance/enhanced_assistant.py
└── Enhanced Assistance
    ├── Class: EnhancedAssistant
    │   ├── __init__(self, context: ConversationContext, manager: ConversationManager)
    │   ├── process_turn(self, turn_content: str, speaker: str) -> AsyncIterator
    │   └── suggest_questions(self, context: Dict) -> AsyncIterator
    └── Dependencies
        ├── conversation.context
        ├── conversation.manager
        └── assistance.types

src/assistance/analyzers/
├── base.py
│   └── Class: BaseAnalyzer
├── interview.py
│   └── Class: InterviewAnalyzer
└── support.py
    └── Class: SupportAnalyzer
```

## 8. Real-time Processing
```
src/realtime/processor.py
└── Real-time Coordination
    ├── Class: RealtimeProcessor
    │   ├── __init__(self, event_bus: EventBus, managers...)
    │   ├── start(self, region: str, audio_config: Optional[Dict]) -> None
    │   ├── _handle_audio_chunk(self, event: Event) -> None
    │   ├── _handle_transcript(self, event: Event) -> None
    │   └── _handle_context_update(self, event: Event) -> None
    └── Dependencies
        ├── conversation.manager
        ├── conversation.context
        ├── transcription.aws_transcribe
        └── events.bus

src/realtime/events.py
└── Real-time Events
    ├── Class: EventType (Enum)
    ├── Class: Event
    └── Class: EventEmitter
```

Each module's dependencies are strictly defined, and the flow of data follows these relationships. Would you like me to:

1. Detail any specific module's internal functions?
2. Show the exact call flow for specific operations?
3. Detail specific component interactions?
4. Explain any particular dependency chain?