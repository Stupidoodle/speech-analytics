from typing import Dict, Any, List, Optional
import asyncio
from amazon_transcribe.client import TranscribeStreamingClient

from .exceptions import (
    TranscriptionError,
    ServiceUnavailableException,
    RateLimitError,
    ConnectionError
)


class TranscriptionConfig:
    """Configuration for AWS Transcribe streaming."""

    def __init__(
        self,
        language_code: str = "en-US",
        media_sample_rate_hz: int = 16000,
        media_encoding: str = "pcm",
        vocabulary_name: Optional[str] = None,
        session_id: Optional[str] = None,
        vocab_filter_name: Optional[str] = None,
        vocab_filter_method: Optional[str] = None,
        show_speaker_label: bool = False,
        enable_channel_identification: bool = False,
        number_of_channels: int = 1,
        enable_partial_results_stabilization: bool = True,
        partial_results_stability: str = "high",
        content_identification_type: Optional[str] = None,
        content_redaction_type: Optional[str] = None,
        pii_entity_types: Optional[str] = None,
        language_model_name: Optional[str] = None,
        identify_language: bool = False,
        language_options: Optional[List[str]] = None,
        preferred_language: Optional[str] = None,
        vocabulary_names: Optional[List[str]] = None,
        vocabulary_filter_names: Optional[List[str]] = None
    ):
        """Initialize transcription configuration.

        Args:
            language_code: Language code for transcription
            media_sample_rate_hz: Audio sample rate
            media_encoding: Audio encoding format
            vocabulary_name: Custom vocabulary name
            session_id: Session identifier
            vocab_filter_name: Vocabulary filter name
            vocab_filter_method: Vocabulary filter method
            show_speaker_label: Enable speaker labels
            enable_channel_identification: Enable channel identification
            number_of_channels: Number of audio channels
            enable_partial_results_stabilization: Enable result stabilization
            partial_results_stability: Stability level for partial results
            content_identification_type: Content identification type
            content_redaction_type: Content redaction type
            pii_entity_types: PII entity types to redact
            language_model_name: Custom language model name
            identify_language: Enable language identification
            language_options: List of possible languages
            preferred_language: Preferred language for identification
            vocabulary_names: List of custom vocabularies
            vocabulary_filter_names: List of vocabulary filters
        """
        self.language_code = language_code
        self.media_sample_rate_hz = media_sample_rate_hz
        self.media_encoding = media_encoding
        self.vocabulary_name = vocabulary_name
        self.session_id = session_id
        self.vocab_filter_name = vocab_filter_name
        self.vocab_filter_method = vocab_filter_method
        self.show_speaker_label = show_speaker_label
        self.enable_channel_identification = enable_channel_identification
        self.number_of_channels = number_of_channels
        self.enable_partial_results_stabilization = (
            enable_partial_results_stabilization
        )
        self.partial_results_stability = partial_results_stability
        self.content_identification_type = content_identification_type
        self.content_redaction_type = content_redaction_type
        self.pii_entity_types = pii_entity_types
        self.language_model_name = language_model_name
        self.identify_language = identify_language
        self.language_options = language_options
        self.preferred_language = preferred_language
        self.vocabulary_names = vocabulary_names
        self.vocabulary_filter_names = vocabulary_filter_names

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for AWS API."""
        config = {
            "language_code": self.language_code,
            "media_sample_rate_hz": self.media_sample_rate_hz,
            "media_encoding": self.media_encoding,
            "vocabulary_name": self.vocabulary_name,
            "session_id": self.session_id,
            "vocab_filter_name": self.vocab_filter_name,
            "vocab_filter_method": self.vocab_filter_method,
            "show_speaker_label": self.show_speaker_label,
            "enable_channel_identification":
                self.enable_channel_identification,
            "number_of_channels": self.number_of_channels,
            "enable_partial_results_stabilization": (
                self.enable_partial_results_stabilization
            ),
            "partial_results_stability": self.partial_results_stability,
            "content_identification_type": self.content_identification_type,
            "content_redaction_type": self.content_redaction_type,
            "pii_entity_types": self.pii_entity_types,
            "language_model_name": self.language_model_name,
            "identify_language": self.identify_language,
            "language_options": self.language_options,
            "preferred_language": self.preferred_language,
            "vocabulary_names": self.vocabulary_names,
            "vocabulary_filter_names": self.vocabulary_filter_names
        }
        return {k: v for k, v in config.items() if v is not None}


class TranscribeManager:
    """Manages AWS Transcribe streaming sessions with retry logic."""

    def __init__(
        self,
        region: str,
        config: Optional[TranscriptionConfig] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize the transcription manager.

        Args:
            region: AWS region for transcription
            config: Transcription configuration
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.region = region
        self.config = config or TranscriptionConfig()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        self.stream = None
        self.handler = None

    async def _create_client(self) -> None:
        """Create AWS Transcribe client with retry logic."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                self.client = TranscribeStreamingClient(region=self.region)
                return
            except Exception as e:
                attempt += 1
                if attempt == self.max_retries:
                    raise ConnectionError(
                        f"Failed to create client after {self.max_retries} "
                        f"attempts: {e}"
                    )
                await asyncio.sleep(self.retry_delay * attempt)

    async def start_stream(self) -> None:
        """Start a new transcription stream."""
        if not self.client:
            await self._create_client()

        try:
            self.stream = await self.client.start_stream_transcription(
                **self.config.to_dict()
            )
        except Exception as e:
            raise TranscriptionError(f"Failed to start stream: {e}")

    async def process_audio(
        self,
        audio_chunk: bytes,
        retry_on_failure: bool = True
    ) -> None:
        """Process audio chunk with retry logic.

        Args:
            audio_chunk: Raw audio data to process
            retry_on_failure: Whether to retry on failure
        """
        if not self.stream:
            raise TranscriptionError("Stream not started")

        attempt = 0
        while attempt < (self.max_retries if retry_on_failure else 1):
            try:
                await self.stream.input_stream.send_audio_event(
                    audio_chunk=audio_chunk
                )
                return
            except Exception as e:
                attempt += 1
                if attempt == self.max_retries or not retry_on_failure:
                    if "ThrottlingException" in str(e):
                        raise RateLimitError(
                            "Transcription rate limit exceeded"
                        )
                    elif "ServiceUnavailable" in str(e):
                        raise ServiceUnavailableException(
                            "AWS Transcribe service unavailable"
                        )
                    else:
                        raise TranscriptionError(
                            f"Failed to process audio: {e}"
                            )
                await asyncio.sleep(self.retry_delay * attempt)

    async def stop_stream(self) -> Dict[str, List[Dict[str, Any]]]:
        """Stop transcription and return results."""
        if self.stream:
            try:
                await self.stream.input_stream.end_stream()
                results = await self._get_final_results()
                self.stream = None
                return results
            except Exception as e:
                raise TranscriptionError(f"Failed to stop stream: {e}")
        return {'combined': [], 'channels': {'ch_0': [], 'ch_1': []}}

    async def _get_final_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get final transcription results."""
        # Implementation depends on how you want to handle final results
        # This is a placeholder that should be customized
        return {'combined': [], 'channels': {'ch_0': [], 'ch_1': []}}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_stream()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.stream:
            await self.stop_stream()
