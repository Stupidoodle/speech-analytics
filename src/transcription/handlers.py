from typing import Dict, Any, List, Callable
from datetime import datetime


class TranscriptHandler:
    """Handles and processes transcription results"""

    def __init__(self):
        self.transcripts = {
            'mic': [],      # Microphone transcripts
            'desktop': [],  # Desktop audio transcripts
            'combined': []  # Combined transcripts
        }
        self._callbacks = []

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback to be called when new transcripts arrive"""
        self._callbacks.append(callback)

    def process_transcript(self,
                           transcript_data: Dict[str, Any]
                           ) -> Dict[str, List[Dict[str, Any]]]:
        """Process and store new transcript data"""
        timestamp = datetime.now().isoformat()

        # Process channel-specific transcripts
        if 'channels' in transcript_data:
            for channel_id, transcripts in transcript_data['channels'].items():
                target = 'mic' if channel_id == 'ch_0' else 'desktop'
                for transcript in transcripts:
                    entry = {
                        'text': transcript['transcript'],
                        'confidence': transcript.get('confidence', 0.0),
                        'timestamp': timestamp,
                        'start_time': transcript.get('start_time'),
                        'end_time': transcript.get('end_time')
                    }
                    self.transcripts[target].append(entry)

        # Process combined transcripts
        if 'combined' in transcript_data:
            for transcript in transcript_data['combined']:
                entry = {
                    'text': transcript['transcript'],
                    'confidence': transcript.get('confidence', 0.0),
                    'timestamp': timestamp,
                    'start_time': transcript.get('start_time'),
                    'end_time': transcript.get('end_time')
                }
                self.transcripts['combined'].append(entry)

        # Notify callbacks
        for callback in self._callbacks:
            callback(self.transcripts)

        return self.transcripts

    def clear(self) -> None:
        """Clear stored transcripts"""
        self.transcripts = {
            'mic': [],
            'desktop': [],
            'combined': []
        }
