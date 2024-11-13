# src/types.py
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class EventType(Enum):
    AUDIO_CHUNK = "audio_chunk"
    TRANSCRIPT = "transcript"
    DOCUMENT_PROCESSED = "document_processed"
    ASSISTANCE = "assistance"
    TOOL_USE = "tool_use"
    CONTEXT_UPDATE = "context_update"
    MESSAGE_SENT = "message_sent"
    RESPONSE_RECEIVED = "response_received"
    DOCUMENT_ADDED = "document_added"
    ERROR = "error"
    METRICS = "metrics"

class Event(BaseModel):
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict[str, Any]] = None
