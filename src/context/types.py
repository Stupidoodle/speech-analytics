from enum import Enum
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel

class ContextLevel(str, Enum):
    """Context importance levels."""
    CRITICAL = "critical"     # Must be considered
    IMPORTANT = "important"   # Should be considered
    RELEVANT = "relevant"     # Good to consider
    BACKGROUND = "background" # Optional context


class ContextSource(str, Enum):
    """Sources of context information."""
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    ANALYSIS = "analysis"
    USER_INPUT = "user_input"
    SYSTEM = "system"
    EXTERNAL = "external"


class ContextState(str, Enum):
    """States of context entries."""
    ACTIVE = "active"       # Currently relevant
    ARCHIVED = "archived"   # Stored but not active
    PENDING = "pending"     # Awaiting validation
    INVALID = "invalid"     # Not valid/usable


@dataclass
class ContextMetadata:
    """Metadata for context entries."""
    source: ContextSource
    level: ContextLevel
    state: ContextState = ContextState.ACTIVE
    timestamp: datetime = field(default_factory=datetime.now)
    expiry: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    references: Set[str] = field(default_factory=set)
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextEntry:
    """Individual context entry."""
    id: str
    content: Any
    metadata: ContextMetadata
    validation_info: Optional[Dict[str, Any]] = None


class ContextQuery(BaseModel):
    """Query for retrieving context."""
    sources: Optional[Set[ContextSource]] = None
    levels: Optional[Set[ContextLevel]] = None
    states: Optional[Set[ContextState]] = None
    tags: Optional[Set[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: Optional[int] = None


class ContextUpdate(BaseModel):
    """Update to context entry."""
    entry_id: str
    content: Optional[Any] = None
    metadata_updates: Optional[Dict[str, Any]] = None
    validation_info: Optional[Dict[str, Any]] = None


class ContextConfig(BaseModel):
    """Configuration for context management."""
    enabled_sources: Set[ContextSource]
    max_entries: int = 1000
    retention_period: Optional[int] = None  # days
    auto_archive: bool = True
    validation_required: bool = False
    merge_strategy: str = "latest_wins"
    level_thresholds: Dict[ContextLevel, float] = {}
