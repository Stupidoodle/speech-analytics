"""Exceptions for context management."""
from typing import Dict, Any, Optional, List


class ContextError(Exception):
    """Base exception for context management."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ContextNotFoundError(ContextError):
    """Error when context entry not found."""

    def __init__(
        self,
        entry_id: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message or f"Context entry not found: {entry_id}",
            details
        )
        self.entry_id = entry_id


class ContextValidationError(ContextError):
    """Error in context validation."""

    def __init__(
        self,
        message: str,
        validation_errors: List[Dict[str, Any]],
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.validation_errors = validation_errors


class ContextStorageError(ContextError):
    """Error in context storage operations."""
    pass


class ContextUpdateError(ContextError):
    """Error during context update."""

    def __init__(
        self,
        message: str,
        entry_id: str,
        update_type: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.entry_id = entry_id
        self.update_type = update_type


class ContextMergeError(ContextError):
    """Error during context merging."""

    def __init__(
        self,
        message: str,
        entries: List[str],
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.entries = entries


class ContextQueryError(ContextError):
    """Error in context querying."""

    def __init__(
        self,
        message: str,
        query: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.query = query
