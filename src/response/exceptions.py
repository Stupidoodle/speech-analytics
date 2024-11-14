"""Response layer exceptions."""
from typing import Dict, Any, List, Optional

class ResponseError(Exception):
    """Base exception for response errors."""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class GenerationError(ResponseError):
    """Error in response generation."""
    def __init__(
        self,
        message: str,
        response_type: str,
        query: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.response_type = response_type
        self.query = query

class TemplateError(ResponseError):
    """Error in template processing."""
    def __init__(
        self,
        message: str,
        template_name: str,
        variables: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.template_name = template_name
        self.variables = variables

class ValidationError(ResponseError):
    """Error in response validation."""
    def __init__(
        self,
        message: str,
        validation_errors: List[str],
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.validation_errors = validation_errors

class ConfigError(ResponseError):
    """Error in configuration."""
    def __init__(
        self,
        message: str,
        config: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.config = config

class ContextError(ResponseError):
    """Error in context handling."""
    def __init__(
        self,
        message: str,
        context_refs: List[str],
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.context_refs = context_refs

class PriorityError(ResponseError):
    """Error in priority handling."""
    def __init__(
        self,
        message: str,
        priority: float,
        threshold: float,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.priority = priority
        self.threshold = threshold