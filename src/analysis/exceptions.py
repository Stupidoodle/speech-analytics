"""Exceptions for the analysis layer."""
from typing import Dict, Any, List, Optional
from datetime import datetime


class AnalysisError(Exception):
    """Base exception for analysis errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()


class AnalysisFormatError(AnalysisError):
    """Error when AI response format is invalid."""
    def __init__(
            self,
            message: str,
            response: Dict[str, Any],
            expected_format: Dict[str, Any],
    ):
        super().__init__(message)
        self.response = response
        self.expected_format = expected_format

class AnalysisTaskError(AnalysisError):
    """Error in analysis task execution."""

    def __init__(
        self,
        message: str,
        task_id: str,
        task_type: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.task_id = task_id
        self.task_type = task_type


class AnalysisPipelineError(AnalysisError):
    """Error in analysis pipeline execution."""

    def __init__(
        self,
        message: str,
        stage: str,
        failed_tasks: List[str],
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.stage = stage
        self.failed_tasks = failed_tasks


class AnalysisConfigError(AnalysisError):
    """Error in analysis configuration."""

    def __init__(
        self,
        message: str,
        config_errors: Dict[str, str],
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.config_errors = config_errors


class AnalysisTimeoutError(AnalysisError):
    """Error when analysis times out."""

    def __init__(
        self,
        message: str,
        timeout: float,
        task_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.timeout = timeout
        self.task_id = task_id


class AnalysisDependencyError(AnalysisError):
    """Error in analysis task dependencies."""

    def __init__(
        self,
        message: str,
        task_id: str,
        missing_deps: List[str],
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.task_id = task_id
        self.missing_deps = missing_deps


class AnalysisValidationError(AnalysisError):
    """Error in analysis input validation."""

    def __init__(
        self,
        message: str,
        validation_errors: List[Dict[str, Any]],
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.validation_errors = validation_errors


class AnalysisResourceError(AnalysisError):
    """Error due to resource constraints."""

    def __init__(
        self,
        message: str,
        resource_type: str,
        current_usage: Dict[str, float],
        limits: Dict[str, float],
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limits = limits


class AnalysisStateError(AnalysisError):
    """Error in analysis state transition."""

    def __init__(
        self,
        message: str,
        current_state: str,
        target_state: str,
        task_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.current_state = current_state
        self.target_state = target_state
        self.task_id = task_id


class AnalyzerNotFoundError(AnalysisError):
    """Error when analyzer is not found."""

    def __init__(
        self,
        message: str,
        analyzer_type: str,
        available_analyzers: List[str],
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.analyzer_type = analyzer_type
        self.available_analyzers = available_analyzers