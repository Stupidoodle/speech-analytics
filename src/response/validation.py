"""Response validation and filtering."""
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.conversation.manager import ConversationManager
from src.context.manager import ContextManager
from src.events.bus import EventBus
from src.events.types import Event, EventType

from .types import (
    ResponseType,
    GeneratedResponse,
    ResponseRequest,
    ResponseValidation,
    ResponsePriority
)
from .exceptions import ValidationError


class ResponseValidator:
    """Validates generated responses."""

    def __init__(
            self,
            event_bus: EventBus,
            conversation_manager: ConversationManager,
            context_manager: ContextManager
    ):
        self.event_bus = event_bus
        self.conversation = conversation_manager
        self.context = context_manager
        self._setup_validation_rules()

    async def validate_response(
            self,
            response: GeneratedResponse,
            request: ResponseRequest
    ) -> ResponseValidation:
        """Validate generated response."""
        errors = []
        warnings = []
        suggestions = []

        # Content validation
        content_validation = await self._validate_content(
            response.content,
            request
        )
        errors.extend(content_validation.get("errors", []))
        warnings.extend(content_validation.get("warnings", []))
        suggestions.extend(content_validation.get("suggestions", []))

        # Context validation
        if request.context:
            context_validation = await self._validate_context_usage(
                response,
                request.context
            )
            errors.extend(context_validation.get("errors", []))
            warnings.extend(context_validation.get("warnings", []))
            suggestions.extend(context_validation.get("suggestions", []))

        # Type-specific validation
        type_validation = await self._validate_response_type(
            response,
            request.response_type
        )
        errors.extend(type_validation.get("errors", []))
        warnings.extend(type_validation.get("warnings", []))
        suggestions.extend(type_validation.get("suggestions", []))

        # Priority validation
        priority_validation = await self._validate_priority(
            response.priority,
            request.priority
        )
        errors.extend(priority_validation.get("errors", []))
        warnings.extend(priority_validation.get("warnings", []))

        # Create validation result
        validation = ResponseValidation(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metadata={
                "validation_time": datetime.now().isoformat(),
                "context_used": response.context_used,
                "analysis_used": response.analysis_used
            }
        )

        # Emit validation event
        await self.event_bus.publish(Event(
            type=EventType.RESPONSE,
            data={
                "action": "response_validated",
                "is_valid": validation.is_valid,
                "error_count": len(errors)
            }
        ))

        return validation

    async def _validate_content(
            self,
            content: str,
            request: ResponseRequest
    ) -> Dict[str, List[str]]:
        """Validate response content."""
        prompt = (
            "Validate this response for quality and appropriateness. "
            "Consider:\n"
            "1. Relevance to query\n"
            "2. Clarity and coherence\n"
            "3. Professional tone\n"
            "4. Factual accuracy\n\n"
            f"Query: {request.query}\n"
            f"Response: {content}\n\n"
            "Provide validation results as JSON:\n"
            "{\n"
            '  "errors": ["list of critical issues"],\n'
            '  "warnings": ["list of potential issues"],\n'
            '  "suggestions": ["list of improvements"]\n'
            "}"
        )

        result = {"errors": [], "warnings": [], "suggestions": []}
        async for response in self.conversation.send_message(prompt):
            if response.text:
                try:
                    import json
                    result = json.loads(response.text)
                except json.JSONDecodeError:
                    continue

        return result

    async def _validate_context_usage(
            self,
            response: GeneratedResponse,
            context: ContextEntry
    ) -> Dict[str, List[str]]:
        """Validate context usage in response."""
        result = {
            "errors": [],
            "warnings": [],
            "suggestions": []
        }

        # Check context references
        if not response.context_used:
            result["warnings"].append(
                "Response doesn't reference provided context"
            )

        # Check context relevance
        if context.content:
            referenced_content = any(
                str(value) in response.content
                for value in context.content.values()
                if isinstance(value, (str, int, float))
            )
            if not referenced_content:
                result["warnings"].append(
                    "Response may not effectively use context"
                )

        return result

    async def _validate_response_type(
            self,
            response: GeneratedResponse,
            expected_type: Optional[ResponseType]
    ) -> Dict[str, List[str]]:
        """Validate response type consistency."""
        result = {
            "errors": [],
            "warnings": [],
            "suggestions": []
        }

        if not expected_type:
            return result

        if response.type != expected_type:
            result["errors"].append(
                f"Response type mismatch: expected {expected_type}, "
                f"got {response.type}"
            )

        # Type-specific checks
        if expected_type == ResponseType.CLARIFICATION:
            if "?" not in response.content:
                result["errors"].append(
                    "Clarification response should include a question"
                )

        elif expected_type == ResponseType.FOLLOW_UP:
            if not any(
                    word in response.content.lower()
                    for word in ["earlier", "previously", "mentioned"]
            ):
                result["warnings"].append(
                    "Follow-up might not reference previous context"
                )

        elif expected_type == ResponseType.SUGGESTION:
            if not any(
                    word in response.content.lower()
                    for word in ["suggest", "recommend", "try", "consider"]
            ):
                result["warnings"].append(
                    "Suggestion response should offer clear recommendations"
                )

        elif expected_type == ResponseType.SUMMARY:
            if len(response.content.split()) < 20:
                result["warnings"].append(
                    "Summary might be too brief"
                )

        return result

    async def _validate_priority(
            self,
            actual: ResponsePriority,
            expected: ResponsePriority
    ) -> Dict[str, List[str]]:
        """Validate response priority."""
        result = {
            "errors": [],
            "warnings": [],
            "suggestions": []
        }

        if actual < expected:
            result["errors"].append(
                f"Priority too low: expected {expected}, got {actual}"
            )
        elif actual > expected:
            result["warnings"].append(
                f"Priority higher than necessary: {actual} > {expected}"
            )

        return result

    def _setup_validation_rules(self) -> None:
        """Set up validation rules."""
        self.rules = {
            "content": {
                "min_length": 10,
                "max_length": 1000,
                "required_elements": {
                    ResponseType.CLARIFICATION: ["?"],
                    ResponseType.SUGGESTION: [
                        "suggest", "recommend", "try", "consider"
                    ]
                }
            },
            "context": {
                "min_references": 1,
                "max_references": 5
            },
            "priority": {
                "allowed_variance": 1.0
            }
        }


class ResponseFilter:
    """Filters and sanitizes responses."""

    def __init__(self):
        self._setup_filters()

    async def filter_response(
            self,
            response: GeneratedResponse
    ) -> GeneratedResponse:
        """Filter and clean response."""
        # Apply content filters
        content = response.content
        for filter_func in self.content_filters:
            content = filter_func(content)

        # Apply metadata filters
        metadata = response.metadata
        for filter_func in self.metadata_filters:
            metadata = filter_func(metadata)

        # Create filtered response
        return GeneratedResponse(
            content=content,
            type=response.type,
            priority=response.priority,
            context_used=response.context_used,
            analysis_used=response.analysis_used,
            template_used=response.template_used,
            duration=response.duration,
            confidence=response.confidence,
            metadata=metadata,
            timestamp=response.timestamp
        )

    def _setup_filters(self) -> None:
        """Set up response filters."""
        self.content_filters = [
            self._remove_sensitive_data,
            self._normalize_whitespace,
            self._fix_formatting
        ]

        self.metadata_filters = [
            self._clean_metadata,
            self._validate_references
        ]

    def _remove_sensitive_data(self, content: str) -> str:
        """Remove potentially sensitive information."""
        # Implement sensitive data removal
        return content

    def _normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace in content."""
        return " ".join(content.split())

    def _fix_formatting(self, content: str) -> str:
        """Fix common formatting issues."""
        # Implement formatting fixes
        return content

    def _clean_metadata(
            self,
            metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Clean metadata fields."""
        # Implement metadata cleaning
        return metadata

    def _validate_references(
            self,
            metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate metadata references."""
        # Implement reference validation
        return metadata