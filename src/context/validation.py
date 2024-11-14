"""Context validation functionality."""
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from src.conversation.manager import ConversationManager
from .types import (
    ContextLevel,
    ContextSource,
    ContextState,
    ContextEntry,
    ContextMetadata
)
from .exceptions import ContextValidationError


class ValidationRule:
    """Rule for context validation."""

    def __init__(
            self,
            condition: Callable[[Any], bool],
            error_message: str,
            level: ContextLevel = ContextLevel.IMPORTANT
    ):
        """Initialize validation rule.

        Args:
            condition: Validation function
            error_message: Error message if validation fails
            level: Rule importance level
        """
        self.condition = condition
        self.error_message = error_message
        self.level = level


class ContextValidator:
    """Handles context validation."""

    def __init__(
            self,
            conversation_manager: ConversationManager
    ):
        """Initialize validator.

        Args:
            conversation_manager: For AI validation
        """
        self.conversation = conversation_manager
        self.rules: Dict[ContextSource, List[ValidationRule]] = {}
        self._setup_default_rules()

    def add_rule(
            self,
            source: ContextSource,
            rule: ValidationRule
    ) -> None:
        """Add validation rule.

        Args:
            source: Context source
            rule: Validation rule
        """
        if source not in self.rules:
            self.rules[source] = []
        self.rules[source].append(rule)

    async def validate(
            self,
            entry: ContextEntry
    ) -> Dict[str, Any]:
        """Validate context entry.

        Args:
            entry: Entry to validate

        Returns:
            Validation results

        Raises:
            ContextValidationError: If validation fails
        """
        errors = []
        warnings = []

        # Apply source-specific rules
        source_rules = self.rules.get(entry.metadata.source, [])
        for rule in source_rules:
            try:
                if not rule.condition(entry.content):
                    if rule.level in (
                            ContextLevel.CRITICAL,
                            ContextLevel.IMPORTANT
                    ):
                        errors.append(rule.error_message)
                    else:
                        warnings.append(rule.error_message)
            except Exception as e:
                errors.append(f"Rule evaluation failed: {str(e)}")

        # Get AI validation
        ai_results = await self._get_ai_validation(entry)
        errors.extend(ai_results.get("errors", []))
        warnings.extend(ai_results.get("warnings", []))

        # Create validation result
        validation = {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "timestamp": datetime.now().isoformat()
        }

        if not validation["is_valid"]:
            raise ContextValidationError(
                "Context validation failed",
                errors
            )

        return validation

    async def validate_update(
            self,
            old_entry: ContextEntry,
            new_content: Any,
            metadata_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate context update.

        Args:
            old_entry: Existing entry
            new_content: Updated content
            metadata_updates: Metadata updates

        Returns:
            Validation results
        """
        # Create temporary entry with updates
        updated_metadata = old_entry.metadata
        for key, value in metadata_updates.items():
            if hasattr(updated_metadata, key):
                setattr(updated_metadata, key, value)

        temp_entry = ContextEntry(
            id=old_entry.id,
            content=new_content,
            metadata=updated_metadata
        )

        # Validate updated entry
        return await self.validate(temp_entry)

    def _setup_default_rules(self) -> None:
        """Set up default validation rules."""
        # Common rules for all sources
        common_rules = [
            ValidationRule(
                lambda x: bool(x),
                "Content cannot be empty",
                ContextLevel.CRITICAL
            ),
            ValidationRule(
                lambda x: len(str(x)) < 10000,
                "Content exceeds maximum length",
                ContextLevel.IMPORTANT
            )
        ]

        for source in ContextSource:
            self.rules[source] = common_rules.copy()

        # Source-specific rules
        self.rules[ContextSource.CONVERSATION].extend([
            ValidationRule(
                lambda x: len(str(x).split()) >= 2,
                "Conversation context too short",
                ContextLevel.RELEVANT
            )
        ])

        self.rules[ContextSource.DOCUMENT].extend([
            ValidationRule(
                lambda x: isinstance(x, (dict, str)),
                "Invalid document content format",
                ContextLevel.CRITICAL
            )
        ])

        self.rules[ContextSource.ANALYSIS].extend([
            ValidationRule(
                lambda x: isinstance(x, dict) and "confidence" in x,
                "Analysis must include confidence score",
                ContextLevel.IMPORTANT
            )
        ])

    async def _get_ai_validation(
            self,
            entry: ContextEntry
    ) -> Dict[str, Any]:
        """Get AI-based validation.

        Args:
            entry: Entry to validate

        Returns:
            Validation results
        """
        # Create validation prompt
        prompts = {
            ContextSource.CONVERSATION: (
                "Validate this conversation context. Consider:\n"
                "1. Relevance to conversation\n"
                "2. Information completeness\n"
                "3. Coherence and clarity"
            ),
            ContextSource.DOCUMENT: (
                "Validate this document context. Check:\n"
                "1. Document structure\n"
                "2. Content completeness\n"
                "3. Reference validity"
            ),
            ContextSource.ANALYSIS: (
                "Validate this analysis context. Verify:\n"
                "1. Analysis completeness\n"
                "2. Logical consistency\n"
                "3. Supporting evidence"
            ),
            ContextSource.USER_INPUT: (
                "Validate this user input context. Check:\n"
                "1. Input validity\n"
                "2. Required information\n"
                "3. Format consistency"
            )
        }

        prompt = (
            f"{prompts.get(entry.metadata.source, 'Validate this context')}.\n\n"
            "Provide validation results in the following JSON format:\n"
            "{\n"
            '  "is_valid": boolean,\n'
            '  "errors": ["list of critical issues"],\n'
            '  "warnings": ["list of potential issues"],\n'
            '  "suggestions": ["list of improvements"]\n'
            "}\n\n"
            f"Content to validate:\n{entry.content}\n\n"
            f"Metadata:\n{entry.metadata.__dict__}"
        )

        # Get AI validation
        responses = []
        async for response in self.conversation.send_message(prompt):
            if response.text:
                responses.append(response.text)

        # Parse results
        try:
            import json
            results = json.loads(''.join(responses))
            return {
                "errors": results.get("errors", []),
                "warnings": results.get("warnings", []),
                "suggestions": results.get("suggestions", [])
            }
        except json.JSONDecodeError:
            return {
                "errors": [],
                "warnings": ["Failed to parse AI validation results"],
                "suggestions": []
            }

    async def check_relevance(
            self,
            entry: ContextEntry,
            query_context: str
    ) -> float:
        """Check context relevance to query.

        Args:
            entry: Context entry
            query_context: Query context

        Returns:
            Relevance score (0-1)
        """
        # Create relevance check prompt
        prompt = (
            "Determine the relevance of this context to the query.\n\n"
            f"Context:\n{entry.content}\n\n"
            f"Query Context:\n{query_context}\n\n"
            "Provide relevance score (0-1) in JSON format:\n"
            "{\n"
            '  "score": float,\n'
            '  "reasoning": "string"\n'
            "}"
        )

        # Get AI assessment
        responses = []
        async for response in self.conversation.send_message(prompt):
            if response.text:
                responses.append(response.text)

        try:
            import json
            results = json.loads(''.join(responses))
            return float(results.get("score", 0.5))
        except (json.JSONDecodeError, ValueError):
            return 0.5

    async def check_consistency(
            self,
            entries: List[ContextEntry]
    ) -> List[Dict[str, Any]]:
        """Check consistency between context entries.

        Args:
            entries: Entries to check

        Returns:
            Consistency issues found
        """
        if len(entries) < 2:
            return []

        # Create consistency check prompt
        prompt = (
                "Check consistency between these context entries:\n\n"
                + "\n---\n".join(
            f"Entry {i + 1}:\n{entry.content}"
            for i, entry in enumerate(entries)
        )
                + "\n\nProvide consistency check results in JSON format:\n"
                  "{\n"
                  '  "issues": [\n'
                  '    {\n'
                  '      "type": "string",\n'
                  '      "description": "string",\n'
                  '      "severity": "high/medium/low",\n'
                  '      "entries_involved": [int]\n'
                  '    }\n'
                  "  ]\n"
                  "}"
        )

        # Get AI assessment
        responses = []
        async for response in self.conversation.send_message(prompt):
            if response.text:
                responses.append(response.text)

        try:
            import json
            results = json.loads(''.join(responses))
            return results.get("issues", [])
        except json.JSONDecodeError:
            return []