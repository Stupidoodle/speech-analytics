"""Integration helpers for conversation and document processing."""
from typing import Dict, Any, Optional

from src.common.types import Role, Document, BedrockConfig
from src.document.roles import DocumentRoles
from src.conversation.manager import ConversationManager


class ProcessingIntegration:
    """Integrates conversation and document processing."""

    def __init__(
            self,
            conversation_manager: ConversationManager,
            role: Role
    ):
        """Initialize integration.

        Args:
            conversation_manager: ConversationManager instance
            role: Current role
        """
        self.conversation = conversation_manager
        self.role = role
        self.doc_roles = DocumentRoles()
        self._role_config = self.doc_roles.get_role_config(role.value)

    async def prepare_document_processing(
            self,
            document: Document,
            context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Prepare conversation manager for document processing.

        Args:
            document: Document to process
            context: Optional additional context
        """
        # Get role-specific system prompt
        system_prompt = self._role_config.system_prompts.get(
            document.doc_type,
            "Analyze this document according to role requirements."
        )

        # Add extraction rules if any
        if self._role_config.extraction_rules:
            system_prompt += "\n\nExtraction Rules:"
            for rule, config in self._role_config.extraction_rules.items():
                system_prompt += f"\n- {rule}: {config}"

        # Add required fields
        if self._role_config.required_fields:
            system_prompt += "\n\nRequired Fields:"
            for field in self._role_config.required_fields:
                system_prompt += f"\n- {field}"

        # Add context if provided
        if context:
            system_prompt += f"\n\nAdditional Context:\n{context}"

        # Set up conversation manager
        await self.conversation.add_system_prompt(system_prompt)
        await self.conversation.add_document(
            content=document.content,
            name=document.name,
            format=document.format.value
        )

    def create_processing_metadata(
            self,
            document: Document
    ) -> Dict[str, Any]:
        """Create metadata for document processing.

        Args:
            document: Document being processed

        Returns:
            Processing metadata
        """
        return {
            "role": self.role.value,
            "doc_type": document.doc_type.value,
            "priorities": self._role_config.priorities,
            "required_fields": self._role_config.required_fields,
            "extraction_rules": self._role_config.extraction_rules
        }

    async def format_context_prompt(
            self,
            document: Document,
            context_vars: Dict[str, str]
    ) -> str:
        """Format context-specific prompt.

        Args:
            document: Document being processed
            context_vars: Variables for prompt formatting

        Returns:
            Formatted prompt
        """
        base_prompt = self._role_config.context_prompts.get(
            document.doc_type,
            "Analyze document in current context."
        )

        try:
            return base_prompt.format(**context_vars)
        except KeyError:
            return base_prompt

    def update_context(
            self,
            context: Dict[str, Any],
    ):
        """Update context with processing results.

        Args:
            context: Context updates
        """
        self.conversation.update_context(context)

    @staticmethod
    def create_bedrock_config(
            model_id: Optional[str] = None,
            **kwargs
    ) -> BedrockConfig:
        """Create Bedrock configuration with defaults.

        Args:
            model_id: Optional model ID override
            **kwargs: Additional configuration options

        Returns:
            Bedrock configuration
        """
        config_dict = {
            "model_id": model_id
        } if model_id else {}

        config_dict.update(kwargs)
        return BedrockConfig(**config_dict)