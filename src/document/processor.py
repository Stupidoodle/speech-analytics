"""Document processor using Bedrock with role-based processing."""
from typing import Dict, Any, AsyncIterator
import json
import time
from datetime import datetime
import uuid

from ..conversation.manager import ConversationManager
from .roles import DocumentRoles
from ..common.types import Role, Document, DocumentType
from ..integration import ProcessingIntegration
from .types import (
    DocumentType,
    ProcessingContext,
    ProcessingResult,
    ProcessingPriority,
    DocumentReference,
    ContextUpdate
)
from .exceptions import (
    DocumentError,
    RoleConfigError,
    AIProcessingError,
)
from src.events.bus import EventBus
from src.events.types import Event, EventType


class DocumentProcessor:
    """Process documents using Bedrock with role-based analysis."""

    def __init__(self, conversation_manager: ConversationManager, event_bus: EventBus):
        """Initialize processor.

        Args:
            conversation_manager: For Bedrock API access
            event_bus: For event publishing
        """
        self.conversation = conversation_manager
        self.event_bus = event_bus
        self.integration = None

    async def process_document(
        self,
        content: bytes,
        context: ProcessingContext
    ) -> AsyncIterator[ProcessingResult]:
        """Process document with role-specific analysis.

        Args:
            content: Document content
            context: Processing context with role information

        Yields:
            Processing results and updates

        Raises:
            RoleConfigError: If role configuration is invalid
            AIProcessingError: If AI processing fails
            DocumentError: For other processing errors
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())

        self.integration = ProcessingIntegration(
            conversation_manager=self.conversation,
            role=Role(context.role)  # Convert to unified Role enum
        )

        document = Document(
            content=content,
            format=context.document_type.name.lower(),  # Map to DocumentFormat
            name=context.metadata.get("filename", "document"),
            doc_type=context.document_type
        )

        # Emit event at the start of processing
        await self.event_bus.publish(Event(
            type=EventType.DOCUMENT_PROCESSED,
            data={
                "document_id": document_id,
                "status": "started",
            }
        ))

        try:
            # Prepare conversation manager
            await self.integration.prepare_document_processing(
                document=document,
                context=context.metadata
            )

            # Create processing metadata
            metadata = self.integration.create_processing_metadata(
                document=document
            )

            async for response in self.conversation.send_message(
                text="Process document based on role and context",
                files=[{
                    "document": {
                        "format": document.format,
                        "name": document.name,
                        "source": {"bytes": document.content}
                    }
                }],
                metadata=metadata
            ):
                if response.text:
                    try:
                        # Parse analysis
                        ref_data = json.loads(response.text)

                        # Create result
                        result = ProcessingResult(
                            document_id=document_id,
                            content_type=context.document_type.value,
                            analysis=ref_data.get("analysis", {}),
                            confidence=ref_data.get("confidence", 0.0),
                            processing_time=time.time() - start_time,
                            context_updates=self._create_context_updates(
                                ref_data,
                                context
                            ),
                            role_specific=ref_data.get("role_specific", {}),
                            extracted_at=datetime.now()
                        )

                        # Update conversation context
                        self.integration.update_context(
                            result.context_updates
                        )

                        # Emit processing event
                        await self.event_bus.publish(Event(
                            type=EventType.DOCUMENT_PROCESSED,
                            data={
                                "document_id": document_id,
                                "status": "result_yielded",
                                "result": result.model_dump()
                            }
                        ))

                        yield result

                    except json.JSONDecodeError as e:
                        raise AIProcessingError(
                            "Failed to parse AI response",
                            str(e)
                        )

            # Emit completion event
            await self.event_bus.publish(Event(
                type=EventType.DOCUMENT_PROCESSED,
                data={
                    "document_id": document_id,
                    "status": "completed",
                    "processing_time": time.time() - start_time
                }
            ))

        except Exception as e:
            # Emit error events
            await self.event_bus.publish(Event(
                type=EventType.ERROR,
                data={
                    "document_id": document_id,
                    "error": str(e),
                    "context": context.model_dump()
                }
            ))

            if isinstance(e, (RoleConfigError, AIProcessingError)):
                raise
            raise DocumentError(f"Processing failed: {str(e)}")

    async def get_document_references(
        self,
        doc_ids: list[str],
        context: ProcessingContext
    ) -> AsyncIterator[DocumentReference]:
        """Get referenced documents with role-specific context.

        Args:
            doc_ids: Document IDs to reference
            context: Current processing context

        Yields:
            Document references with role-specific information
        """
        try:
            # Ensure integration is initialized
            if not self.integration:
                self.integration = ProcessingIntegration(
                    conversation_manager=self.conversation,
                    role=Role(context.role)
                )

            for doc_id in doc_ids:
                # Format context prompt
                context_prompt = await self.integration.format_context_prompt(
                    document=Document(  # Placeholder for reference
                        content=b"",
                        format="unknown",
                        name=doc_id,
                        doc_type=context.document_type,
                    ),
                    context_vars=context.metadata,
                )

                # Process with conversation manager
                async for response in self.conversation.send_message(
                    text=f"Get reference information for document {doc_id}",
                    metadata={
                        "type": "reference",
                        "document_id": doc_id,
                        "context": context.model_dump(),
                    }
                ):
                    if response.text:
                        try:
                            ref_data = json.loads(response.text)
                            yield DocumentReference(
                                id=doc_id,
                                type=context.document_type,
                                name=ref_data.get("name", ""),
                                summary=ref_data.get("summary", ""),
                                key_points=ref_data.get("key_points", []),
                                relevance=ref_data.get("relevance", 0.0),
                                context=ref_data.get("context", {})
                            )
                        except json.JSONDecodeError as e:
                            raise AIProcessingError(
                                "Failed to parse reference data",
                                str(e)
                            )

        except Exception as e:
            if isinstance(e, (RoleConfigError, AIProcessingError)):
                raise
            raise DocumentError(f"Reference processing failed: {str(e)}")

    def _get_system_prompt(
        self,
        role_config: Any,
        doc_type: DocumentType,
        context: ProcessingContext
    ) -> str:
        """Get role and context specific system prompt.

        Args:
            role_config: Role configuration
            doc_type: Document type
            context: Processing context

        Returns:
            System prompt for AI
        """
        # Get base prompt for role and document type
        base_prompt = role_config.system_prompts.get(
            doc_type,
            "Analyze this document according to role requirements."
        )

        # Add context-specific instructions
        additional_context = [
            f"Purpose: {context.purpose}",
            f"Priority: {context.priority.value}"
        ]

        # Add extraction rules
        if role_config.extraction_rules:
            additional_context.append("\nExtraction Rules:")
            for rule, config in role_config.extraction_rules.items():
                additional_context.append(f"- {rule}: {config}")

        # Add required fields
        if role_config.required_fields:
            additional_context.append("\nRequired Fields:")
            for field in role_config.required_fields:
                additional_context.append(f"- {field}")

        # Combine prompts
        full_prompt = f"""{base_prompt}

Context:
{chr(10).join(additional_context)}

Provide output as structured JSON with:
1. analysis: Detailed document analysis
2. confidence: Overall confidence score (0-1)
3. role_specific: Role-specific analysis and insights
4. context_updates: Suggested context updates
"""
        return full_prompt

    def _get_context_prompt(
        self,
        role_config: Any,
        doc_type: DocumentType,
        context: ProcessingContext
    ) -> str:
        """Get context-specific prompt.

        Args:
            role_config: Role configuration
            doc_type: Document type
            context: Processing context

        Returns:
            Context prompt for AI
        """
        # Get base context prompt
        base_prompt = role_config.context_prompts.get(
            doc_type,
            "Analyze document in current context."
        )

        # Add context variables
        try:
            return base_prompt.format(**context.metadata)
        except KeyError:
            # Fall back to base prompt if format fails
            return base_prompt

    def _create_context_updates(
        self,
        analysis: Dict[str, Any],
        role_config: Any,
        # context: ProcessingContext
    ) -> Dict[str, Any]:
        """Create context updates from analysis.

        Args:
            analysis: AI analysis results
            role_config: Role configuration
            # context: Processing context

        Returns:
            Context updates
        """
        updates = {}

        # Process priority-based updates
        for priority in role_config.priorities:
            if priority in analysis:
                updates[priority] = ContextUpdate(
                    type=priority,
                    content=analysis[priority],
                    priority=ProcessingPriority.HIGH
                    if priority in role_config.required_fields
                    else ProcessingPriority.MEDIUM,
                    requires_action=self._requires_action(
                        analysis[priority],
                        # role_config
                    )
                )

        # Add role-specific updates
        if "role_specific" in analysis:
            updates["role_specific"] = analysis["role_specific"]

        return updates

    def _requires_action(
        self,
        content: Dict[str, Any],
        # role_config: Any
    ) -> bool:
        """Determine if content requires action.

        Args:
            content: Content to check
            # role_config: Role configuration

        Returns:
            Whether action is required
        """
        # Implement action detection logic
        # This could be enhanced with AI-based detection
        return any(
            keyword in str(content).lower()
            for keyword in [
                "action",
                "required",
                "needed",
                "todo",
                "follow-up"
            ]
        )
