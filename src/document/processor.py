"""Document processor using direct messaging with role-based formats."""

import json
from typing import Dict, Any, AsyncIterator
from datetime import datetime

from src.conversation.manager import ConversationManager
from src.context.manager import ContextManager
from src.events.bus import EventBus
from src.events.types import Event, EventType
from src.conversation.types import (
    Message,
    Request,
    ContentBlock,
    Source,
    Document,
    SystemContent,
)

from .types import Document, ProcessingContext, ProcessingResult
from .roles import DocumentRoles


class DocumentProcessor:
    """Process documents using direct document messaging with role formats."""

    def __init__(
        self,
        conversation_manager: ConversationManager,
        context_manager: ContextManager,
        event_bus: EventBus,
    ):
        self.conversation = conversation_manager
        self.context = context_manager
        self.event_bus = event_bus
        self.doc_roles = DocumentRoles()

    async def process_document(
        self,
        document: Document,
        context: ProcessingContext,
    ) -> ProcessingResult:
        """Process document by sending it directly with role-specific format.

        Args:
            document: Document to process
            context: Processing context
        """
        try:
            # Get role-specific format and prompt
            role_config = self.doc_roles.get_role_config(context.role)
            response_format = role_config.response_format.get(document.doc_type, "{}")
            system_prompt = role_config.system_prompts.get(document.doc_type)

            request = Request(
                messages=[
                    Message(
                        role="user",
                        content=[
                            ContentBlock(
                                text=f"Analyze this {document.doc_type.value}",
                                document=Document(
                                    format=document.format.value,
                                    name=document.name,
                                    source=Source(
                                        bytes=document.content,
                                    ),
                                ),
                            )
                        ],
                    )
                ],
                system=[
                    SystemContent(
                        text=(
                            f"{system_prompt} "
                            f"Provide your analysis in exactly this JSON format: "
                            f"{response_format} "
                            f"Return only valid JSON without any additional text."
                        )
                    )
                ],
            )

            # Process through conversation
            start_time = datetime.now()
            content = ""
            async for response in self.conversation.send_message(
                message=request,
                client_type="pre_processing",
            ):
                if isinstance(response, str):
                    # Create result
                    content += response

            result = ProcessingResult(
                document_id=document.name,
                content_type=document.doc_type.value,
                analysis=content,  # Will be JSON from role format
                processing_time=(datetime.now() - start_time).total_seconds(),
                context_updates=self._create_context_updates(
                    content, document.doc_type, context.role
                ),
                role_specific={
                    "role": context.role,
                    "format_used": response_format,
                },
                extracted_at=datetime.now(),
            )

            # Emit event
            await self.event_bus.publish(
                Event(
                    type=EventType.DOCUMENT_PROCESSED,
                    data={
                        "status": "processing_complete",
                        "document_id": document.name,
                        "result": result.model_dump(),
                    },
                )
            )

            return result

        except Exception as e:
            await self.event_bus.publish(
                Event(
                    type=EventType.ERROR,
                    data={
                        "error": str(e),
                        "document_id": document.name,
                        "processing_stage": "document_processing",
                    },
                )
            )
            raise

    @staticmethod
    def _create_context_updates(
        analysis: str, doc_type: str, role: str
    ) -> Dict[str, Any]:
        """Create context updates from analysis."""
        try:
            # Parse the JSON response
            parsed = json.loads(analysis)

            # Create context updates based on document type and role
            updates = {
                "document_type": doc_type,
                "role": role,
                "timestamp": datetime.now().isoformat(),
                "analysis": parsed,  # Full analysis
            }

            # Add type-specific extracts
            if doc_type == "cv":
                updates["skills"] = parsed.get("technical_skills", [])
                updates["experience"] = parsed.get("experience", [])
            elif doc_type == "job_description":
                updates["requirements"] = parsed.get("key_requirements", [])
                updates["responsibilities"] = parsed.get("responsibilities", [])

            return updates

        except json.JSONDecodeError:
            # If JSON parsing fails, store raw analysis
            return {
                "document_type": doc_type,
                "role": role,
                "timestamp": datetime.now().isoformat(),
                "raw_analysis": analysis,
            }
