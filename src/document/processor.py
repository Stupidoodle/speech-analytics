from typing import Dict, Any, List, Optional, Callable
import json
from ..conversation.manager import ConversationManager


class DocumentProcessor:
    """Processes documents into structured data for context."""

    def __init__(self, conversation_manager: ConversationManager):
        self.ai = conversation_manager
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        self.update_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

    async def add_update_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Add callback for context updates."""
        self.update_callbacks.append(callback)

    async def process_document(
        self,
        content: str,
        doc_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process document using AI analysis."""
        prompt = self._create_processing_prompt(content, doc_type, metadata)

        async for response in self.ai.send_message(prompt):
            if response.text:
                try:
                    analysis = json.loads(response.text)
                    self.context_cache[doc_type] = analysis
                    await self._notify_update(doc_type, analysis)
                    return analysis
                except json.JSONDecodeError:
                    continue

        return {}

    def _create_processing_prompt(
        self,
        content: str,
        doc_type: str,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Create context-aware document processing prompt."""
        if doc_type == "cv":
            return (
                "Analyze this CV/resume and extract key information. "
                "Provide analysis in JSON format including:\n"
                "1. Key skills and expertise levels\n"
                "2. Experience timeline and highlights\n"
                "3. Educational background\n"
                "4. Project highlights\n"
                "5. Notable achievements\n"
                "6. Technical proficiencies\n\n"
                f"Document content:\n{content}"
            )
        elif doc_type == "job_description":
            return (
                "Analyze this job description and extract key information. "
                "Provide analysis in JSON format including:\n"
                "1. Required skills and experience levels\n"
                "2. Key responsibilities\n"
                "3. Qualifications needed\n"
                "4. Company context\n"
                "5. Role objectives\n\n"
                f"Document content:\n{content}"
            )
        else:
            return (
                "Analyze this document and extract key information. "
                "Provide analysis in JSON format including:\n"
                "1. Main topics and themes\n"
                "2. Key points and insights\n"
                "3. Important details\n"
                "4. Context and relevance\n"
                "5. Action items or next steps\n\n"
                f"Document content:\n{content}"
            )

    async def update_context(
        self,
        doc_type: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update existing context with new information using AI."""
        existing_context = self.context_cache.get(doc_type, {})

        prompt = (
            "Update the existing context with new information. "
            "Provide updated analysis in JSON format.\n\n"
            f"Existing context:\n{json.dumps(existing_context, indent=2)}\n\n"
            f"New information:\n{new_content}"
        )

        async for response in self.ai.send_message(prompt):
            if response.text:
                try:
                    updated_analysis = json.loads(response.text)
                    self.context_cache[doc_type] = updated_analysis
                    await self._notify_update(doc_type, updated_analysis)
                    return updated_analysis
                except json.JSONDecodeError:
                    continue

        return existing_context

    async def _notify_update(
        self,
        doc_type: str,
        analysis: Dict[str, Any]
    ) -> None:
        """Notify callbacks of context updates."""
        for callback in self.update_callbacks:
            try:
                await callback(doc_type, analysis)
            except Exception as e:
                print(f"Error in update callback: {e}")


class DocumentProcessingError(Exception):
    """Exception raised for document processing errors."""
    pass
