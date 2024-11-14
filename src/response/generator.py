"""Core response generation functionality."""
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime
import asyncio
import json

from src.conversation.manager import ConversationManager
from src.context.manager import ContextManager
from src.analysis.engine import AnalysisEngine
from src.events.bus import EventBus
from src.events.types import Event, EventType

from .types import (
    ResponseType,
    ResponsePriority,
    ResponseCandidate,
    ResponseConfig,
    ResponseResult,
    ResponseRequest
)


class ResponseGenerator:
    """Generates contextually aware responses."""

    def __init__(
        self,
        event_bus: EventBus,
        conversation_manager: ConversationManager,
        context_manager: ContextManager,
        analysis_engine: AnalysisEngine,
        config: Optional[ResponseConfig] = None
    ):
        """Initialize generator.

        Args:
            event_bus: Event bus instance
            conversation_manager: For AI model access
            context_manager: Context manager
            analysis_engine: Analysis engine
            config: Optional configuration
        """
        self.event_bus = event_bus
        self.conversation = conversation_manager
        self.context = context_manager
        self.analysis = analysis_engine
        self.config = config or ResponseConfig()

        # Response tracking
        self.response_history: Dict[str, List[ResponseResult]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    async def generate_response(
        self,
        request: ResponseRequest
    ) -> AsyncIterator[ResponseResult]:
        """Generate response for request.

        Args:
            request: Response request

        Yields:
            Generated responses
        """
        try:
            # Generate candidates
            candidates = await self._generate_candidates(request)

            # Select best candidates
            selected = await self._select_candidates(
                candidates,
                request.config or self.config
            )

            # Generate final response
            response = await self._generate_final_response(
                selected,
                request
            )

            # Track response
            session_id = self._get_session_id(request)
            if session_id not in self.response_history:
                self.response_history[session_id] = []
            self.response_history[session_id].append(response)

            # Emit event
            # Changed EventType.RESPONSE to EventType.RESPONSE_RECEIVED
            await self.event_bus.publish(Event(
                type=EventType.RESPONSE_RECEIVED,
                data={
                    "status": "response_generated",
                    "type": response.type,
                    "confidence": response.confidence
                }
            ))

            yield response

        except Exception as e:
            # Generate fallback response
            fallback = await self._generate_fallback(request, str(e))
            yield fallback

    async def _generate_candidates(
        self,
        request: ResponseRequest
    ) -> List[ResponseCandidate]:
        """Generate response candidates.

        Args:
            request: Response request

        Returns:
            Response candidates
        """
        candidates = []

        # Generate AI-based candidates
        ai_prompt = self._create_response_prompt(request)
        ai_candidates = await self._get_ai_candidates(ai_prompt)
        candidates.extend(ai_candidates)

        # Generate template-based candidates
        template_candidates = await self._get_template_candidates(request)
        candidates.extend(template_candidates)

        return candidates

    async def _get_ai_candidates(
        self,
        prompt: str
    ) -> List[ResponseCandidate]:
        """Get AI-generated candidates.

        Args:
            prompt: Generation prompt

        Returns:
            Generated candidates
        """
        try:
            responses = []
            async for response in self.conversation.send_message(prompt):
                if response.text:
                    responses.append(response.text)

            response_text = ''.join(responses)

            try:
                # Parse structured response
                data = json.loads(response_text)
                return [
                    ResponseCandidate(
                        content=c["content"],
                        type=ResponseType(c.get("type", "direct")),
                        confidence=c.get("confidence", 0.5),
                        context_refs=c.get("context_refs", []),
                        metadata=c.get("metadata", {})
                    )
                    for c in data.get("candidates", [])
                ]
            except json.JSONDecodeError:
                # Handle unstructured response
                return [
                    ResponseCandidate(
                        content=response_text,
                        type=ResponseType.DIRECT,
                        confidence=0.5
                    )
                ]

        except Exception as e:
            print(f"Error getting AI candidates: {e}")
            return []

    async def _get_template_candidates(
        self,
        request: ResponseRequest
    ) -> List[ResponseCandidate]:
        """Get template-based candidates.

        Args:
            request: Response request

        Returns:
            Generated candidates
        """
        candidates = []

        # Load templates for response type
        response_type = request.response_type or self.config.default_type
        templates = self._get_templates(response_type)

        for template in templates:
            try:
                # Fill template with context
                content = await self._fill_template(
                    template,
                    request
                )
                if content:
                    candidates.append(
                        ResponseCandidate(
                            content=content,
                            type=response_type,
                            confidence=0.7,  # Template confidence
                            context_refs=[],  # Add relevant refs
                            metadata={"source": "template"}
                        )
                    )
            except Exception as e:
                print(f"Template fill error: {e}")
                continue

        return candidates

    def _get_templates(
        self,
        response_type: ResponseType
    ) -> List[str]:
        """Get templates for response type.

        Args:
            response_type: Type of response

        Returns:
            List of templates
        """
        # Template registry - could be moved to config/database
        templates = {
            ResponseType.CLARIFYING: [
                "Could you clarify what you mean by {topic}?",
                "I'm not sure I understand about {topic}. Can you explain?",
                "Just to make sure I understand correctly: {context}?"
            ],
            ResponseType.FOLLOW_UP: [
                "Based on {context}, would you like to know more about {topic}?",
                "That's interesting. How do you feel about {topic}?",
                "Could you tell me more about {aspect}?"
            ],
            ResponseType.SUGGESTION: [
                "Have you considered {suggestion}?",
                "You might want to try {suggestion}.",
                "Based on {context}, I recommend {suggestion}."
            ],
            ResponseType.SUMMARY: [
                "To summarize: {summary}",
                "Here's what we've covered: {summary}",
                "The main points are: {summary}"
            ]
        }
        return templates.get(response_type, [])

    async def _fill_template(
        self,
        template: str,
        request: ResponseRequest
    ) -> Optional[str]:
        """Fill template with context.

        Args:
            template: Template to fill
            request: Response request

        Returns:
            Filled template if successful
        """
        try:
            # Extract variables from template
            vars_needed = {
                var.strip('{}')
                for var in template.split()
                if var.startswith('{') and var.endswith('}')
            }

            # Get values from context
            values = {}
            if request.context:
                for var in vars_needed:
                    value = await self._extract_value(
                        var,
                        request.context,
                        request.analysis
                    )
                    if value:
                        values[var] = value

            # Fill template if we have all values
            if len(values) == len(vars_needed):
                return template.format(**values)
            return None

        except Exception as e:
            print(f"Template fill error: {e}")
            return None

    async def _select_candidates(
        self,
        candidates: List[ResponseCandidate],
        config: ResponseConfig
    ) -> List[ResponseCandidate]:
        """Select best candidates.

        Args:
            candidates: All candidates
            config: Response configuration

        Returns:
            Selected candidates
        """
        if not candidates:
            return []

        # Filter by confidence
        viable = [
            c for c in candidates
            if c.confidence >= config.min_confidence
        ]

        # Sort by confidence
        sorted_candidates = sorted(
            viable,
            key=lambda x: x.confidence,
            reverse=True
        )

        # Return top N
        return sorted_candidates[:config.max_candidates]

    async def _generate_final_response(
        self,
        candidates: List[ResponseCandidate],
        request: ResponseRequest
    ) -> ResponseResult:
        """Generate final response.

        Args:
            candidates: Selected candidates
            request: Original request

        Returns:
            Final response
        """
        if not candidates:
            return await self._generate_fallback(
                request,
                "No viable candidates"
            )

        # Use the best candidate as main response
        best = candidates[0]
        return ResponseResult(
            content=best.content,
            type=best.type,
            confidence=best.confidence,
            alternatives=candidates[1:],
            context_used=best.context_refs,
            analysis_used=[],  # Add analysis refs
            metadata={
                **best.metadata,
                "request_type": request.response_type,
                "candidates_generated": len(candidates)
            }
        )

    async def _generate_fallback(
        self,
        request: ResponseRequest,
        error: str
    ) -> ResponseResult:
        """Generate fallback response.

        Args:
            request: Original request
            error: Error message

        Returns:
            Fallback response
        """
        # Get fallback for type
        response_type = request.response_type or self.config.default_type
        fallbacks = self.config.fallback_responses.get(
            response_type,
            ["I'm not sure how to respond to that."]
        )

        return ResponseResult(
            content=fallbacks[0],
            type=ResponseType.FALLBACK,
            confidence=0.5,
            alternatives=[],
            context_used=[],
            analysis_used=[],
            metadata={
                "error": error,
                "original_type": response_type
            }
        )

    def _get_session_id(self, request: ResponseRequest) -> str:
        """Get session ID from request.

        Args:
            request: Response request

        Returns:
            Session identifier
        """
        # Extract from context or generate
        if request.context:
            return request.context.metadata.get(
                "session_id",
                str(datetime.now().timestamp())
            )
        return str(datetime.now().timestamp())

    async def _extract_value(
        self,
        variable: str,
        context: Any,
        analysis: Optional[AnalysisResult]
    ) -> Optional[str]:
        """Extract template variable value.

        Args:
            variable: Variable to extract
            context: Context entry
            analysis: Optional analysis result

        Returns:
            Extracted value if found
        """
        # Add value extraction logic
        return None  # Placeholder