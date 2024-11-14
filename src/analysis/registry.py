"""Analyzer registry and base analyzer implementation."""
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod

from src.conversation.manager import ConversationManager
from src.context.types import ContextEntry

from .types import (
    AnalysisType,
    AnalysisInsight,
    AnalysisConfig
)
from .exceptions import (
    AnalyzerNotFoundError,
    AnalysisError
)


class BaseAnalyzer(ABC):
    """Base class for all analyzers."""

    def __init__(
            self,
            conversation_manager: ConversationManager,
            config: Optional[Dict[str, Any]] = None
    ):
        """Initialize analyzer.

        Args:
            conversation_manager: For AI model access
            config: Optional analyzer configuration
        """
        self.conversation = conversation_manager
        self.config = config or {}

    @abstractmethod
    async def analyze(
            self,
            content: Dict[str, Any],
            context: Optional[ContextEntry] = None,
            task_config: Optional[Dict[str, Any]] = None
    ) -> List[AnalysisInsight]:
        """Perform analysis.

        Args:
            content: Content to analyze
            context: Optional context entry
            task_config: Optional task-specific config

        Returns:
            Analysis insights

        Raises:
            AnalysisError: If analysis fails
        """
        pass

    async def _get_ai_analysis(
            self,
            prompt: str,
            max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Get analysis from AI model.

        Args:
            prompt: Analysis prompt
            max_tokens: Maximum tokens for response

        Returns:
            AI analysis result

        Raises:
            AnalysisError: If AI analysis fails
        """
        try:
            responses = []
            async for response in self.conversation.send_message(prompt):
                if response.text:
                    responses.append(response.text)

            analysis = ''.join(responses)

            # Parse analysis into structured format
            # This would depend on how we format our prompts and responses
            # For now, assuming JSON responses
            try:
                import json
                return json.loads(analysis)
            except json.JSONDecodeError:
                return {"text": analysis}

        except Exception as e:
            raise AnalysisError(
                f"AI analysis failed: {str(e)}"
            )


class AnalyzerRegistry:
    """Registry for analyzer implementations."""

    def __init__(self):
        """Initialize registry."""
        self._analyzers: Dict[
            AnalysisType,
            Type[BaseAnalyzer]
        ] = {}

    def register(
            self,
            analyzer_type: AnalysisType,
            analyzer_class: Type[BaseAnalyzer]
    ) -> None:
        """Register analyzer implementation.

        Args:
            analyzer_type: Type of analyzer
            analyzer_class: Analyzer implementation class
        """
        self._analyzers[analyzer_type] = analyzer_class

    def get_analyzer(
            self,
            analyzer_type: AnalysisType,
            conversation_manager: ConversationManager,
            config: Optional[Dict[str, Any]] = None
    ) -> BaseAnalyzer:
        """Get analyzer instance.

        Args:
            analyzer_type: Type of analyzer
            conversation_manager: For AI model access
            config: Optional analyzer configuration

        Returns:
            Analyzer instance

        Raises:
            AnalyzerNotFoundError: If analyzer not found
        """
        if analyzer_type not in self._analyzers:
            raise AnalyzerNotFoundError(
                f"Analyzer not found: {analyzer_type}",
                analyzer_type.value,
                list(self._analyzers.keys())
            )

        analyzer_class = self._analyzers[analyzer_type]
        return analyzer_class(conversation_manager, config)


# Global registry instance
analyzer_registry = AnalyzerRegistry()