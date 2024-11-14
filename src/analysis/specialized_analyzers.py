"""Specialized analyzer implementations."""
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
from collections import defaultdict

from src.conversation.manager import ConversationManager
from src.context.types import ContextEntry

from .types import (
    AnalysisType,
    AnalysisInsight,
    AnalysisPriority
)
from .registry import BaseAnalyzer, analyzer_registry


class EngagementAnalyzer(BaseAnalyzer):
    """Analyzes conversation engagement levels."""

    async def analyze(
            self,
            content: Dict[str, Any],
            context: Optional[ContextEntry] = None,
            task_config: Optional[Dict[str, Any]] = None
    ) -> List[AnalysisInsight]:
        """Analyze engagement.

        Args:
            content: Content to analyze
            context: Optional context
            task_config: Optional config

        Returns:
            Engagement insights
        """
        insights = []

        # Extract conversation turns
        turns = content.get("turns", [])

        # Get AI-based engagement analysis
        ai_prompt = (
            "Analyze the conversation engagement level. Consider:\n"
            "1. Participant responsiveness\n"
            "2. Turn-taking patterns\n"
            "3. Response depth and relevance\n"
            "4. Active listening indicators\n\n"
            f"Conversation turns: {turns}"
        )

        expected_format = {
            "engagement_score": "float (0-1)",
            "participation_balance": "float (0-1)",
            "interaction_quality": [
                {
                    "aspect": "string",
                    "score": "float (0-1)",
                    "evidence": ["list of string"]
                }
            ],
            "recommendations": ["list of string"]
        }

        ai_analysis = await self._get_ai_analysis(
            ai_prompt,
            expected_format
        )

        # Create AI-based insight
        insights.append(AnalysisInsight(
            type=AnalysisType.ENGAGEMENT,
            content=ai_analysis,
            confidence=0.8,
            source="ai_analysis",
            timestamp=datetime.now()
        ))

        # Add quantitative metrics
        metrics = await self._calculate_engagement_metrics(turns)
        insights.append(AnalysisInsight(
            type=AnalysisType.ENGAGEMENT,
            content=metrics,
            confidence=0.9,
            source="metric_analysis",
            timestamp=datetime.now()
        ))

        return insights

    async def _calculate_engagement_metrics(
            self,
            turns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate engagement metrics.

        Args:
            turns: Conversation turns

        Returns:
            Engagement metrics
        """
        if not turns:
            return {
                "response_rate": 0.0,
                "avg_response_time": 0.0,
                "turn_distribution": {},
                "engagement_patterns": []
            }

        # Calculate metrics
        speaker_turns = defaultdict(int)
        response_times = []
        prev_time = None

        for turn in turns:
            speaker = turn.get("speaker")
            speaker_turns[speaker] += 1

            timestamp = turn.get("timestamp")
            if prev_time and timestamp:
                response_times.append(
                    (timestamp - prev_time).total_seconds()
                )
            prev_time = timestamp

        total_turns = len(turns)
        return {
            "response_rate": len(response_times) / total_turns,
            "avg_response_time": (
                sum(response_times) / len(response_times)
                if response_times else 0
            ),
            "turn_distribution": dict(speaker_turns),
            "engagement_patterns": await self._detect_patterns(turns)
        }

    async def _detect_patterns(
            self,
            turns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect engagement patterns.

        Args:
            turns: Conversation turns

        Returns:
            List of detected patterns
        """
        patterns = []

        # Look for patterns like:
        # - Question-answer sequences
        # - Topic continuation
        # - Engagement drops
        # - Active listening signals

        sequence_length = 3
        for i in range(len(turns) - sequence_length + 1):
            sequence = turns[i:i + sequence_length]
            pattern = await self._analyze_sequence(sequence)
            if pattern:
                patterns.append(pattern)

        return patterns

    async def _analyze_sequence(
            self,
            sequence: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze turn sequence for patterns.

        Args:
            sequence: Sequence of turns

        Returns:
            Detected pattern if any
        """
        # Look for specific patterns
        is_qa = any('?' in turn.get('text', '') for turn in sequence)
        is_topic_continuation = self._check_topic_continuation(sequence)
        has_active_listening = self._check_active_listening(sequence)

        if any([is_qa, is_topic_continuation, has_active_listening]):
            return {
                "type": "qa" if is_qa else (
                    "topic_continuation" if is_topic_continuation
                    else "active_listening"
                ),
                "turns": len(sequence),
                "speakers": list(set(
                    turn.get('speaker') for turn in sequence
                ))
            }
        return None

    def _check_topic_continuation(
            self,
            sequence: List[Dict[str, Any]]
    ) -> bool:
        """Check for topic continuation pattern."""
        # Simple check for repeated key terms
        terms = set()
        for turn in sequence:
            text = turn.get('text', '').lower()
            words = set(re.findall(r'\w+', text))
            if terms and words.intersection(terms):
                return True
            terms.update(words)
        return False

    def _check_active_listening(
            self,
            sequence: List[Dict[str, Any]]
    ) -> bool:
        """Check for active listening signals."""
        active_listening_phrases = {
            "i see", "understood", "right",
            "got it", "makes sense", "exactly"
        }

        for turn in sequence:
            text = turn.get('text', '').lower()
            if any(phrase in text for phrase in active_listening_phrases):
                return True
        return False


class BehavioralAnalyzer(BaseAnalyzer):
    """Analyzes behavioral patterns in conversation."""

    async def analyze(
            self,
            content: Dict[str, Any],
            context: Optional[ContextEntry] = None,
            task_config: Optional[Dict[str, Any]] = None
    ) -> List[AnalysisInsight]:
        """Analyze behavioral patterns.

        Args:
            content: Content to analyze
            context: Optional context
            task_config: Optional config

        Returns:
            Behavioral insights
        """
        insights = []
        text = content.get("text", "")

        # Get AI-based behavioral analysis
        ai_prompt = (
            "Analyze behavioral patterns in this conversation. Consider:\n"
            "1. Communication styles\n"
            "2. Decision-making patterns\n"
            "3. Problem-solving approaches\n"
            "4. Interpersonal dynamics\n\n"
            f"Text: {text}"
        )

        expected_format = {
            "behaviors": [
                {
                    "type": "string",
                    "frequency": "float (0-1)",
                    "context": "string",
                    "impact": "string"
                }
            ],
            "patterns": [
                {
                    "description": "string",
                    "evidence": ["list of string"],
                    "significance": "float (0-1)"
                }
            ],
            "recommendations": ["list of string"]
        }

        ai_analysis = await self._get_ai_analysis(
            ai_prompt,
            expected_format
        )

        # Create behavioral insight
        insights.append(AnalysisInsight(
            type=AnalysisType.BEHAVIORAL,
            content=ai_analysis,
            confidence=0.7,
            source="ai_analysis",
            timestamp=datetime.now()
        ))

        # Add behavioral metrics
        metrics = await self._calculate_behavioral_metrics(text)
        insights.append(AnalysisInsight(
            type=AnalysisType.BEHAVIORAL,
            content=metrics,
            confidence=0.8,
            source="metric_analysis",
            timestamp=datetime.now()
        ))

        return insights

    async def _calculate_behavioral_metrics(
            self,
            text: str
    ) -> Dict[str, Any]:
        """Calculate behavioral metrics.

        Args:
            text: Text to analyze

        Returns:
            Behavioral metrics
        """
        # Communication style indicators
        assertive_words = {
            "definitely", "certainly", "absolutely",
            "must", "should", "will"
        }
        collaborative_words = {
            "we", "together", "let's",
            "agree", "share", "help"
        }
        analytical_words = {
            "analyze", "consider", "evaluate",
            "data", "evidence", "logic"
        }

        words = text.lower().split()

        # Calculate style scores
        total_words = len(words)
        if total_words == 0:
            return {
                "communication_style": "unknown",
                "style_scores": {},
                "interaction_patterns": [],
                "decisiveness": 0.0
            }

        style_scores = {
            "assertive": sum(
                1 for w in words if w in assertive_words
            ) / total_words,
            "collaborative": sum(
                1 for w in words if w in collaborative_words
            ) / total_words,
            "analytical": sum(
                1 for w in words if w in analytical_words
            ) / total_words
        }

        # Determine dominant style
        dominant_style = max(
            style_scores.items(),
            key=lambda x: x[1]
        )[0]

        return {
            "communication_style": dominant_style,
            "style_scores": style_scores,
            "interaction_patterns": await self._detect_interaction_patterns(text),
            "decisiveness": await self._calculate_decisiveness(text)
        }

    async def _detect_interaction_patterns(
            self,
            text: str
    ) -> List[Dict[str, Any]]:
        """Detect interaction patterns.

        Args:
            text: Text to analyze

        Returns:
            List of detected patterns
        """
        patterns = []

        # Check for various interaction patterns
        if self._has_turn_taking(text):
            patterns.append({
                "type": "turn_taking",
                "strength": "high"
            })

        if self._has_active_discussion(text):
            patterns.append({
                "type": "active_discussion",
                "strength": "medium"
            })

        if self._has_problem_solving(text):
            patterns.append({
                "type": "problem_solving",
                "strength": "high"
            })

        return patterns

    async def _calculate_decisiveness(
            self,
            text: str
    ) -> float:
        """Calculate decisiveness score.

        Args:
            text: Text to analyze

        Returns:
            Decisiveness score
        """
        decisive_indicators = {
            "decide", "chosen", "selected",
            "will", "going to", "plan"
        }
        uncertain_indicators = {
            "maybe", "perhaps", "might",
            "could", "possibly", "not sure"
        }

        words = text.lower().split()
        decisive_count = sum(
            1 for w in words if w in decisive_indicators
        )
        uncertain_count = sum(
            1 for w in words if w in uncertain_indicators
        )

        total = decisive_count + uncertain_count
        if total == 0:
            return 0.5

        return decisive_count / total

    def _has_turn_taking(self, text: str) -> bool:
        """Check for turn-taking pattern."""
        # Look for turn indicators
        turn_indicators = {
            "you mentioned", "as you said",
            "to add to that", "building on"
        }
        return any(
            indicator in text.lower()
            for indicator in turn_indicators
        )

    def _has_active_discussion(self, text: str) -> bool:
        """Check for active discussion pattern."""
        # Look for discussion indicators
        discussion_indicators = {
            "what if", "how about",
            "another approach", "alternatively"
        }
        return any(
            indicator in text.lower()
            for indicator in discussion_indicators
        )

    def _has_problem_solving(self, text: str) -> bool:
        """Check for problem-solving pattern."""
        # Look for problem-solving indicators
        problem_solving_indicators = {
            "solution", "resolve", "address",
            "fix", "improve", "optimize"
        }
        return any(
            indicator in text.lower()
            for indicator in problem_solving_indicators
        )


# Register specialized analyzers
analyzer_registry.register(AnalysisType.ENGAGEMENT, EngagementAnalyzer)
analyzer_registry.register(AnalysisType.BEHAVIORAL, BehavioralAnalyzer)