"""Specific analyzer implementations."""
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
from collections import Counter

from src.conversation.manager import ConversationManager
from src.context.types import ContextEntry

from .types import (
    AnalysisType,
    AnalysisInsight,
    AnalysisPriority
)
from .registry import BaseAnalyzer, analyzer_registry


class SentimentAnalyzer(BaseAnalyzer):
    """Analyzes sentiment in content."""

    async def analyze(
            self,
            content: Dict[str, Any],
            context: Optional[ContextEntry] = None,
            task_config: Optional[Dict[str, Any]] = None
    ) -> List[AnalysisInsight]:
        """Analyze sentiment.

        Args:
            content: Content to analyze
            context: Optional context
            task_config: Optional config

        Returns:
            Sentiment insights
        """
        insights = []
        text = content.get("text", "")

        # Get AI-based sentiment analysis
        ai_prompt = (
            "Analyze the sentiment in this text. Consider:\n"
            "1. Overall sentiment polarity (positive/negative/neutral)\n"
            "2. Confidence in analysis (0-1)\n"
            "3. Specific sentiment indicators (words/phrases)\n"
            "4. Emotional undertones\n"
            "5. Sentiment intensity\n"
            "6. Key sentiment-bearing phrases\n\n"
            f"Text: {text}\n\n"
        )

        expected_format = {
            "sentiment": "string (positive/negative/neutral)",
            "confidence": "float (0-1)",
            "indicators": ["list of string"],
            "emotions": ["list of string"],
            "intensity": "float (0-1)",
            "key_phrases": ["list of string"]
        }

        ai_analysis = await self._get_ai_analysis(
            ai_prompt, expected_format
        )

        # Create sentiment insight
        insights.append(AnalysisInsight(
            type=AnalysisType.SENTIMENT,
            content={
                "sentiment": ai_analysis.get("sentiment", "neutral"),
                "indicators": ai_analysis.get("indicators", []),
                "emotions": ai_analysis.get("emotions", [])
            },
            confidence=ai_analysis.get("confidence", 0.5),
            source="ai_analysis",
            timestamp=datetime.now()
        ))

        # Add basic sentiment metrics
        insights.append(AnalysisInsight(
            type=AnalysisType.SENTIMENT,
            content=await self._calculate_sentiment_metrics(text),
            confidence=0.7,
            source="metric_analysis",
            timestamp=datetime.now()
        ))

        return insights

    async def _calculate_sentiment_metrics(
            self,
            text: str
    ) -> Dict[str, Any]:
        """Calculate basic sentiment metrics.

        Args:
            text: Text to analyze

        Returns:
            Sentiment metrics
        """
        # Simple word-based sentiment scoring
        positive_words = {
            "good", "great", "excellent", "happy", "positive",
            "wonderful", "fantastic", "amazing", "helpful"
        }
        negative_words = {
            "bad", "poor", "terrible", "unhappy", "negative",
            "awful", "horrible", "useless", "disappointing"
        }

        words = text.lower().split()
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)

        total = positive_count + negative_count
        if total == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / total

        return {
            "sentiment_score": sentiment_score,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "word_count": len(words)
        }


class TopicAnalyzer(BaseAnalyzer):
    """Analyzes topics in content."""

    async def analyze(
            self,
            content: Dict[str, Any],
            context: Optional[ContextEntry] = None,
            task_config: Optional[Dict[str, Any]] = None
    ) -> List[AnalysisInsight]:
        """Analyze topics.

        Args:
            content: Content to analyze
            context: Optional context
            task_config: Optional config

        Returns:
            Topic insights
        """
        insights = []
        text = content.get("text", "")

        # Get AI-based topic analysis
        ai_prompt = (
            "Analyze the main topics in this text, providing:\n"
            "1. List of topics (name, relevance (0-1), mentions, related terms)\n"
            "2. Relationships between topics (strength, type)\n"
            "3. Importance of each topic\n\n"
            f"Text: {text}"
        )

        expected_format = {
            "topics": [{
                "name": "string",
                "relevance": "float (0-1)",
                "mentions": "integer",
                "related_terms": ["list of string"]
            }],
            "relationships": [{
                "topic1": "string",
                "topic2": "string",
                "strength": "float (0-1)",
                "type": "string"
            }],
            "importance": {
                "topic_name": "float (0-1)"
            }
        }

        ai_analysis = await self._get_ai_analysis(
            ai_prompt,
            expected_format
        )

        # Create topic insight
        insights.append(AnalysisInsight(
            type=AnalysisType.TOPIC,
            content={
                "topics": ai_analysis.get("topics", []),
                "relationships": ai_analysis.get("relationships", []),
                "importance": ai_analysis.get("importance", {})
            },
            confidence=ai_analysis.get("confidence", 0.5),
            source="ai_analysis",
            timestamp=datetime.now()
        ))

        # Add statistical topic analysis
        insights.append(AnalysisInsight(
            type=AnalysisType.TOPIC,
            content=await self._analyze_topic_distribution(text),
            confidence=0.8,
            source="statistical_analysis",
            timestamp=datetime.now()
        ))

        return insights

    async def _analyze_topic_distribution(
            self,
            text: str
    ) -> Dict[str, Any]:
        """Analyze topic distribution.

        Args:
            text: Text to analyze

        Returns:
            Topic distribution metrics
        """
        # Remove common words and get word frequencies
        common_words = {
            "the", "be", "to", "of", "and", "a", "in", "that",
            "have", "i", "it", "for", "not", "on", "with", "he",
            "as", "you", "do", "at"
        }

        words = [
            word.lower() for word in re.findall(r'\w+', text)
            if word.lower() not in common_words
        ]

        # Get word frequencies
        word_freq = Counter(words)
        top_words = word_freq.most_common(10)

        # Group similar words (simple stemming)
        grouped_topics = {}
        for word, count in top_words:
            stem = word[:4]  # Simple prefix grouping
            if stem in grouped_topics:
                grouped_topics[stem]["count"] += count
                grouped_topics[stem]["words"].append(word)
            else:
                grouped_topics[stem] = {
                    "count": count,
                    "words": [word]
                }

        return {
            "top_words": dict(top_words),
            "topic_groups": grouped_topics,
            "total_words": len(words)
        }


class QualityAnalyzer(BaseAnalyzer):
    """Analyzes conversation quality."""

    async def analyze(
            self,
            content: Dict[str, Any],
            context: Optional[ContextEntry] = None,
            task_config: Optional[Dict[str, Any]] = None
    ) -> List[AnalysisInsight]:
        """Analyze conversation quality.

        Args:
            content: Content to analyze
            context: Optional context
            task_config: Optional config

        Returns:
            Quality insights
        """
        insights = []
        text = content.get("text", "")

        # Get AI-based quality analysis
        ai_prompt = (
            "Analyze the conversation quality, providing:\n"
            "1. Clarity score (0-1)\n"
            "2. Engagement level (0-1)\n"
            "3. Communication effectiveness (0-1)\n"
            "4. Suggestions for improvement (aspects, suggestions, priorities(high/medium/low)\n"
            "5. Metrics for coherence, relevance, and completeness\n\n"
            f"Text: {text}"
        )

        expected_format = {
            "clarity": "float (0-1)",
            "engagement": "float (0-1)",
            "effectiveness": "float (0-1)",
            "improvements": [{
                "aspect": "string",
                "suggestion": "string",
                "priority": "string (high/medium/low)"
            }],
            "metrics": {
                "coherence": "float (0-1)",
                "relevance": "float (0-1)",
                "completeness": "float (0-1)"
            }
        }

        ai_analysis = await self._get_ai_analysis(
            ai_prompt,
            expected_format
        )

        # Create quality insight
        insights.append(AnalysisInsight(
            type=AnalysisType.QUALITY,
            content={
                "clarity": ai_analysis.get("clarity", 0.0),
                "engagement": ai_analysis.get("engagement", 0.0),
                "effectiveness": ai_analysis.get("effectiveness", 0.0),
                "improvements": ai_analysis.get("improvements", [])
            },
            confidence=ai_analysis.get("confidence", 0.5),
            source="ai_analysis",
            timestamp=datetime.now()
        ))

        # Add measurable quality metrics
        insights.append(AnalysisInsight(
            type=AnalysisType.QUALITY,
            content=await self._calculate_quality_metrics(text),
            confidence=0.9,
            source="metric_analysis",
            timestamp=datetime.now()
        ))

        return insights

    async def _calculate_quality_metrics(
            self,
            text: str
    ) -> Dict[str, Any]:
        """Calculate conversation quality metrics.

        Args:
            text: Text to analyze

        Returns:
            Quality metrics
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Calculate metrics
        avg_sentence_length = (
            sum(len(s.split()) for s in sentences) /
            len(sentences) if sentences else 0
        )

        # Check for question-response pairs
        questions = sum(1 for s in sentences if '?' in s)
        responses = len(sentences) - questions

        # Calculate turn-taking ratio
        turn_ratio = (
            min(questions, responses) /
            max(questions, responses) if max(questions, responses) > 0
            else 0
        )

        return {
            "avg_sentence_length": avg_sentence_length,
            "turn_taking_ratio": turn_ratio,
            "question_count": questions,
            "response_count": responses,
            "total_turns": len(sentences)
        }


# Register analyzers
analyzer_registry.register(AnalysisType.SENTIMENT, SentimentAnalyzer)
analyzer_registry.register(AnalysisType.TOPIC, TopicAnalyzer)
analyzer_registry.register(AnalysisType.QUALITY, QualityAnalyzer)