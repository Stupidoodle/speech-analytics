"""Compliance analysis and cross-analyzer integration."""
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import re
from collections import defaultdict

from src.conversation.manager import ConversationManager
from src.context.types import ContextEntry
from src.events.bus import EventBus

from .types import (
    AnalysisType,
    AnalysisInsight,
    AnalysisPriority,
    AnalysisResult
)
from .registry import BaseAnalyzer, analyzer_registry
from ..events.types import Event


class ComplianceAnalyzer(BaseAnalyzer):
    """Analyzes compliance and regulatory aspects."""

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
        super().__init__(conversation_manager, config)
        self.compliance_rules = config.get('compliance_rules', {})
        self.risk_thresholds = config.get('risk_thresholds', {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        })

    async def analyze(
            self,
            content: Dict[str, Any],
            context: Optional[ContextEntry] = None,
            task_config: Optional[Dict[str, Any]] = None
    ) -> List[AnalysisInsight]:
        """Analyze compliance aspects.

        Args:
            content: Content to analyze
            context: Optional context
            task_config: Optional config

        Returns:
            Compliance insights
        """
        insights = []
        text = content.get("text", "")
        role = content.get("role", "general")

        # Get AI-based compliance analysis
        ai_prompt = (
            "Analyze compliance and regulatory aspects. Consider:\n"
            "1. Data privacy compliance\n"
            "2. Regulatory requirements\n"
            "3. Policy adherence\n"
            "4. Risk assessment\n\n"
            f"Text: {text}\nRole: {role}"
        )

        expected_format = {
            "compliance_status": {
                "overall": "string (compliant/non_compliant/needs_review)",
                "risk_level": "string (high/medium/low)",
                "violations": [
                    {
                        "type": "string",
                        "severity": "float (0-1)",
                        "context": "string",
                        "remediation": "string"
                    }
                ]
            },
            "risk_assessment": {
                "risk_factors": ["list of string"],
                "risk_score": "float (0-1)",
                "mitigation_steps": ["list of string"]
            }
        }

        ai_analysis = await self._get_ai_analysis(
            ai_prompt,
            expected_format
        )

        # Create compliance insight
        insights.append(AnalysisInsight(
            type=AnalysisType.CUSTOM,
            content=ai_analysis,
            confidence=0.8,
            source="ai_analysis",
            timestamp=datetime.now()
        ))

        # Add rule-based compliance checks
        compliance_checks = await self._check_compliance_rules(
            text,
            role
        )
        insights.append(AnalysisInsight(
            type=AnalysisType.CUSTOM,
            content=compliance_checks,
            confidence=0.9,
            source="rule_analysis",
            timestamp=datetime.now()
        ))

        return insights

    async def _check_compliance_rules(
            self,
            text: str,
            role: str
    ) -> Dict[str, Any]:
        """Check compliance against defined rules.

        Args:
            text: Text to check
            role: Role context

        Returns:
            Compliance check results
        """
        results = {
            "role_specific_checks": [],
            "general_checks": [],
            "risk_indicators": []
        }

        # Check role-specific rules
        role_rules = self.compliance_rules.get(role, [])
        for rule in role_rules:
            check_result = await self._apply_rule(text, rule)
            if check_result:
                results["role_specific_checks"].append(check_result)

        # Check general rules
        general_rules = self.compliance_rules.get("general", [])
        for rule in general_rules:
            check_result = await self._apply_rule(text, rule)
            if check_result:
                results["general_checks"].append(check_result)

        # Check risk indicators
        risk_indicators = await self._check_risk_indicators(text)
        results["risk_indicators"] = risk_indicators

        return results

    async def _apply_rule(
            self,
            text: str,
            rule: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Apply compliance rule.

        Args:
            text: Text to check
            rule: Rule definition

        Returns:
            Rule check result if any
        """
        rule_type = rule.get("type")
        pattern = rule.get("pattern")

        if rule_type == "regex":
            matches = re.finditer(pattern, text, re.IGNORECASE)
            violations = [
                {
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                }
                for match in matches
            ]
            if violations:
                return {
                    "rule": rule.get("name"),
                    "severity": rule.get("severity", "medium"),
                    "violations": violations,
                    "remediation": rule.get("remediation")
                }

        elif rule_type == "keyword":
            keywords = set(pattern.split("|"))
            found = set()
            for word in text.lower().split():
                if word in keywords:
                    found.add(word)
            if found:
                return {
                    "rule": rule.get("name"),
                    "severity": rule.get("severity", "medium"),
                    "violations": list(found),
                    "remediation": rule.get("remediation")
                }

        return None

    async def _check_risk_indicators(
            self,
            text: str
    ) -> List[Dict[str, Any]]:
        """Check for risk indicators.

        Args:
            text: Text to check

        Returns:
            List of risk indicators
        """
        indicators = []
        risk_patterns = {
            "pii_exposure": r"\b(?:ssn|passport|credit.?card)\b",
            "confidential": r"\b(?:confidential|classified|restricted)\b",
            "financial": r"\b(?:account.?number|routing.?number)\b",
            "security": r"\b(?:password|credentials|authentication)\b"
        }

        for risk_type, pattern in risk_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append({
                    "type": risk_type,
                    "text": match.group(),
                    "position": match.start(),
                    "severity": self._assess_risk_severity(
                        risk_type,
                        match.group()
                    )
                })

        return indicators

    def _assess_risk_severity(
            self,
            risk_type: str,
            text: str
    ) -> str:
        """Assess risk severity.

        Args:
            risk_type: Type of risk
            text: Risk text

        Returns:
            Risk severity level
        """
        # Implement risk severity assessment logic
        severity_scores = {
            "pii_exposure": 0.9,
            "confidential": 0.8,
            "financial": 0.7,
            "security": 0.6
        }

        base_score = severity_scores.get(risk_type, 0.5)

        # Adjust based on context
        if len(text) > 20:  # More context usually means higher risk
            base_score += 0.1

        # Determine severity level
        if base_score >= self.risk_thresholds["high"]:
            return "high"
        elif base_score >= self.risk_thresholds["medium"]:
            return "medium"
        return "low"


class AnalysisAggregator:
    """Aggregates and correlates analysis results."""

    def __init__(
            self,
            event_bus: EventBus
    ):
        """Initialize aggregator.

        Args:
            event_bus: Event bus instance
        """
        self.event_bus = event_bus
        self.insights: Dict[str, List[AnalysisInsight]] = defaultdict(list)
        self.correlations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.analysis_scores: Dict[str, float] = defaultdict(float)

    async def add_result(
            self,
            result: AnalysisResult
    ) -> None:
        """Add analysis result.

        Args:
            result: Analysis result to add
        """
        # Store insights
        session_id = result.task_id.split('_')[0]
        self.insights[session_id].extend(result.insights)

        # Update analysis scores
        self.analysis_scores[session_id] = (
                                                   self.analysis_scores[session_id] + result.confidence
                                           ) / 2

        # Find correlations
        await self._find_correlations(session_id, result.insights)

        # Emit update event
        await self.event_bus.publish(Event(
            type="analysis_update",
            data={
                "session_id": session_id,
                "scores": self.get_scores(session_id),
                "correlations": self.get_correlations(session_id)
            }
        ))

    async def get_summary(
            self,
            session_id: str
    ) -> Dict[str, Any]:
        """Get analysis summary.

        Args:
            session_id: Session identifier

        Returns:
            Analysis summary
        """
        insights = self.insights.get(session_id, [])
        if not insights:
            return {}

        return {
            "key_insights": self._get_key_insights(insights),
            "scores": self.get_scores(session_id),
            "correlations": self.get_correlations(session_id),
            "recommendations": await self._generate_recommendations(
                session_id
            )
        }

    async def _find_correlations(
            self,
            session_id: str,
            insights: List[AnalysisInsight]
    ) -> None:
        """Find correlations between insights.

        Args:
            session_id: Session identifier
            insights: New insights
        """
        existing = self.insights[session_id]
        for new_insight in insights:
            for existing_insight in existing:
                if new_insight != existing_insight:
                    correlation = await self._correlate_insights(
                        new_insight,
                        existing_insight
                    )
                    if correlation:
                        self.correlations[session_id].append(correlation)

    def get_scores(
            self,
            session_id: str
    ) -> Dict[str, float]:
        """Get analysis scores.

        Args:
            session_id: Session identifier

        Returns:
            Analysis scores
        """
        insights = self.insights.get(session_id, [])
        if not insights:
            return {}

        scores = defaultdict(list)
        for insight in insights:
            scores[insight.type].append(insight.confidence)

        return {
            type_: sum(values) / len(values)
            for type_, values in scores.items()
        }

    def get_correlations(
            self,
            session_id: str
    ) -> List[Dict[str, Any]]:
        """Get insight correlations.

        Args:
            session_id: Session identifier

        Returns:
            Insight correlations
        """
        return self.correlations.get(session_id, [])

    async def _correlate_insights(
            self,
            insight1: AnalysisInsight,
            insight2: AnalysisInsight
    ) -> Optional[Dict[str, Any]]:
        """Correlate two insights.

        Args:
            insight1: First insight
            insight2: Second insight

        Returns:
            Correlation if found
        """
        # Implement correlation logic
        if insight1.type == insight2.type:
            return None  # Skip same-type correlations

        # Look for related content
        common_references = (
            insight1.references.intersection(insight2.references)
        )
        if common_references:
            return {
                "type": "reference_overlap",
                "insights": [insight1.type, insight2.type],
                "references": list(common_references),
                "strength": len(common_references) / len(
                    insight1.references.union(insight2.references)
                )
            }

        return None

    def _get_key_insights(
            self,
            insights: List[AnalysisInsight]
    ) -> List[Dict[str, Any]]:
        """Get key insights.

        Args:
            insights: All insights

        Returns:
            Key insights
        """
        # Filter and sort insights
        key_insights = []
        seen_content = set()

        for insight in sorted(
                insights,
                key=lambda x: x.confidence,
                reverse=True
        ):
            # Skip similar content
            content_hash = hash(str(insight.content))
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)

            key_insights.append({
                "type": insight.type,
                "content": insight.content,
                "confidence": insight.confidence,
                "source": insight.source
            })

        return key_insights[:5]  # Return top 5 insights

    async def _generate_recommendations(
            self,
            session_id: str
    ) -> List[Dict[str, Any]]:
        """Generate recommendations.

        Args:
            session_id: Session identifier

        Returns:
            Recommendations
        """
        insights = self.insights.get(session_id, [])
        if not insights:
            return []

        recommendations = []
        seen = set()

        for insight in insights:
            if isinstance(insight.content, dict):
                recs = insight.content.get("recommendations", [])
                for rec in recs:
                    rec_hash = hash(str(rec))
                    if rec_hash not in seen:
                        recommendations.append({
                            "text": rec,
                            "source": insight.type,
                            "confidence": insight.confidence
                        })
                        seen.add(rec_hash)

        return sorted(
            recommendations,
            key=lambda x: x["confidence"],
            reverse=True
        )


# Register compliance analyzer
analyzer_registry.register(AnalysisType.COMPLIANCE, ComplianceAnalyzer)