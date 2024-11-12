from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseAnalyzer
from ...conversation.manager import ConversationManager
from ...assistance.types import SupportAnalysis, ActionItem


class SupportAnalyzer(BaseAnalyzer):
    """Analyzer for customer support conversations."""

    def __init__(self, conversation_manager: ConversationManager):
        super().__init__(conversation_manager)
        self.product_docs: Dict[str, Any] = {}
        self.knowledge_base: Dict[str, Any] = {}
        self.issue_history = {}

    async def set_product_docs(self, docs: Dict[str, Any]) -> None:
        """Set product documentation context."""
        self.product_docs = docs

    async def set_knowledge_base(self, kb: Dict[str, Any]) -> None:
        """Set knowledge base for reference."""
        self.knowledge_base = kb

    async def add_issue_history(self, history: Dict[str, Any]) -> None:
        """Add customer issue history."""
        self.issue_history.update(history)

    async def analyze_segment(
        self,
        segment: str,
        timestamp: Optional[datetime] = None
    ) -> SupportAnalysis:
        """Analyze a support conversation segment."""
        prompt = self._create_analysis_prompt(segment)

        responses = []
        async for response in self.conversation.send_message(prompt):
            if response.text:
                responses.append(response.text)

        analysis = ''.join(responses)
        return self._parse_analysis(analysis, timestamp or datetime.now())

    async def get_solution_suggestions(self,
                                       issue_context: str
                                       ) -> List[Dict[str, Any]]:
        """Generate relevant solution suggestions."""
        prompt = (
            "Based on the customer's issue and our documentation, "
            "suggest solutions. Context:\n\n"
            f"{issue_context}\n\n"
            "Consider:\n"
            "1. Known solutions from documentation\n"
            "2. Similar past issues\n"
            "3. Workarounds\n"
            "4. Escalation paths if needed"
        )

        responses = []
        async for response in self.conversation.send_message(prompt):
            if response.text:
                responses.append(response.text)

        analysis = ''.join(responses)
        return self._parse_solutions(analysis)

    async def get_relevant_docs(self, context: str) -> List[Dict[str, Any]]:
        """Find relevant documentation based on context."""
        prompt = (
            "Find relevant documentation sections for this context:\n\n"
            f"{context}\n\n"
            "Available documentation:\n"
            f"{list(self.product_docs.keys())}\n\n"
            "Provide section references with relevance scores."
        )

        responses = []
        async for response in self.conversation.send_message(prompt):
            if response.text:
                responses.append(response.text)

        analysis = ''.join(responses)
        return self._parse_doc_references(analysis)

    def _create_analysis_prompt(self, segment: str) -> str:
        """Create context-aware analysis prompt."""
        return (
            "Analyze this support conversation segment:\n\n"
            f"{segment}\n\n"
            "Provide:\n"
            "1. Issue identification\n"
            "2. Customer's main concerns\n"
            "3. Technical details mentioned\n"
            "4. Current issue status\n"
            "5. Solution attempts made\n"
            "6. Customer sentiment\n"
            "7. Suggested next steps\n"
            "8. Documentation references needed"
        )

    def _parse_analysis(
        self,
        analysis: str,
        timestamp: datetime
    ) -> SupportAnalysis:
        """Parse analysis response into structured data."""
        sections = {}
        current_section = None

        for line in analysis.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line[0].isdigit() and '.' in line[:3]:
                current_section = line.split('.', 1)[1].strip().lower()
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

        # Extract issue status and progress
        issue_status = self._determine_issue_status(sections)
        solution_progress = self._calculate_solution_progress(sections)

        # Find relevant documentation
        relevant_docs = self._find_relevant_docs(analysis)

        return SupportAnalysis(
            key_points=sections.get('issue identification', []),
            action_items=self._parse_action_items(
                sections.get('suggested next steps', [])
            ),
            questions=self._extract_questions(analysis),
            follow_up_topics=self._extract_follow_up_topics(sections),
            context_specific={
                'technical_details': sections.get(
                    'technical details mentioned',
                    []
                    ),
                'solution_attempts': sections.get(
                    'solution attempts made',
                    []
                    ),
                'customer_concerns': sections.get(
                    "customer's main concerns",
                    []
                    )
            },
            issue_type=self._determine_issue_type(sections),
            issue_status=issue_status,
            relevant_docs=relevant_docs,
            solution_progress=solution_progress,
            customer_sentiment=self._analyze_sentiment(
                sections.get('customer sentiment', [])
            ),
            timestamp=timestamp
        )

    def _parse_action_items(self, items: List[str]) -> List[ActionItem]:
        """Convert next steps to action items."""
        action_items = []
        for item in items:
            # Determine if this is customer or support action
            assignee = "support"
            if any(word in item.lower()
                   for word in ['customer', 'user', 'client', 'they']):
                assignee = "customer"

            # Determine priority based on keywords
            priority = "medium"
            if any(word in item.lower()
                   for word in ['urgent', 'critical', 'immediate']):
                priority = "high"
            elif any(word in item.lower()
                     for word in ['later', 'eventually', 'when possible']):
                priority = "low"

            action_items.append(
                ActionItem(
                    description=item,
                    assignee=assignee,
                    deadline=None,
                    status="pending",
                    priority=priority,
                    context={
                        'type': 'support_action',
                        'category': 'resolution_step'
                    },
                    timestamp=datetime.now()
                )
            )
        return action_items

    def _determine_issue_status(
        self,
        sections: Dict[str, List[str]]
    ) -> str:
        """Determine current issue status."""
        status_section = sections.get('current issue status', [])

        if not status_section:
            return "unknown"

        status_text = ' '.join(status_section).lower()

        if any(word in status_text
               for word in ['resolved', 'fixed', 'completed']):
            return "resolved"
        elif any(word in status_text
                 for word in ['in progress', 'working', 'investigating']):
            return "in_progress"
        elif any(word in status_text
                 for word in ['blocked', 'waiting', 'pending']):
            return "blocked"
        elif any(word in status_text
                 for word in ['new', 'initial', 'started']):
            return "new"

        return "unknown"

    def _calculate_solution_progress(
        self,
        sections: Dict[str, List[str]]
    ) -> float:
        """Calculate solution progress as percentage."""
        status = self._determine_issue_status(sections)
        attempts = len(sections.get('solution attempts made', []))
        next_steps = len(sections.get('suggested next steps', []))

        if status == "resolved":
            return 1.0
        elif status == "new":
            return 0.0

        # Calculate progress based on attempts and remaining steps
        if attempts + next_steps > 0:
            progress = attempts / (attempts + next_steps)
            return min(max(progress, 0.0), 0.9)  # Cap at 90% if not resolved

        return 0.0

    def _determine_issue_type(
        self,
        sections: Dict[str, List[str]]
    ) -> str:
        """Determine the type of issue being discussed."""
        issue_section = sections.get('issue identification', [])
        if not issue_section:
            return "unknown"

        issue_text = ' '.join(issue_section).lower()

        # Match against common issue types
        if any(word in issue_text
               for word in ['error', 'bug', 'crash', 'failed']):
            return "technical_error"
        elif any(word in issue_text
                 for word in ['how to', 'help with', 'guide']):
            return "how_to"
        elif any(word in issue_text
                 for word in ['account', 'login', 'password']):
            return "account_access"
        elif any(word in issue_text
                 for word in ['slow', 'performance', 'speed']):
            return "performance"
        elif any(word in issue_text
                 for word in ['feature', 'request', 'enhancement']):
            return "feature_request"

        return "other"

    def _analyze_sentiment(self, sentiment_lines: List[str]) -> str:
        """Analyze customer sentiment from sentiment section."""
        if not sentiment_lines:
            return "neutral"

        sentiment_text = ' '.join(sentiment_lines).lower()

        # Count sentiment indicators
        positive_indicators = sum(
            1 for word in ['happy', 'satisfied', 'pleased', 'grateful', 'good']
            if word in sentiment_text
        )
        negative_indicators = sum(
            1 for word in ['frustrated', 'angry', 'upset', 'unhappy', 'bad']
            if word in sentiment_text
        )

        if positive_indicators > negative_indicators:
            return "positive"
        elif negative_indicators > positive_indicators:
            return "negative"
        return "neutral"

    def _find_relevant_docs(self, analysis: str) -> List[Dict[str, Any]]:
        """Find documentation references in analysis."""
        relevant_docs = []
        doc_references = set()

        # Extract documentation references
        for doc_name, content in self.product_docs.items():
            if doc_name.lower() in analysis.lower():
                doc_references.add(doc_name)

        # Create doc reference entries
        for doc_name in doc_references:
            relevant_docs.append({
                'name': doc_name,
                'content': self.product_docs[doc_name],
                # Could be enhanced with similarity scoring
                'relevance': 'high',
                'timestamp': datetime.now().isoformat()
            })

        return relevant_docs

    def _extract_questions(self, analysis: str) -> List[str]:
        """Extract questions from analysis."""
        questions = []
        for line in analysis.split('\n'):
            if '?' in line:
                questions.append(line.strip())
        return questions

    def _extract_follow_up_topics(
        self,
        sections: Dict[str, List[str]]
    ) -> List[str]:
        """Extract topics needing follow-up."""
        topics = set()

        # Add topics from relevant sections
        for section in ['suggested next steps',
                        'documentation references needed'
                        ]:
            for item in sections.get(section, []):
                words = item.split()
                if words:
                    topics.add(words[0])

        return list(topics)

    def _parse_solutions(self, analysis: str) -> List[Dict[str, Any]]:
        """Parse solution suggestions into structured format."""
        solutions = []
        current_solution = None

        for line in analysis.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line[0].isdigit() and '.' in line[:3]:
                if current_solution:
                    solutions.append(current_solution)
                current_solution = {
                    'description': line.split('.', 1)[1].strip(),
                    'steps': [],
                    'doc_references': [],
                    'complexity': 'medium',
                    'timestamp': datetime.now().isoformat()
                }
            elif current_solution:
                current_solution['steps'].append(line)

        if current_solution:
            solutions.append(current_solution)

        return solutions

    def _parse_doc_references(
        self,
        analysis: str
    ) -> List[Dict[str, Any]]:
        """Parse documentation references into structured format."""
        references = []
        current_ref = None

        for line in analysis.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line[0].isdigit() and '.' in line[:3]:
                if current_ref:
                    references.append(current_ref)
                current_ref = {
                    'section': line.split('.', 1)[1].strip(),
                    'relevance_score': 0.0,
                    'content': '',
                    'timestamp': datetime.now().isoformat()
                }
            elif current_ref:
                if 'relevance:' in line.lower():
                    try:
                        score = float(
                            line.split(':')[1].strip().strip('%')
                            ) / 100
                        current_ref['relevance_score'] = score
                    except (ValueError, IndexError):
                        pass
                else:
                    current_ref['content'] += line + '\n'

        if current_ref:
            references.append(current_ref)

        return references
