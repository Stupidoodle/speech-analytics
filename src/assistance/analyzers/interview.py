from typing import Dict, Any, List, Optional
from datetime import datetime

from ..types import InterviewAnalysis, ActionItem
from ...conversation.manager import ConversationManager


class InterviewAnalyzer:
    """Analyzer for interview conversations."""

    def __init__(self, conversation_manager: ConversationManager):
        self.conversation = conversation_manager
        self.cv_context = {}
        self.job_requirements = {}
        self.qualification_matches = {}

    async def set_cv_context(self, cv_data: Dict[str, Any]) -> None:
        """Set CV context for analysis."""
        self.cv_context = cv_data
        await self._update_qualification_matches()

    async def set_job_requirements(self, requirements: Dict[str, Any]) -> None:
        """Set job requirements for matching."""
        self.job_requirements = requirements
        await self._update_qualification_matches()

    async def analyze_segment(
        self,
        segment: str,
        timestamp: Optional[datetime] = None
    ) -> InterviewAnalysis:
        """Analyze an interview conversation segment."""
        # Create analysis prompt
        prompt = self._create_analysis_prompt(segment)

        # Get model response
        responses = []
        async for response in self.conversation.send_message(prompt):
            if response.text:
                responses.append(response.text)

        # Parse the analysis
        analysis = ''.join(responses)
        return self._parse_analysis(analysis, timestamp or datetime.now())

    async def get_next_questions(self, context: str) -> List[str]:
        """Generate relevant follow-up questions."""
        prompt = (
            "Based on the conversation so far and candidate's profile, "
            "suggest relevant follow-up questions. Context:\n\n"
            f"{context}\n\n"
            "Consider:\n"
            "1. Areas not yet covered\n"
            "2. Points needing clarification\n"
            "3. Technical expertise validation\n"
            "4. Behavioral assessments"
        )

        responses = []
        questions = []
        async for response in self.conversation.send_message(prompt):
            if response.text:
                responses.append(response.text)

        analysis = ''.join(responses)
        for line in analysis.split('\n'):
            if '?' in line:
                questions.append(line.strip())

        return questions

    async def _update_qualification_matches(self) -> None:
        """Update qualification matches between CV and job requirements."""
        if not self.cv_context or not self.job_requirements:
            return

        prompt = (
            "Compare the candidate's qualifications with job requirements.\n\n"
            f"CV:\n{self.cv_context}\n\n"
            f"Requirements:\n{self.job_requirements}\n\n"
            "Provide a detailed match analysis with confidence scores."
        )

        responses = []
        async for response in self.conversation.send_message(prompt):
            if response.text:
                responses.append(response.text)

        analysis = ''.join(responses)
        self.qualification_matches = self._parse_qualification_matches(
            analysis
            )

    def _create_analysis_prompt(self, segment: str) -> str:
        """Create context-aware analysis prompt."""
        return (
            "Analyze this interview segment considering the candidate's "
            "profile and job requirements:\n\n"
            f"{segment}\n\n"
            "Provide:\n"
            "1. Key points discussed\n"
            "2. Candidate's strengths demonstrated\n"
            "3. Areas needing further exploration\n"
            "4. Technical expertise demonstrated\n"
            "5. Behavioral insights\n"
            "6. Red flags or concerns (if any)\n"
            "7. Suggested follow-up questions\n"
            "8. Overall segment assessment"
        )

    def _parse_analysis(
        self,
        analysis: str,
        timestamp: datetime
    ) -> InterviewAnalysis:
        """Parse analysis response into structured data."""
        # Split analysis into sections
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

        # Create structured analysis
        return InterviewAnalysis(
            key_points=sections.get('key points discussed', []),
            action_items=self._parse_action_items(
                sections.get('areas needing further exploration', [])
            ),
            questions=sections.get('suggested follow-up questions', []),
            follow_up_topics=self._extract_follow_up_topics(sections),
            context_specific={
                'technical_expertise': sections.get(
                    'technical expertise demonstrated',
                    []
                ),
                'behavioral_insights': sections.get('behavioral insights', []),
                'red_flags': sections.get('red flags or concerns', [])
            },
            candidate_strengths=sections.get(
                "candidate's strengths demonstrated",
                []
                ),
            areas_to_explore=sections.get(
                'areas needing further exploration',
                []
                ),
            qualification_matches=self.qualification_matches,
            cv_references=self._find_cv_references(analysis),
            timestamp=timestamp
        )

    def _parse_action_items(self, items: List[str]) -> List[ActionItem]:
        """Convert exploration areas to action items."""
        action_items = []
        for item in items:
            action_items.append(
                ActionItem(
                    description=item,
                    assignee="interviewer",
                    deadline=None,
                    status="pending",
                    priority="medium",
                    context={
                        'type': 'interview_follow_up',
                        'category': 'exploration'
                    },
                    timestamp=datetime.now()
                )
            )
        return action_items

    def _extract_follow_up_topics(
        self,
        sections: Dict[str, List[str]]
    ) -> List[str]:
        """Extract topics needing follow-up from various sections."""
        topics = set()

        # Add topics from relevant sections
        for section in ['areas needing further exploration',
                        'suggested follow-up questions']:
            for item in sections.get(section, []):
                # Extract main topic from item
                topic = item.split()[0] if item else ''
                if topic:
                    topics.add(topic)

        return list(topics)

    def _find_cv_references(self, analysis: str) -> List[Dict[str, Any]]:
        """Find references to CV content in the analysis."""
        references = []
        cv_content = str(self.cv_context)

        # Simple matching for now - could be enhanced with NLP
        for line in analysis.split('\n'):
            for cv_line in cv_content.split('\n'):
                if cv_line in line:
                    references.append({
                        'cv_content': cv_line,
                        'analysis_context': line,
                        'timestamp': datetime.now().isoformat()
                    })

        return references

    def _parse_qualification_matches(
        self,
        analysis: str
    ) -> Dict[str, float]:
        """Parse qualification matching analysis into scores."""
        matches = {}
        for line in analysis.split('\n'):
            if ':' in line:
                skill, score_text = line.split(':', 1)
                try:
                    # Extract percentage or score from text
                    score = float(
                        ''.join(c for c in score_text
                                if c.isdigit() or c == '.')
                    )
                    matches[skill.strip()] = min(score / 100, 1.0)
                except (ValueError, TypeError):
                    continue
        return matches
