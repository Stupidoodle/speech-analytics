from typing import Dict, Any, AsyncIterator
from datetime import datetime

from ..conversation.manager import ConversationManager
from ..conversation.types import StreamResponse


class ConversationAssistant:
    """Real-time conversation assistant providing suggestions and insights."""

    def __init__(
        self,
        conversation_manager: ConversationManager,
        context_type: str = "general"
    ):
        """Initialize conversation assistant.

        Args:
            conversation_manager: The conversation manager instance
            context_type: Type of conversation (interview, support, meeting)
        """
        self.conversation = conversation_manager
        self.context_type = context_type
        self.insights = []
        self.suggestions = []
        self.action_items = []

    async def get_real_time_suggestion(
        self,
        latest_transcription: str
    ) -> AsyncIterator[StreamResponse]:
        """Get real-time response suggestions.

        Args:
            latest_transcription: Latest transcribed text
        """
        # Create context-aware prompt based on conversation type
        prompt = self._create_suggestion_prompt(latest_transcription)

        async for response in self.conversation.send_message(prompt):
            yield response

    async def analyze_conversation(
        self,
        segment: str
    ) -> Dict[str, Any]:
        """Analyze conversation segment for insights.

        Args:
            segment: Text segment to analyze
        """
        prompt = self._create_analysis_prompt(segment)

        responses = []
        async for response in self.conversation.send_message(prompt):
            if response.text:
                responses.append(response.text)

        analysis = ''.join(responses)
        return self._parse_analysis(analysis)

    def _create_suggestion_prompt(self, latest_text: str) -> str:
        """Create context-appropriate prompt for suggestions."""
        base_prompt = (
            "Based on the conversation context and recent text, "
            "suggest an appropriate response. Latest text:\n\n"
            f"{latest_text}\n\n"
        )

        if self.context_type == "interview":
            return base_prompt + (
                "Consider the candidate's CV and previous responses. "
                "Focus on exploring their experience and qualifications."
            )

        elif self.context_type == "support":
            return base_prompt + (
                "Consider the product documentation and user's issue. "
                "Provide helpful, solution-focused responses."
            )

        elif self.context_type == "meeting":
            return base_prompt + (
                "Consider the agenda and previous discussion points. "
                "Focus on action items and next steps."
            )

        return base_prompt + "Provide a helpful and relevant response."

    def _create_analysis_prompt(self, segment: str) -> str:
        """Create context-appropriate prompt for analysis."""
        base_prompt = (
            "Analyze this conversation segment and provide insights:\n\n"
            f"{segment}\n\n"
            "Please provide:\n"
            "1. Key points discussed\n"
            "2. Action items or tasks\n"
            "3. Important questions or concerns\n"
            "4. Suggested follow-up topics\n"
        )

        if self.context_type == "interview":
            base_prompt += (
                "5. Candidate strengths and qualifications\n"
                "6. Areas needing further exploration\n"
            )

        elif self.context_type == "support":
            base_prompt += (
                "5. Issue identification and status\n"
                "6. Relevant documentation or resources\n"
            )

        elif self.context_type == "meeting":
            base_prompt += (
                "5. Progress on agenda items\n"
                "6. Decisions made\n"
            )

        return base_prompt

    def _parse_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse analysis response into structured data."""
        try:
            # sections = analysis.split("\n\n")
            result = {
                'key_points': [],
                'action_items': [],
                'questions': [],
                'follow_up_topics': [],
                'context_specific': {},
                'timestamp': datetime.now().isoformat()
            }

            current_section = None
            for line in analysis.split('\n'):
                line = line.strip()
                if line.startswith('1.'):
                    current_section = 'key_points'
                elif line.startswith('2.'):
                    current_section = 'action_items'
                elif line.startswith('3.'):
                    current_section = 'questions'
                elif line.startswith('4.'):
                    current_section = 'follow_up_topics'
                elif line.startswith('5.') or line.startswith('6.'):
                    current_section = 'context_specific'
                elif line and current_section:
                    if current_section == 'context_specific':
                        key, value = line.split(':', 1) if ':'\
                            in line else (line, '')
                        result[current_section][key.strip()] = value.strip()
                    else:
                        result[current_section].append(line)

            return result
        except Exception as e:
            print(f"Error parsing analysis: {e}")
            return {
                'error': str(e),
                'raw_analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
