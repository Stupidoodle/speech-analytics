from enum import Enum


class Role(Enum):
    INTERVIEWER = "interviewer"
    INTERVIEWEE = "interviewee"
    SUPPORT_AGENT = "support_agent"
    CUSTOMER = "customer"
    MEETING_HOST = "meeting_host"
    MEETING_PARTICIPANT = "meeting_participant"

    def get_prompt_context(self) -> str:
        """Get role-specific prompt context."""
        contexts = {
            Role.INTERVIEWER: (
                "You are an interviewer. Focus on:\n"
                "- Evaluating candidate responses\n"
                "- Identifying areas to probe deeper\n"
                "- Assessing technical and soft skills\n"
                "- Maintaining interview structure\n"
                "- Following up on unclear points"
            ),
            Role.INTERVIEWEE: (
                "You are a job candidate. Focus on:\n"
                "- Highlighting relevant experience\n"
                "- Demonstrating technical knowledge\n"
                "- Showing problem-solving ability\n"
                "- Asking insightful questions\n"
                "- Clarifying job requirements"
            ),
            Role.SUPPORT_AGENT: (
                "You are a support agent. Focus on:\n"
                "- Understanding customer issues\n"
                "- Providing clear solutions\n"
                "- Following up on resolution\n"
                "- Documenting interactions\n"
                "- Escalating when necessary"
            ),
            Role.CUSTOMER: (
                "You are a customer. Focus on:\n"
                "- Describing issues clearly\n"
                "- Providing relevant details\n"
                "- Confirming understanding\n"
                "- Validating solutions\n"
                "- Rating satisfaction"
            ),
            Role.MEETING_HOST: (
                "You are a meeting host. Focus on:\n"
                "- Managing discussion flow\n"
                "- Ensuring participation\n"
                "- Tracking action items\n"
                "- Maintaining schedule\n"
                "- Summarizing key points"
            ),
            Role.MEETING_PARTICIPANT: (
                "You are a meeting participant. Focus on:\n"
                "- Contributing relevant points\n"
                "- Engaging in discussion\n"
                "- Taking notes\n"
                "- Following up on actions\n"
                "- Providing updates"
            )
        }
        return contexts.get(self, "No specific context available.")
