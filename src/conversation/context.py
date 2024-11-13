import aiofiles
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from .types import (
    Role,
    Document,
    DocumentFormat,
)

from .exceptions import (
    DocumentError,
)


class ConversationContext:
    """Manages conversation context, documents, and role-specific behavior."""

    def __init__(self, role: Role) -> None:
        """Initialize conversation context.

        Args:
            role: User's role in conversation
        """
        self.role = role
        self.documents: Dict[str, Document] = {}
        self.system_prompts: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

        # Add role-specific system prompt
        self._add_role_prompt()

    def _add_role_prompt(self) -> None:
        """Add role-specific system prompt and behaviors."""

        prompts = {
            Role.INTERVIEWER: """You are conducting an interview. Focus on:
            - Evaluating candidate responses against requirements
            - Asking targeted follow-up questions
            - Assessing technical and soft skills
            - Maintaining structured interview flow
            - Documenting key insights and red flags
            - Referencing CV and job requirements
            - Ensuring all key areas are covered
            - Providing clear transitions between topics""",
            Role.INTERVIEWEE: """You are a job candidate in an interview. Focus on:
            - Providing clear, structured responses
            - Using the STAR method for experience examples
            - Demonstrating relevant skills and experience
            - Showing enthusiasm and cultural fit
            - Asking insightful questions about the role
            - Connecting experience to job requirements
            - Highlighting unique value proposition
            - Maintaining professional communication
            - Following up on interviewer's questions""",
            Role.SUPPORT_AGENT: """You are providing customer support. Focus on:
            - Understanding customer issues thoroughly
            - Providing clear, step-by-step solutions
            - Maintaining empathetic communication
            - Following established procedures
            - Documenting issue resolution
            - Escalating when appropriate
            - Referencing relevant documentation
            - Ensuring customer satisfaction
            - Following up on resolution""",
            Role.CUSTOMER: """You are seeking customer support. Focus on:
            - Clearly describing your issue or need
            - Providing relevant context and details
            - Following troubleshooting instructions
            - Asking clarifying questions
            - Confirming understanding
            - Expressing concerns clearly
            - Testing proposed solutions
            - Providing feedback on resolution""",
            Role.MEETING_HOST: """You are hosting a meeting. Focus on:
            - Managing meeting flow and timing
            - Ensuring participant engagement
            - Keeping discussion on track
            - Documenting key decisions
            - Assigning action items
            - Facilitating effective discussion
            - Managing time for each agenda item
            - Summarizing key points
            - Ensuring clear next steps
            - Following up on action items""",
            Role.MEETING_PARTICIPANT: """You are participating in a meeting. Focus on:
            - Contributing relevant insights
            - Staying on topic
            - Respecting time allocations
            - Taking notes on key points
            - Volunteering for action items
            - Asking clarifying questions
            - Supporting meeting objectives
            - Following up on commitments
            - Engaging constructively"""
        }

        # Add base role prompt
        if self.role in prompts:
            self.system_prompts.append({
                "text": prompts[self.role],
                "metadata": {"type": "role_prompt"}
            })

        # Add role-specific behaviors
        behaviors = self._get_role_behaviors()
        if behaviors:
            self.system_prompts.append({
                "text": behaviors,
                "metadata": {"type": "role_behaviors"}
            })

    def _get_role_behaviors(self) -> Optional[str]:
        """Get role-specific behaviors and guidelines."""
        behaviors = {
            Role.INTERVIEWER: """Behavioral Guidelines:
            1. Time Management:
               - Keep questions focused and relevant
               - Allow adequate response time
               - Maintain interview schedule
        
            2. Assessment:
               - Use behavioral questioning techniques
               - Look for specific examples
               - Compare against job requirements
               - Document candidate responses
        
            3. Communication:
               - Use clear, professional language
               - Provide smooth transitions
               - Give clear instructions
               - Maintain neutral tone
        
            4. Documentation:
               - Note key qualifications
               - Record red flags
               - Track required skill coverage
               - Document follow-up items""",
            Role.INTERVIEWEE: """Behavioral Guidelines:
            1. Response Structure:
               - Use STAR method (Situation, Task, Action, Result)
               - Provide specific examples
               - Keep responses focused
               - Highlight achievements
        
            2. Professional Conduct:
               - Maintain positive attitude
               - Show active listening
               - Express genuine interest
               - Demonstrate preparation
        
            3. Question Handling:
               - Ask for clarification if needed
               - Take time to formulate responses
               - Provide relevant context
               - Connect answers to role
        
            4. Follow-up:
               - Note areas for elaboration
               - Remember questions to ask
               - Track discussion points
               - Show initiative""",
            Role.SUPPORT_AGENT: """Behavioral Guidelines:
            1. Issue Resolution:
               - Follow troubleshooting steps
               - Document all actions taken
               - Verify resolution
               - Set clear expectations
        
            2. Communication:
               - Use clear, simple language
               - Show empathy and patience
               - Provide progress updates
               - Confirm understanding
        
            3. Documentation:
               - Record all interactions
               - Note solution steps
               - Track issue status
               - Document follow-up needs
        
            4. Escalation:
               - Know escalation criteria
               - Follow escalation procedures
               - Keep customer informed
               - Track escalated issues""",
            Role.CUSTOMER: """Behavioral Guidelines:
            1. Issue Reporting:
               - Describe problem clearly
               - Provide relevant details
               - Follow instructions
               - Report outcomes
        
            2. Communication:
               - Stay focused on issue
               - Ask for clarification
               - Provide feedback
               - Maintain patience
        
            3. Solution Testing:
               - Follow steps exactly
               - Report results clearly
               - Note any issues
               - Confirm resolution
        
            4. Follow-up:
               - Document solution
               - Track issue status
               - Keep reference numbers
               - Note contact points""",
            Role.MEETING_HOST: """Behavioral Guidelines:
            1. Meeting Management:
               - Follow agenda strictly
               - Manage time effectively
               - Ensure participation
               - Document decisions
        
            2. Facilitation:
               - Encourage discussion
               - Manage conflicts
               - Keep focus on goals
               - Drive to outcomes
        
            3. Documentation:
               - Record key decisions
               - Track action items
               - Note agreements
               - Capture next steps
        
            4. Follow-up:
               - Distribute minutes
               - Track action items
               - Schedule follow-ups
               - Monitor progress""",
            Role.MEETING_PARTICIPANT: """Behavioral Guidelines:
            1. Participation:
               - Come prepared
               - Stay engaged
               - Contribute constructively
               - Respect time limits
        
            2. Communication:
               - Be clear and concise
               - Listen actively
               - Support discussion flow
               - Ask relevant questions
        
            3. Documentation:
               - Take personal notes
               - Track commitments
               - Record decisions
               - Note action items
        
            4. Follow-up:
               - Complete assigned tasks
               - Report progress
               - Share relevant updates
               - Meet deadlines"""
        }

        return behaviors.get(self.role)

    async def add_document(
            self,
            path: str,
            doc_type: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Add document to context.

        Args:
            path: Path to document
            doc_type: Type of document (e.g., 'cv', 'spec')
            metadata: Optional document metadata

        Returns:
            Processed document

        Raises:
            DocumentError: On document processing errors
        """
        try:
            # Read document
            doc_path = Path(path)
            if not doc_path.exists():
                raise DocumentError(f"Document not found: {path}")

            # Determine format
            format_map = {
                ".pdf": DocumentFormat.PDF,
                ".csv": DocumentFormat.CSV,
                ".doc": DocumentFormat.DOC,
                ".docx": DocumentFormat.DOCX,
                ".txt": DocumentFormat.TXT,
                ".md": DocumentFormat.MD
            }
            doc_format = format_map.get(doc_path.suffix.lower())
            if not doc_format:
                raise DocumentError(f"Unsupported format: {doc_path.suffix}")

            # Read content
            content = await self._read_file(doc_path)

            # Create document
            document = Document(
                content=content,
                mime_type=doc_format.value,
                name=doc_path.name,
                metadata=metadata or {}
            )

            # Store document
            doc_id = f"{doc_type}_{datetime.now().isoformat()}"
            self.documents[doc_id] = document

            # Add document prompt based on type
            await self._add_document_prompt(document, doc_type)

            return document

        except Exception as e:
            raise DocumentError(f"Failed to add document: {str(e)}")

    async def _read_file(self, path: Path) -> bytes:
        """Read file content.

        Args:
            path: File path

        Returns:
            File content as bytes
        """
        try:
            async with aiofiles.open(path, mode="rb") as file:
                content = await file.read()
            return content
        except Exception as e:
            raise DocumentError(f"Failed to read file: {str(e)}")

    async def _add_document_prompt(
            self,
            document: Document,
            doc_type: str
    ) -> None:
        """Add document-specific system prompt based on role."""
        prompts = {
            # Interviewer Documents
            ("cv", Role.INTERVIEWER): {
                "text": """Review the candidate's CV focusing on:
- Relevant experience and skills
- Career progression
- Technical expertise
- Project achievements
- Education and certifications
Use this information to:
- Ask targeted questions
- Validate claimed experience
- Assess skill levels
- Identify areas for deeper exploration""",
                "importance": "high"
            },
            ("job_description", Role.INTERVIEWER): {
                "text": """Use this job description to:
- Assess candidate fit against requirements
- Focus questions on key skills needed
- Evaluate technical competencies
- Gauge culture fit
- Verify experience levels""",
                "importance": "high"
            },

            # Interviewee Documents
            ("job_description", Role.INTERVIEWEE): {
                "text": """Review the job description to:
- Align responses with requirements
- Highlight relevant experience
- Demonstrate understanding of role
- Show enthusiasm for responsibilities
- Connect skills to needs""",
                "importance": "high"
            },
            ("company_info", Role.INTERVIEWEE): {
                "text": """Use company information to:
- Show company research
- Align with culture
- Ask informed questions
- Demonstrate interest""",
                "importance": "medium"
            },

            # Support Agent Documents
            ("manual", Role.SUPPORT_AGENT): {
                "text": """Reference this manual for:
- Standard procedures
- Troubleshooting steps
- Technical specifications
- Solution guidelines
- Escalation criteria""",
                "importance": "high"
            },
            ("ticket_history", Role.SUPPORT_AGENT): {
                "text": """Review ticket history to:
- Understand issue context
- Check previous solutions
- Avoid duplicate efforts
- Ensure consistent support""",
                "importance": "medium"
            },

            # Customer Documents
            ("product_manual", Role.CUSTOMER): {
                "text": """Reference product documentation to:
- Understand features
- Follow instructions
- Verify specifications
- Check requirements""",
                "importance": "medium"
            },

            # Meeting Host Documents
            ("agenda", Role.MEETING_HOST): {
                "text": """Use this agenda to:
- Guide discussion flow
- Manage time allocation
- Track completion
- Ensure coverage
- Document decisions""",
                "importance": "high"
            },
            ("previous_minutes", Role.MEETING_HOST): {
                "text": """Review previous minutes to:
- Follow up on actions
- Maintain continuity
- Track progress
- Update status""",
                "importance": "medium"
            },

            # Meeting Participant Documents
            ("meeting_materials", Role.MEETING_PARTICIPANT): {
                "text": """Review materials to:
- Prepare contributions
- Support discussion
- Reference data
- Track decisions""",
                "importance": "high"
            }
        }

        # Get prompt for document type and role
        prompt_info = prompts.get((doc_type, self.role))
        if prompt_info:
            self.system_prompts.append({
                "text": prompt_info["text"],
                "document": {
                    "format": document.mime_type,
                    "name": document.name,
                    "source": {"bytes": document.content}
                },
                "metadata": {
                    "type": "document_prompt",
                    "doc_type": doc_type,
                    "importance": prompt_info["importance"]
                }
            })

    async def clear_context(
            self,
            context_type: Optional[str] = None
    ) -> None:
        """Clear specified context type.

        Args:
            context_type: Type of context to clear ('documents', 'prompts', or None for all)
        """
        if context_type == "documents":
            self.documents.clear()
        elif context_type == "prompts":
            self.system_prompts = []
            self._add_role_prompt()  # Keep role prompt
        elif context_type is None:
            self.documents.clear()
            self.system_prompts.clear()
            self.metadata.clear()
            self._add_role_prompt()  # Keep role prompt
